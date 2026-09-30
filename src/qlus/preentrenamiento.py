"""Preentrenamiento autosupervisado con clips SIN anotar (reconstrucción enmascarada).

Sin máscaras no se aprende a segmentar, pero sí cómo se ven en clips reales la
pleura, las costillas y sus sombras, las líneas A y B, el pulmón blanco o una
consolidación. La U-Net recibe los frames t−1, t y t+1 con parches borrados y
debe reconstruir el frame t (estilo "masked autoencoder"):

- modo espacial: los mismos parches borrados en los tres frames, lo que obliga a
  usar el contexto de la imagen;
- modo temporal (1 de cada 3 ejemplos): parches borrados solo en el frame t, lo
  que obliga a usar los frames vecinos (deslizamiento pleural, líneas B que se
  mueven). Es la tarea más fácil, por eso es la minoritaria.

La pérdida (L1) se mide solo en lo borrado. Se usa la misma U-Net que en la
segmentación, con 1 canal de salida: todo salvo la capa final se reutiliza luego
con `entrenar(..., inicial=...)`.

Memoria acotada: en cada época se cargan `clips_en_memoria` clips al azar, así
una biblioteca de miles de clips no necesita caber en la RAM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .biblioteca import escribir_png
from .entrenamiento import aumentar, paciente_de, particionar, pila_temporal
from .io import cargar_npz
from .modelos import UNet

TIPO = "preentrenamiento_reconstruccion"


def mascara_parches(rng: np.random.Generator, n: int, parche: int, frac: float) -> np.ndarray:
    k = -(-n // parche)
    m = rng.random((k, k)) < frac
    return np.kron(m, np.ones((parche, parche), bool))[:n, :n]


def enmascarar(x: np.ndarray, rng: np.random.Generator, temporal: bool, parche: int = 16,
               frac_espacial: float = 0.5, frac_temporal: float = 0.75):
    """x (3, n, n) en 0–1 → (entrada, objetivo = frame t, máscara de lo borrado)."""
    m = mascara_parches(rng, x.shape[-1], parche, frac_temporal if temporal else frac_espacial)
    entrada = x.copy()
    relleno = float(x.mean())
    if temporal:
        entrada[1][m] = relleno
    else:
        entrada[:, m] = relleno
    return entrada, x[1].copy(), m


def _a_tamano(x: np.ndarray, n: int) -> np.ndarray:
    return np.stack([cv2.resize(ch, (n, n), interpolation=cv2.INTER_LINEAR) for ch in x]) / 255.0


class ConjuntoSinAnotar(torch.utils.data.Dataset):
    def __init__(self, rutas: list[Path], tamano: int, muestras: int, clips_en_memoria: int, semilla: int = 0):
        self.rutas, self.tamano, self.muestras = rutas, tamano, muestras
        self.clips_en_memoria = clips_en_memoria
        self.rng = np.random.default_rng(semilla)
        self.renovar()

    def renovar(self):
        """Carga otro subconjunto al azar de clips (se llama al inicio de cada época)."""
        k = min(self.clips_en_memoria, len(self.rutas))
        self.clips = [cargar_npz(self.rutas[i]).frames for i in self.rng.choice(len(self.rutas), k, replace=False)]

    def __len__(self):
        return self.muestras

    def __getitem__(self, _):
        fr = self.clips[self.rng.integers(len(self.clips))]
        x, _ = aumentar(pila_temporal(fr, int(self.rng.integers(fr.shape[0]))).astype(np.float32), None, self.rng)
        temporal = fr.shape[0] > 1 and self.rng.random() < 1 / 3
        entrada, objetivo, m = enmascarar(_a_tamano(x, self.tamano).astype(np.float32), self.rng, temporal)
        return torch.from_numpy(entrada), torch.from_numpy(objetivo), torch.from_numpy(m)


def conjunto_validacion(rutas: list[Path], tamano: int, semilla: int, max_clips: int = 50,
                        por_clip: int = 4) -> tuple[torch.Tensor, ...]:
    """Ejemplos fijos (sin aumentos, máscaras con semilla) para que la pérdida sea comparable entre épocas."""
    rng = np.random.default_rng(semilla + 1)
    ent, obj, mas = [], [], []
    for r in rutas[:max_clips]:
        fr = cargar_npz(r).frames
        for j, t in enumerate(np.linspace(0, fr.shape[0] - 1, min(por_clip, fr.shape[0])).astype(int)):
            x = _a_tamano(pila_temporal(fr, int(t)).astype(np.float32), tamano).astype(np.float32)
            e, o, m = enmascarar(x, rng, temporal=fr.shape[0] > 1 and j % 3 == 2)
            ent.append(e), obj.append(o), mas.append(m)
    return torch.from_numpy(np.stack(ent)), torch.from_numpy(np.stack(obj)), torch.from_numpy(np.stack(mas))


def perdida_enmascarada(pred: torch.Tensor, objetivo: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    return ((pred - objetivo).abs() * m).sum() / m.sum().clamp(min=1.0)


def reconstruir(modelo: nn.Module, entrada: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(modelo(entrada))[:, 0]


@torch.no_grad()
def evaluar(modelo: nn.Module, val: tuple[torch.Tensor, ...], disp, lote: int = 16) -> float:
    modelo.eval()
    ent, obj, mas = val
    suma = n = 0.0
    for i in range(0, len(ent), lote):
        m = mas[i:i + lote].to(disp)
        pred = reconstruir(modelo, ent[i:i + lote].to(disp))
        suma += float(((pred - obj[i:i + lote].to(disp)).abs() * m).sum())
        n += float(m.sum())
    return suma / max(n, 1.0)


@torch.no_grad()
def lamina_ejemplos(modelo: nn.Module, val: tuple[torch.Tensor, ...], disp, n: int = 4) -> np.ndarray:
    """Filas: frame t con parches borrados | reconstrucción | original."""
    modelo.eval()
    idx = torch.from_numpy(np.linspace(0, len(val[0]) - 1, min(n, len(val[0]))).astype(np.int64))
    ent, obj = val[0][idx], val[1][idx]
    pred = reconstruir(modelo, ent.to(disp)).cpu()
    sep = np.full((ent.shape[-1], 3), 255, np.uint8)
    filas = [np.hstack([(ent[i, 1].numpy() * 255).astype(np.uint8), sep,
                        (pred[i].numpy() * 255).astype(np.uint8), sep,
                        (obj[i].numpy() * 255).astype(np.uint8)]) for i in range(len(ent))]
    return np.vstack([np.vstack([f, np.full((3, f.shape[1]), 255, np.uint8)]) for f in filas])


def preentrenar(carpeta: str | Path, salida: str | Path, epocas: int = 30, tamano: int = 256, lote: int = 8,
                base: int = 32, niveles: int = 4, iter_por_epoca: int = 200, lr: float = 1e-3,
                clips_en_memoria: int = 100, frac_val: float = 0.1, semilla: int = 0, hilos: int | None = None,
                registro=print) -> dict:
    torch.manual_seed(semilla)
    if hilos:
        torch.set_num_threads(hilos)
    rutas = sorted(Path(carpeta).glob("*.npz"))
    if not rutas:
        raise ValueError(f"No hay clips .npz en {carpeta} (use antes: qlus biblioteca)")
    tr, va = particionar(rutas, frac_val, semilla)
    registro(f"Clips sin anotar: {len(tr)} entrenamiento / {len(va)} validación "
             f"({len({paciente_de(r) for r in rutas})} grupos)")
    ds = ConjuntoSinAnotar(tr, tamano, iter_por_epoca * lote, clips_en_memoria, semilla)
    cargador = torch.utils.data.DataLoader(ds, batch_size=lote, shuffle=False, num_workers=0)
    val = conjunto_validacion(va or tr[:1], tamano, semilla)
    m = val[2].float()
    base_val = float(((val[0][:, 1] - val[1]).abs() * m).sum() / m.sum().clamp(min=1.0))

    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = UNet(canales_entrada=3, n_clases=1, base=base, niveles=niveles).to(disp)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epocas * iter_por_epoca)

    mejor, historial = float("inf"), []
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epocas + 1):
        if ep > 1:
            ds.renovar()
        modelo.train()
        t0, acum = time.time(), 0.0
        for entrada, objetivo, mascara in cargador:
            pred = reconstruir(modelo, entrada.to(disp))
            loss = perdida_enmascarada(pred, objetivo.to(disp), mascara.to(disp))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acum += loss.item()
        pv = evaluar(modelo, val, disp)
        historial.append({"epoca": ep, "perdida": acum / iter_por_epoca, "perdida_val": pv})
        registro(f"época {ep:3d}  pérdida {acum / iter_por_epoca:.4f}  val {pv:.4f} "
                 f"(sin modelo {base_val:.4f})  ({time.time() - t0:.0f} s)")
        if pv < mejor:
            mejor = pv
            torch.save({"estado": modelo.state_dict(), "config": modelo.config, "tamano": tamano, "tipo": TIPO,
                        "perdida_val": pv, "perdida_val_sin_modelo": base_val, "epoca": ep,
                        "n_clips": len(rutas)}, salida)
    with open(salida.with_suffix(".historial.json"), "w") as f:
        json.dump(historial, f, indent=1, ensure_ascii=False)
    modelo.load_state_dict(torch.load(str(salida), map_location=disp, weights_only=False)["estado"])
    escribir_png(salida.with_suffix(".ejemplos.png"), lamina_ejemplos(modelo, val, disp))
    return {"mejor_perdida_val": mejor, "perdida_val_sin_modelo": base_val, "historial": historial}
