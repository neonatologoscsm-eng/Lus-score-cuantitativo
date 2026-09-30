"""Entrenamiento de la segmentación por píxel.

Datos: carpeta con clips .npz (frames + etiquetas + mm_por_pixel). El paciente
se toma del nombre del archivo, antes de "__" (p. ej. "P001__ASD.npz"); la
partición entrenamiento/validación es SIEMPRE por paciente para no filtrar
frames del mismo niño entre ambos conjuntos.

Aumentos de datos pensados para la variabilidad entre equipos: desplazamiento
de ganancia (aditivo en gris), cambio de rango dinámico (escala), ruido,
volteo horizontal y recorte/reescalado.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .etiquetas import N_CLASES, NOMBRES, SIN_ANOTAR
from .io import cargar_npz
from .modelos import UNet


def paciente_de(ruta: Path) -> str:
    return ruta.stem.split("__")[0]


def particionar(rutas: list[Path], frac_val: float, semilla: int) -> tuple[list[Path], list[Path]]:
    pacientes = sorted({paciente_de(r) for r in rutas})
    rng = np.random.default_rng(semilla)
    rng.shuffle(pacientes)
    n_val = max(1, int(round(frac_val * len(pacientes)))) if len(pacientes) > 1 else 0
    val = set(pacientes[:n_val])
    return [r for r in rutas if paciente_de(r) not in val], [r for r in rutas if paciente_de(r) in val]


def pila_temporal(frames: np.ndarray, t: int) -> np.ndarray:
    T = frames.shape[0]
    return np.stack([frames[max(t - 1, 0)], frames[t], frames[min(t + 1, T - 1)]])


class ConjuntoClips(torch.utils.data.Dataset):
    def __init__(self, rutas: list[Path], tamano: int, aumentar: bool, muestras: int | None = None,
                 semilla: int = 0):
        self.clips = [cargar_npz(r) for r in rutas]
        for c, r in zip(self.clips, rutas):
            if c.etiquetas is None:
                raise ValueError(f"{r} no tiene etiquetas")
        # solo frames anotados (la anotación puede ser dispersa: 1 de cada n frames)
        self.indice = [(i, t) for i, c in enumerate(self.clips) for t in range(c.frames.shape[0])
                       if (c.etiquetas[t] != SIN_ANOTAR).any()]
        if not self.indice:
            raise ValueError("Ningún frame anotado en los clips")
        self.tamano = tamano
        self.aumentar = aumentar
        self.muestras = muestras
        self.rng = np.random.default_rng(semilla)

    def __len__(self):
        return self.muestras or len(self.indice)

    def __getitem__(self, k):
        if self.muestras:
            i, t = self.indice[self.rng.integers(len(self.indice))]
        else:
            i, t = self.indice[k]
        c = self.clips[i]
        x = pila_temporal(c.frames, t).astype(np.float32)
        y = c.etiquetas[t]
        if self.aumentar:
            x, y = self._aumentar(x, y)
        x = np.stack([cv2.resize(ch, (self.tamano, self.tamano), interpolation=cv2.INTER_LINEAR) for ch in x])
        y = cv2.resize(y, (self.tamano, self.tamano), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(x / 255.0), torch.from_numpy(y.astype(np.int64))

    def _aumentar(self, x, y):
        r = self.rng
        H, W = y.shape
        esc = r.uniform(0.8, 1.0)
        h, w = int(H * esc), int(W * esc)
        y0, x0 = r.integers(0, H - h + 1), r.integers(0, W - w + 1)
        x, y = x[:, y0:y0 + h, x0:x0 + w], y[y0:y0 + h, x0:x0 + w]
        if r.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        x = (x - 128.0) * r.uniform(0.8, 1.25) + 128.0 + r.uniform(-25, 25)  # rango dinámico + ganancia
        x = x + r.normal(0, r.uniform(0, 6), x.shape)
        return np.clip(x, 0, 255).astype(np.float32), np.ascontiguousarray(y)


def pesos_clase(conjunto: ConjuntoClips) -> torch.Tensor:
    cuenta = np.zeros(N_CLASES)
    for c in conjunto.clips:
        cuenta += np.bincount(c.etiquetas.ravel(), minlength=256)[:N_CLASES]
    frec = cuenta / cuenta.sum()
    w = np.where(frec > 0, 1.0 / np.sqrt(frec + 1e-6), 0.0)
    w = w / w[w > 0].mean()
    return torch.tensor(np.clip(w, 0, 10), dtype=torch.float32)


def perdida_dice(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    valido = (y != SIN_ANOTAR).unsqueeze(1).float()
    p = logits.softmax(1) * valido
    oh = nn.functional.one_hot(torch.where(y == SIN_ANOTAR, 0, y), logits.shape[1]).permute(0, 3, 1, 2).float()
    oh = oh * valido
    inter = (p * oh).sum((0, 2, 3))
    union = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    presentes = oh.sum((0, 2, 3)) > 0
    dice = (2 * inter + 1) / (union + 1)
    return 1 - dice[presentes].mean()


@torch.no_grad()
def evaluar(modelo: nn.Module, cargador, dispositivo) -> dict:
    modelo.eval()
    inter = np.zeros(N_CLASES)
    suma = np.zeros(N_CLASES)
    for x, y in cargador:
        pred = modelo(x.to(dispositivo)).argmax(1).cpu().numpy()
        y = y.numpy()
        v = y != SIN_ANOTAR
        for c in range(N_CLASES):
            inter[c] += np.sum((pred == c) & (y == c) & v)
            suma[c] += np.sum((pred == c) & v) + np.sum(y == c)
    dice = {NOMBRES[c]: float(2 * inter[c] / suma[c]) for c in range(N_CLASES) if suma[c] > 0}
    return {"dice": dice, "dice_medio": float(np.mean(list(dice.values())))}


def entrenar(carpeta: str | Path, salida: str | Path, epocas: int = 20, tamano: int = 256, lote: int = 8,
             base: int = 32, iter_por_epoca: int = 200, lr: float = 1e-3, frac_val: float = 0.2,
             semilla: int = 0, hilos: int | None = None, registro=print) -> dict:
    torch.manual_seed(semilla)
    if hilos:
        torch.set_num_threads(hilos)
    rutas = sorted(Path(carpeta).glob("*.npz"))
    if not rutas:
        raise ValueError(f"No hay clips .npz en {carpeta}")
    tr, va = particionar(rutas, frac_val, semilla)
    registro(f"Clips: {len(tr)} entrenamiento / {len(va)} validación "
             f"({len({paciente_de(r) for r in tr})} / {len({paciente_de(r) for r in va})} pacientes)")
    ds_tr = ConjuntoClips(tr, tamano, aumentar=True, muestras=iter_por_epoca * lote, semilla=semilla)
    ds_va = ConjuntoClips(va or tr[:1], tamano, aumentar=False)
    cl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=lote, shuffle=False, num_workers=0)
    cl_va = torch.utils.data.DataLoader(ds_va, batch_size=lote, shuffle=False, num_workers=0)

    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = UNet(canales_entrada=3, n_clases=N_CLASES, base=base).to(disp)
    ce = nn.CrossEntropyLoss(weight=pesos_clase(ds_tr).to(disp), ignore_index=SIN_ANOTAR)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epocas * iter_por_epoca)

    mejor, historial = -1.0, []
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epocas + 1):
        modelo.train()
        t0, acum = time.time(), 0.0
        for x, y in cl_tr:
            x, y = x.to(disp), y.to(disp)
            logits = modelo(x)
            loss = ce(logits, y) + perdida_dice(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acum += loss.item()
        m = evaluar(modelo, cl_va, disp)
        historial.append({"epoca": ep, "perdida": acum / iter_por_epoca, **m})
        registro(f"época {ep:3d}  pérdida {acum / iter_por_epoca:.3f}  dice medio val {m['dice_medio']:.3f}  "
                 f"({time.time() - t0:.0f} s)")
        if m["dice_medio"] > mejor:
            mejor = m["dice_medio"]
            torch.save({"estado": modelo.state_dict(), "config": modelo.config, "tamano": tamano,
                        "clases": NOMBRES, "metricas_val": m, "epoca": ep}, salida)
    with open(salida.with_suffix(".historial.json"), "w") as f:
        json.dump(historial, f, indent=1, ensure_ascii=False)
    return {"mejor_dice_medio": mejor, "historial": historial}
