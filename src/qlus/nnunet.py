"""Exportación de clips anotados al formato de nnU-Net v2 (2D, PNG).

Cada frame es un caso con 3 canales (t−1, t, t+1). Se escribe además
`splits_final.json` con folds agrupados por paciente: nnU-Net, por defecto,
reparte casos al azar y mezclaría frames del mismo niño entre entrenamiento y
validación. Copiar ese archivo a nnUNet_preprocessed/DatasetXXX_*/ antes de
entrenar.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .entrenamiento import paciente_de, pila_temporal
from .etiquetas import N_CLASES, NOMBRES, SIN_ANOTAR
from .io import cargar_npz


def exportar(carpeta: str | Path, salida: str | Path, id_dataset: int = 501, nombre: str = "LUSNeo",
             cada_n_frames: int = 1, n_folds: int = 5, semilla: int = 0) -> Path:
    rutas = sorted(Path(carpeta).glob("*.npz"))
    if not rutas:
        raise ValueError(f"No hay clips .npz en {carpeta}")
    raiz = Path(salida) / f"Dataset{id_dataset:03d}_{nombre}"
    (raiz / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raiz / "labelsTr").mkdir(parents=True, exist_ok=True)
    casos_por_paciente: dict[str, list[str]] = {}
    n = 0
    hay_ignorados = False
    for r in rutas:
        clip = cargar_npz(r)
        if clip.etiquetas is None:
            continue
        for t in range(0, clip.frames.shape[0], cada_n_frames):
            lab = clip.etiquetas[t]
            if (lab == SIN_ANOTAR).all():
                continue
            if (lab == SIN_ANOTAR).any():
                hay_ignorados = True
                lab = np.where(lab == SIN_ANOTAR, N_CLASES, lab).astype(np.uint8)  # etiqueta "ignore" de nnU-Net
            caso = f"{r.stem}_f{t:04d}"
            for k, canal in enumerate(pila_temporal(clip.frames, t)):
                cv2.imwrite(str(raiz / "imagesTr" / f"{caso}_{k:04d}.png"), canal)
            cv2.imwrite(str(raiz / "labelsTr" / f"{caso}.png"), lab)
            casos_por_paciente.setdefault(paciente_de(r), []).append(caso)
            n += 1
    etiquetas = {("background" if i == 0 else nombre_c): i for i, nombre_c in NOMBRES.items()}
    if hay_ignorados:
        etiquetas["ignore"] = N_CLASES
    with open(raiz / "dataset.json", "w") as f:
        json.dump({"channel_names": {"0": "US_t-1", "1": "US_t", "2": "US_t+1"}, "labels": etiquetas,
                   "numTraining": n, "file_ending": ".png"}, f, indent=1, ensure_ascii=False)

    pacientes = sorted(casos_por_paciente)
    np.random.default_rng(semilla).shuffle(pacientes)
    k = min(n_folds, len(pacientes))
    folds = []
    for i in range(k):
        val = pacientes[i::k]
        folds.append({"train": [c for p in pacientes if p not in val for c in casos_por_paciente[p]],
                      "val": [c for p in val for c in casos_por_paciente[p]]})
    with open(raiz / "splits_final.json", "w") as f:
        json.dump(folds, f)
    return raiz
