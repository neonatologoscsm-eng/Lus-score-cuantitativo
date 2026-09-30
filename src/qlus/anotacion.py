"""Puente con las herramientas de anotación (CVAT, 3D Slicer, Label Studio...).

- `etiquetas_cvat()`: configuración de etiquetas para crear el proyecto en CVAT
  con los mismos nombres y colores que usa el código.
- `importar_mascaras()`: une un clip (DICOM/video) con las máscaras PNG exportadas
  por frame. Acepta máscaras en índices de clase (1 canal) o en color (RGB con la
  paleta de `etiquetas.COLORES`). La anotación puede ser dispersa: los frames sin
  máscara quedan como SIN_ANOTAR y se usan solo como contexto temporal.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .etiquetas import COLORES, N_CLASES, NOMBRES, SIN_ANOTAR
from .io import Clip

# Etiqueta para regiones dudosas: el anotador las marca y el entrenamiento las ignora.
COLOR_IGNORAR = (255, 0, 255)


def etiquetas_cvat() -> list[dict]:
    fila = lambda nombre, rgb: {"name": nombre, "color": "#{:02x}{:02x}{:02x}".format(*rgb), "type": "any",
                                "attributes": []}
    return [fila(NOMBRES[int(c)], rgb) for c, rgb in COLORES.items() if int(c) != 0] + [fila("ignorar", COLOR_IGNORAR)]


def _indice_frame(ruta: Path) -> int:
    numeros = re.findall(r"\d+", ruta.stem)
    if not numeros:
        raise ValueError(f"{ruta.name}: el nombre debe contener el número de frame")
    return int(numeros[-1])


def _mascara_a_clases(img: np.ndarray, ruta: Path) -> np.ndarray:
    if img.ndim == 2:
        if img.max() >= N_CLASES and not (img[img >= N_CLASES] == SIN_ANOTAR).all():
            raise ValueError(f"{ruta.name}: valores de clase fuera de rango")
        return img.astype(np.uint8)
    rgb = img[..., :3][..., ::-1]  # cv2 lee BGR
    salida = np.full(rgb.shape[:2], SIN_ANOTAR, np.uint8)
    for c, color in COLORES.items():
        salida[(rgb == np.array(color, np.uint8)).all(-1)] = int(c)
    desconocidos = (salida == SIN_ANOTAR) & ~(rgb == np.array(COLOR_IGNORAR, np.uint8)).all(-1)
    if desconocidos.mean() > 0.01:
        colores = np.unique(rgb[desconocidos].reshape(-1, 3), axis=0)[:5]
        raise ValueError(f"{ruta.name}: colores que no están en la paleta, p. ej. {colores.tolist()}")
    return salida


def importar_mascaras(clip: Clip, carpeta: str | Path, desfase_frame: int = 0) -> Clip:
    rutas = sorted(Path(carpeta).glob("*.png"))
    if not rutas:
        raise ValueError(f"No hay máscaras PNG en {carpeta}")
    T, H, W = clip.frames.shape
    etiquetas = np.full((T, H, W), SIN_ANOTAR, np.uint8)
    for r in rutas:
        t = _indice_frame(r) - desfase_frame
        if not 0 <= t < T:
            raise ValueError(f"{r.name}: frame {t} fuera del clip (0–{T - 1}); revise --desfase-frame")
        img = cv2.imread(str(r), cv2.IMREAD_UNCHANGED)
        if img.shape[:2] != (H, W):
            raise ValueError(f"{r.name}: tamaño {img.shape[:2]} distinto del clip {(H, W)}")
        etiquetas[t] = _mascara_a_clases(img, r)
    return Clip(frames=clip.frames, mm_por_pixel=clip.mm_por_pixel, fps=clip.fps, etiquetas=etiquetas,
                meta={**clip.meta, "frames_anotados": int((etiquetas != SIN_ANOTAR).any(axis=(1, 2)).sum())})
