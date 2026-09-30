"""Superposiciones para revisar qué segmentó la IA y qué columnas se excluyeron."""

from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from .etiquetas import COLORES, Clase


def superponer(gris: np.ndarray, etiquetas: np.ndarray, excluidas: np.ndarray | None = None,
               alfa: float = 0.35) -> np.ndarray:
    """Imagen BGR: ecografía + clases coloreadas; columnas excluidas con rayado gris."""
    base = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR).astype(np.float32)
    color = np.zeros_like(base)
    for c, rgb in COLORES.items():
        color[etiquetas == c] = rgb[::-1]
    mezcla = np.where((etiquetas != Clase.FONDO)[..., None], (1 - alfa) * base + alfa * color, base)
    pleura = etiquetas == Clase.LINEA_PLEURAL
    mezcla[pleura] = np.array(COLORES[Clase.LINEA_PLEURAL][::-1], np.float32)
    if excluidas is not None and excluidas.any():
        H = gris.shape[0]
        rayado = ((np.arange(H)[:, None] + np.arange(gris.shape[1])[None, :]) % 8) < 2
        m = excluidas[None, :] & rayado
        mezcla[m] = 0.5 * mezcla[m] + 0.5 * np.array([160, 160, 160], np.float32)
    return np.clip(mezcla, 0, 255).astype(np.uint8)


def lamina_resumen(gris: np.ndarray, etiquetas: np.ndarray, excluidas: np.ndarray | None,
                   texto: list[str]) -> np.ndarray:
    """Original | superposición, con líneas de texto debajo."""
    izq = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    der = superponer(gris, etiquetas, excluidas)
    img = np.hstack([izq, der])
    alto_txt = 18 * len(texto) + 8
    lienzo = np.zeros((img.shape[0] + alto_txt, img.shape[1], 3), np.uint8)
    lienzo[: img.shape[0]] = img
    for i, linea in enumerate(texto):
        linea = unicodedata.normalize("NFKD", linea).encode("ascii", "ignore").decode()  # fuentes Hershey: solo ASCII
        cv2.putText(lienzo, linea, (6, img.shape[0] + 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return lienzo
