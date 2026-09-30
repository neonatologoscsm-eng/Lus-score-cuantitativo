"""Post-proceso de la segmentación: línea pleural y exclusión autónoma de sombras costales.

La exclusión de sombras combina tres fuentes para no depender de un único punto
de falla:

1. IA: columnas donde la red segmenta costilla o sombra acústica.
2. Red de seguridad por intensidad: columnas donde la línea pleural esperada
   (interpolada desde las columnas vecinas) no tiene brillo y todo lo que está
   bajo ella es oscuro. Así una sombra que la red no vio igual se elimina.
3. Consistencia temporal: la sonda está quieta, las costillas no se mueven;
   una columna sombreada en la mayoría de los frames se excluye en todo el clip.

Finalmente se agrega un margen lateral para eliminar la penumbra del borde.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from .etiquetas import Clase


@dataclass
class ParametrosSombras:
    frac_sombra_columna: float = 0.25   # fracción de la columna bajo la pleura etiquetada sombra/costilla
    frac_frames_clip: float = 0.5       # sombreada en ≥ 50 % de los frames ⇒ excluida en todo el clip
    margen_mm: float = 0.5              # dilatación lateral (penumbra)
    ventana_pleura_mm: float = 0.5      # ventana vertical para medir el brillo pleural
    brillo_pleural_min: float = 0.5     # fracción del brillo pleural mediano del frame


def filas_pleura(etiquetas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fila superior e inferior del primer tramo de línea pleural por columna (NaN si no hay)."""
    m = etiquetas == Clase.LINEA_PLEURAL
    H = m.shape[0]
    hay = m.any(axis=0)
    sup = np.argmax(m, axis=0)
    filas = np.arange(H)[:, None]
    fuera = (~m) & (filas > sup[None, :])
    inf = np.where(fuera.any(axis=0), np.argmax(fuera, axis=0), H) - 1
    return np.where(hay, sup, np.nan).astype(float), np.where(hay, inf, np.nan).astype(float)


def _sombra_por_ia(etiquetas: np.ndarray, sup: np.ndarray, p: ParametrosSombras) -> np.ndarray:
    H, W = etiquetas.shape
    costilla = (etiquetas == Clase.COSTILLA).any(axis=0)
    ref = np.nanmedian(sup) if np.isfinite(sup).any() else 0.0
    bajo = etiquetas[int(ref):] if np.isfinite(ref) else etiquetas
    frac = (bajo == Clase.SOMBRA_COSTAL).mean(axis=0)
    return costilla | (frac > p.frac_sombra_columna)


def _sombra_por_intensidad(gris: np.ndarray, sup: np.ndarray, inf: np.ndarray,
                           candidatas: np.ndarray, mm_fila: float,
                           p: ParametrosSombras) -> np.ndarray:
    """Red de seguridad: sin brillo pleural y oscuro debajo ⇒ sombra."""
    H, W = gris.shape
    cols = np.arange(W)
    con_pleura = np.isfinite(sup) & ~candidatas
    if con_pleura.sum() < 2:
        return np.zeros(W, bool)
    centro = np.interp(cols, cols[con_pleura], ((sup + inf) / 2)[con_pleura])
    w = max(1, int(round(p.ventana_pleura_mm / mm_fila)))
    filas = np.arange(H)[:, None]
    g = gris.astype(np.float32)
    en_ventana = np.abs(filas - centro[None, :]) <= w
    brillo = np.where(en_ventana, g, -1).max(axis=0)
    debajo = filas > centro[None, :] + w
    media_debajo = np.where(debajo, g, 0).sum(axis=0) / np.maximum(debajo.sum(axis=0), 1)
    ref_brillo = np.median(brillo[con_pleura])
    ref_debajo = np.median(media_debajo[con_pleura])
    return (brillo < p.brillo_pleural_min * ref_brillo) & (media_debajo <= ref_debajo)


def columnas_excluidas(frames: np.ndarray, etiquetas: np.ndarray, mm_por_pixel: tuple[float, float],
                       p: ParametrosSombras | None = None) -> tuple[np.ndarray, dict]:
    """Columnas a excluir por costilla/sombra acústica. Devuelve (T, W) bool e información de trazabilidad."""
    p = p or ParametrosSombras()
    T, H, W = frames.shape
    por_ia = np.zeros((T, W), bool)
    por_intensidad = np.zeros((T, W), bool)
    for t in range(T):
        sup, inf = filas_pleura(etiquetas[t])
        por_ia[t] = _sombra_por_ia(etiquetas[t], sup, p)
        por_intensidad[t] = _sombra_por_intensidad(frames[t], sup, inf, por_ia[t], mm_por_pixel[0], p)
    por_frame = por_ia | por_intensidad
    por_clip = por_frame.mean(axis=0) >= p.frac_frames_clip
    excl = por_frame | por_clip[None, :]
    margen = int(round(p.margen_mm / mm_por_pixel[1]))
    if margen > 0:
        estructura = np.ones((1, 2 * margen + 1), bool)
        excl = binary_dilation(excl, structure=estructura)
    info = {
        "frac_columnas_excluidas": float(excl.mean()),
        "frac_por_ia": float(por_ia.mean()),
        "frac_por_red_seguridad": float((por_intensidad & ~por_ia).mean()),
        "frac_por_consistencia_temporal": float((por_clip[None, :] & ~por_frame).mean()),
    }
    return excl, info
