"""Normalización de intensidad con el hígado del propio paciente.

Fundamento: el modo B muestra la amplitud comprimida en escala logarítmica, así
que un cambio de ganancia desplaza todos los grises en la misma cantidad. La
diferencia de gris entre un píxel pulmonar y el hígado, convertida a dB, no
depende de la ganancia. El hígado es un tejido grande, homogéneo, sin aire y
accesible con la misma sonda y el mismo preset en cada examen.

Como la atenuación hace que el gris del hígado caiga con la profundidad, se
ajusta una recta gris = a + b·profundidad y el pulmón de cada píxel se compara
con el hígado a su misma profundidad.

Modos:
- "un_punto": dB por nivel de gris = rango dinámico del preset / 255.
- "dos_puntos": además usa un vaso hepático anecoico (sangre) como segundo
  ancla; corrige diferencias de rango dinámico entre equipos si se conoce el
  contraste hígado–sangre en dB. Solo es válido si el vaso no está recortado a 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .etiquetas import Clase

# Anclas de la escala de blancura en dB relativos al hígado:
# (pulmón aireado entre líneas A → 0 ; líneas B confluentes → 1).
# PROVISIONALES: deben fijarse una sola vez con los datos de desarrollo y luego congelarse.
ANCLAS_DB = (-20.0, 15.0)


@dataclass
class Referencia:
    clase: str
    modo: str
    intercepto: float            # gris del tejido de referencia a 0 mm de profundidad
    pendiente: float             # gris por mm
    db_por_gris: float
    gris_anecoico: float | None
    n_pixeles: int
    heterogeneidad_db: float     # dispersión entre bloques de ~3 mm, tras corregir profundidad
    rango_prof_mm: tuple[float, float]
    gris_pared: float | None = None  # mediana de la pared torácica del mismo clip (control de ganancia)
    advertencias: list[str] = field(default_factory=list)

    def gris_referencia(self, prof_mm):
        return self.intercepto + self.pendiente * prof_mm

    def a_db(self, gris, prof_mm):
        """Ecogenicidad relativa a la referencia a la misma profundidad, en dB."""
        return (np.asarray(gris, np.float32) - self.gris_referencia(prof_mm)) * self.db_por_gris

    def como_dict(self) -> dict:
        d = asdict(self)
        d["rango_prof_mm"] = list(self.rango_prof_mm)
        return d


def blancura(gris, prof_mm, ref: Referencia, anclas=ANCLAS_DB):
    """Blancura normalizada 0–1 a partir de dB relativos a la referencia."""
    r = ref.a_db(gris, prof_mm)
    return np.clip((r - anclas[0]) / (anclas[1] - anclas[0]), 0.0, 1.0)


def estimar_referencia(frames: np.ndarray, etiquetas: np.ndarray, mm_por_pixel: tuple[float, float],
                       rango_dinamico_db: float | None = None,
                       contraste_tejido_sangre_db: float | None = None,
                       clase: Clase = Clase.HIGADO, margen_mm: float = 1.0,
                       min_pixeles: int = 2000) -> Referencia:
    """Estima la referencia de intensidad desde un clip que muestra el hígado (u otro tejido)."""
    T, H, W = frames.shape
    my, mx = mm_por_pixel
    k = max(1, int(round(margen_mm / min(my, mx))))
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    kv = max(1, int(round(0.3 / min(my, mx))))   # los vasos son chicos: solo se quita el borde
    nucleo_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kv + 1, 2 * kv + 1))
    prof_fila = np.arange(H) * my
    prof, gris, vasos, pared = [], [], [], []
    mascaras = []
    for t in range(T):
        tejido = (etiquetas[t] == clase).astype(np.uint8)
        vaso = (etiquetas[t] == Clase.VASO_ANECOICO).astype(np.uint8)
        util = cv2.erode(tejido, nucleo).astype(bool) & ~cv2.dilate(vaso, nucleo).astype(bool)
        mascaras.append(util)
        f, c = np.nonzero(util)
        prof.append(prof_fila[f])
        gris.append(frames[t][f, c])
        nucleo_vaso = cv2.erode(vaso, nucleo_v).astype(bool)
        vasos.append(frames[t][nucleo_vaso])
        pared.append(frames[t][etiquetas[t] == Clase.PARED_TORACICA])
    prof = np.concatenate(prof)
    gris = np.concatenate(gris).astype(np.float32)
    advertencias: list[str] = []
    if gris.size < min_pixeles:
        raise ValueError(f"Referencia {clase.name.lower()} insuficiente: {gris.size} píxeles útiles (< {min_pixeles}).")

    saturados = float(np.mean((gris >= 250) | (gris <= 5)))
    if saturados > 0.02:
        advertencias.append(f"{saturados:.0%} de la referencia está saturada; revisar ganancia/preset.")

    # recta gris–profundidad sobre medianas por bin de 1 mm (robusta al speckle)
    bins = np.floor(prof).astype(int)
    ub, inv, cnt = np.unique(bins, return_inverse=True, return_counts=True)
    med = np.array([np.median(gris[inv == i]) for i in range(len(ub))])
    ok = cnt >= 50
    zc, mc, wc = ub[ok] + 0.5, med[ok], np.sqrt(cnt[ok])
    if len(zc) >= 3 and (zc.max() - zc.min()) >= 3:
        pendiente, intercepto = np.polyfit(zc, mc, 1, w=wc)
    else:
        pendiente, intercepto = 0.0, float(np.median(gris))
        advertencias.append("Rango de profundidad de la referencia < 3 mm: sin corrección por profundidad.")
    rango = (float(zc.min()) if len(zc) else 0.0, float(zc.max()) if len(zc) else 0.0)

    gris_vaso = float(np.median(np.concatenate(vasos))) if sum(v.size for v in vasos) >= 50 else None
    if contraste_tejido_sangre_db is not None:
        if gris_vaso is None or gris_vaso <= 5:
            raise ValueError("Modo dos puntos no disponible: no hay vaso anecoico útil o está recortado a 0.")
        prof_vaso = float(np.mean(rango))
        dif = float(intercepto + pendiente * prof_vaso) - gris_vaso
        if dif <= 1:
            raise ValueError("Modo dos puntos no disponible: sin contraste entre tejido y vaso.")
        modo, db_por_gris = "dos_puntos", contraste_tejido_sangre_db / dif
    elif rango_dinamico_db is not None:
        modo, db_por_gris = "un_punto", float(rango_dinamico_db) / 255.0
    else:
        raise ValueError("Se requiere el rango dinámico del preset (dB) o el contraste tejido–sangre (modo dos puntos).")

    # heterogeneidad: medianas por bloques de ~3 mm, residuo tras la recta de profundidad
    bloque = max(2, int(round(3.0 / min(my, mx))))
    residuos = []
    for t, util in enumerate(mascaras[:: max(1, T // 8)]):
        tt = t * max(1, T // 8)
        for y0 in range(0, H - bloque + 1, bloque):
            for x0 in range(0, W - bloque + 1, bloque):
                m = util[y0:y0 + bloque, x0:x0 + bloque]
                if m.mean() >= 0.7:
                    g = np.median(frames[tt, y0:y0 + bloque, x0:x0 + bloque][m])
                    z = (y0 + bloque / 2) * my
                    residuos.append(g - (intercepto + pendiente * z))
    heterog = float(np.std(residuos) * db_por_gris) if len(residuos) >= 4 else float("nan")
    if np.isfinite(heterog) and heterog > 3.0:
        advertencias.append(f"Referencia heterogénea ({heterog:.1f} dB): posible lesión focal, vaso no segmentado o artefacto.")

    pared = np.concatenate(pared)
    return Referencia(clase=clase.name.lower(), modo=modo, intercepto=float(intercepto),
                      pendiente=float(pendiente), db_por_gris=float(db_por_gris), gris_anecoico=gris_vaso,
                      n_pixeles=int(gris.size), heterogeneidad_db=heterog, rango_prof_mm=rango,
                      gris_pared=float(np.median(pared)) if pared.size >= 200 else None,
                      advertencias=advertencias)
