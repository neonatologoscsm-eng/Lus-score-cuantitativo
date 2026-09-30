"""Cuantificación determinística y puramente ecográfica.

Entrada: frames en gris + segmentación por píxel (de la IA o de un experto) +
referencia hepática. Salida por zona: FBA, CPB, consolidaciones, derrame,
índice de deslizamiento e IPA. Ningún parámetro se ajusta con datos clínicos.

Definiciones (ver docs/propuesta-tecnica.md):
- Banda de medición: profundidad fija bajo la superficie pulmonar de cada columna válida.
- Blancura por columna: percentil bajo (P25) de la blancura normalizada de los
  píxeles de pulmón en la banda; separa líneas B (blancas de arriba a abajo) de
  líneas A (con huecos negros).
- FBA (fracción de blanco aparente, 0–100): blancura media del pulmón no consolidado.
- CPB (cobertura pleural por patrón B, 0–100): % de columnas con blancura ≥ umbral.
- Consolidación subpleural/extensa: por la profundidad máxima de cada componente.
- IPA (índice de pérdida de aireación, 0–100) = 100·[c_sub + c_ext + (1 − c_sub − c_ext)·FBA].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import label as etiquetar_componentes

from .etiquetas import Clase
from .normalizacion import ANCLAS_DB, Referencia, blancura
from .postproceso import ParametrosSombras, columnas_excluidas, filas_pleura


@dataclass
class Parametros:
    d0_mm: float = 0.5                   # inicio de la banda bajo la superficie pulmonar
    banda_mm: float = 15.0               # alto de la banda de medición
    umbral_subpleural_mm: float = 5.0    # consolidación ≤ 5 mm de profundidad = subpleural
    min_area_consolidacion_mm2: float = 0.5  # componentes menores se descartan como ruido de segmentación
    percentil_vertical: float = 25.0
    umbral_patron_b: float = 0.5         # blancura de columna para contar como patrón B
    anclas_db: tuple[float, float] = ANCLAS_DB
    min_frac_columnas: float = 0.3       # columnas válidas mínimas para aceptar un frame
    min_frac_filas_columna: float = 0.3  # pulmón mínimo en la banda para usar una columna
    max_prof_superficie_mm: float = 20.0 # superficie pulmonar a lo más 20 mm bajo la pleura
    umbral_deslizamiento: float = 0.05
    max_cpb_neumotorax: float = 5.0      # % de patrón B compatible con neumotórax
    min_frames_validos: int = 3
    max_delta_pared_db: float = 3.0      # control de ganancia entre el clip hepático y cada zona
    sombras: ParametrosSombras = field(default_factory=ParametrosSombras)


@dataclass
class MedicionFrame:
    valido: bool
    frac_columnas_validas: float
    fba: float | None = None
    cpb: float | None = None
    cons_subpleural_pct: float = 0.0
    cons_extensa_pct: float = 0.0
    ipa: float | None = None
    cons_prof_max_mm: float = 0.0
    cons_long_max_mm: float = 0.0
    derrame_max_mm: float = 0.0
    derrame_area_mm2: float = 0.0
    prof_banda_mm: tuple[float, float] | None = None


def superficie_pulmonar(etiquetas: np.ndarray, inf_pleura: np.ndarray, mm_fila: float,
                        max_prof_mm: float) -> np.ndarray:
    """Primera fila de pulmón o consolidación bajo la pleura (NaN si no hay o está muy profunda)."""
    H, W = etiquetas.shape
    pul = np.isin(etiquetas, (Clase.PULMON, Clase.CONSOLIDACION))
    base = np.nan_to_num(inf_pleura, nan=H).astype(int)
    filas = np.arange(H)[:, None]
    m = pul & (filas > base[None, :])
    hay = m.any(axis=0) & np.isfinite(inf_pleura)
    s = np.argmax(m, axis=0).astype(float)
    hay &= (s - inf_pleura) * mm_fila <= max_prof_mm
    return np.where(hay, s, np.nan)


def cuantificar_frame(gris: np.ndarray, etiquetas: np.ndarray, mm_por_pixel: tuple[float, float],
                      ref: Referencia, excluidas: np.ndarray, p: Parametros) -> MedicionFrame:
    H, W = gris.shape
    my, mx = mm_por_pixel
    sup, inf = filas_pleura(etiquetas)
    s = superficie_pulmonar(etiquetas, inf, my, p.max_prof_superficie_mm)
    validas = np.isfinite(s) & ~excluidas
    frac_validas = float(validas.mean())
    med = MedicionFrame(valido=False, frac_columnas_validas=frac_validas)

    # derrame (se reporta, no entra al IPA)
    derrame = (etiquetas == Clase.DERRAME) & ~excluidas[None, :]
    if derrame.any():
        med.derrame_max_mm = float(derrame.sum(axis=0).max() * my)
        med.derrame_area_mm2 = float(derrame.sum() * my * mx)

    if frac_validas < p.min_frac_columnas:
        return med

    d0 = int(round(p.d0_mm / my))
    alto = max(1, int(round(p.banda_mm / my)))
    s_int = np.nan_to_num(s, nan=H).astype(int)
    r0 = s_int + d0
    r1 = np.minimum(r0 + alto, H)
    filas = np.arange(H)[:, None]
    banda = (filas >= r0[None, :]) & (filas < r1[None, :]) & validas[None, :]

    pulmon = banda & (etiquetas == Clase.PULMON)
    cons_total = etiquetas == Clase.CONSOLIDACION

    # consolidaciones: subpleural vs extensa por la profundidad máxima de cada componente
    comp, n = etiquetar_componentes(cons_total)
    cons_sub = np.zeros_like(cons_total)
    cons_ext = np.zeros_like(cons_total)
    if n:
        f, c = np.nonzero(cons_total)
        prof = (f - s[c]) * my
        ids = comp[f, c]
        for i in range(1, n + 1):
            sel = ids == i
            if sel.sum() * my * mx < p.min_area_consolidacion_mm2:
                continue
            pmax = np.nanmax(prof[sel]) if np.isfinite(prof[sel]).any() else np.inf
            destino = cons_sub if pmax <= p.umbral_subpleural_mm else cons_ext
            destino[f[sel], c[sel]] = True
            cols = np.unique(c[sel])
            if np.isfinite(pmax):
                med.cons_prof_max_mm = max(med.cons_prof_max_mm, float(pmax + my))
            med.cons_long_max_mm = max(med.cons_long_max_mm, float(len(cols) * mx))
    n_sub = int((banda & cons_sub).sum())
    n_ext = int((banda & cons_ext).sum())
    n_pul = int(pulmon.sum())
    total = n_pul + n_sub + n_ext
    if total == 0:
        return med

    # blancura por columna (percentil bajo vertical) en el pulmón no consolidado
    prof_mm = (np.arange(H) * my)[:, None]
    e = blancura(gris, prof_mm, ref, p.anclas_db)
    e = np.where(pulmon, e, np.nan)
    n_col = pulmon.sum(axis=0)
    usar = validas & (n_col >= p.min_frac_filas_columna * alto)
    c_sub, c_ext = n_sub / total, n_ext / total
    fba = None
    if usar.any():
        b = np.nanpercentile(e[:, usar], p.percentil_vertical, axis=0)
        fba = float(np.sum(b * n_col[usar]) / np.sum(n_col[usar]))
        med.cpb = float(np.mean(b >= p.umbral_patron_b) * 100)
        med.fba = fba * 100
    elif n_pul > 0:
        return med

    med.cons_subpleural_pct = c_sub * 100
    med.cons_extensa_pct = c_ext * 100
    med.ipa = 100 * (c_sub + c_ext + (1 - c_sub - c_ext) * (fba or 0.0))
    med.prof_banda_mm = (float(r0[validas].min() * my), float(r1[validas].max() * my))
    med.valido = True
    return med


def indice_deslizamiento(frames: np.ndarray, etiquetas: np.ndarray, mm_por_pixel: tuple[float, float],
                         excluidas: np.ndarray, fps: float, max_desp_px: int = 3) -> float:
    """Movimiento horizontal bajo la pleura respecto de la pared (≈0 en neumotórax).

    Para cada par de frames se busca el desplazamiento horizontal que mejor
    alinea una franja justo bajo la pleura; la mejora sobre "sin desplazamiento"
    mide cuánto se movió. Se resta la misma medida en la pared torácica.
    """
    T, H, W = frames.shape
    my = mm_por_pixel[0]
    lag = max(1, int(round(fps / 10)))
    if T <= lag:
        return float("nan")
    a0, a1 = int(round(0.5 / my)), max(int(round(3.0 / my)), int(round(0.5 / my)) + 2)
    cols = np.arange(W)

    def franja(t, arriba):
        # arriba: pared torácica sobre la pleura parietal; abajo: bajo la superficie pulmonar
        # (visceral), para que un derrame interpuesto no oculte el deslizamiento.
        sup, inf = filas_pleura(etiquetas[t])
        s = superficie_pulmonar(etiquetas[t], inf, my, 20.0)
        ref = sup if arriba else s
        ok = np.isfinite(ref) & ~excluidas[t]
        if ok.sum() < 10:
            return None, None
        if arriba:
            base = np.interp(cols, cols[ok], sup[ok]) - a1
        else:
            base = np.interp(cols, cols[ok], s[ok]) + a0
        idx = np.clip(base.astype(int)[None, :] + np.arange(a1 - a0)[:, None], 0, H - 1)
        return np.take_along_axis(frames[t].astype(np.float32), idx, axis=0), ok

    def mejora(t, arriba):
        A, okA = franja(t, arriba)
        B, okB = franja(t - lag, arriba)
        if A is None or B is None:
            return None
        ok = okA & okB
        ssd = {}
        for d in range(-max_desp_px, max_desp_px + 1):
            m = ok & np.roll(ok, -d)
            m[max(0, W - d):] = False
            m[:max(0, -d)] = False
            if m.sum() < 10:
                continue
            ssd[d] = float(np.mean((A[:, m] - np.roll(B, -d, axis=1)[:, m]) ** 2))
        if 0 not in ssd or ssd[0] <= 0:
            return 0.0
        return 1.0 - min(ssd.values()) / ssd[0]

    abajo, arriba = [], []
    for t in range(lag, T):
        m_ab, m_ar = mejora(t, False), mejora(t, True)
        if m_ab is not None and m_ar is not None:
            abajo.append(m_ab)
            arriba.append(m_ar)
    if not abajo:
        return float("nan")
    return float(np.mean(abajo) - np.mean(arriba))


@dataclass
class ResultadoZona:
    zona: str
    estado: str                            # "cuantificado" | "no_cuantificable_ntx" | "calidad_insuficiente"
    ipa: float | None = None
    ipa_p25_p75: tuple[float, float] | None = None
    fba: float | None = None
    cpb: float | None = None
    cons_subpleural_pct: float | None = None
    cons_extensa_pct: float | None = None
    cons_prof_max_mm: float = 0.0
    cons_long_max_mm: float = 0.0
    derrame_max_mm: float = 0.0
    derrame_area_mm2: float = 0.0
    indice_deslizamiento: float | None = None
    frames_validos: int = 0
    frames_totales: int = 0
    frac_columnas_validas: float = 0.0
    sombras: dict = field(default_factory=dict)
    advertencias: list[str] = field(default_factory=list)

    def como_dict(self) -> dict:
        return asdict(self)


def _mediana(valores):
    v = [x for x in valores if x is not None]
    return float(np.median(v)) if v else None


def cuantificar_clip(frames: np.ndarray, etiquetas: np.ndarray, mm_por_pixel: tuple[float, float],
                     ref: Referencia, zona: str = "zona", fps: float = 20.0,
                     p: Parametros | None = None) -> ResultadoZona:
    p = p or Parametros()
    T = frames.shape[0]
    excl, info_sombras = columnas_excluidas(frames, etiquetas, mm_por_pixel, p.sombras)
    medidas = [cuantificar_frame(frames[t], etiquetas[t], mm_por_pixel, ref, excl[t], p) for t in range(T)]
    validas = [m for m in medidas if m.valido]
    desliz = indice_deslizamiento(frames, etiquetas, mm_por_pixel, excl, fps)
    res = ResultadoZona(zona=zona, estado="cuantificado", frames_totales=T, frames_validos=len(validas),
                        frac_columnas_validas=float(np.median([m.frac_columnas_validas for m in medidas])),
                        sombras=info_sombras,
                        indice_deslizamiento=None if not np.isfinite(desliz) else desliz)
    res.derrame_max_mm = float(np.median([m.derrame_max_mm for m in medidas]))
    res.derrame_area_mm2 = float(np.median([m.derrame_area_mm2 for m in medidas]))
    if res.derrame_max_mm > 0:
        res.advertencias.append(f"Derrame pleural: {res.derrame_max_mm:.1f} mm (excluido del IPA).")

    if ref.gris_pared is not None:
        pared = frames[etiquetas == Clase.PARED_TORACICA]
        if pared.size >= 200:
            delta = (float(np.median(pared)) - ref.gris_pared) * ref.db_por_gris
            if abs(delta) > p.max_delta_pared_db:
                res.advertencias.append(
                    f"La pared torácica difiere {delta:+.1f} dB del clip hepático: posible cambio de ganancia/preset "
                    "entre clips; la normalización hepática solo es válida con el preset bloqueado.")

    if len(validas) < p.min_frames_validos:
        res.estado = "calidad_insuficiente"
        res.advertencias.append(f"Solo {len(validas)} frames válidos (mínimo {p.min_frames_validos}).")
        return res

    res.cpb = _mediana(m.cpb for m in validas)
    res.cons_subpleural_pct = _mediana(m.cons_subpleural_pct for m in validas)
    res.cons_extensa_pct = _mediana(m.cons_extensa_pct for m in validas)
    cons = (res.cons_subpleural_pct or 0) + (res.cons_extensa_pct or 0)
    sin_desliz = res.indice_deslizamiento is not None and res.indice_deslizamiento < p.umbral_deslizamiento
    if sin_desliz and (res.cpb or 0) <= p.max_cpb_neumotorax and cons < 1.0:
        res.estado = "no_cuantificable_ntx"
        res.advertencias.append("Sin deslizamiento pleural, sin líneas B ni consolidación: sospecha de neumotórax. "
                                "Zona NO cuantificada; confirmar clínicamente (punto pulmonar, modo M).")
        return res

    ipas = [m.ipa for m in validas]
    res.ipa = float(np.median(ipas))
    res.ipa_p25_p75 = (float(np.percentile(ipas, 25)), float(np.percentile(ipas, 75)))
    res.fba = _mediana(m.fba for m in validas)
    res.cons_prof_max_mm = float(np.median([m.cons_prof_max_mm for m in validas]))
    res.cons_long_max_mm = float(np.median([m.cons_long_max_mm for m in validas]))
    bandas = [m.prof_banda_mm for m in validas if m.prof_banda_mm]
    if bandas:
        lo, hi = min(b[0] for b in bandas), max(b[1] for b in bandas)
        if lo < ref.rango_prof_mm[0] - 3 or hi > ref.rango_prof_mm[1] + 3:
            res.advertencias.append(
                f"La banda pulmonar ({lo:.0f}–{hi:.0f} mm) sale del rango de profundidad de la referencia "
                f"({ref.rango_prof_mm[0]:.0f}–{ref.rango_prof_mm[1]:.0f} mm): normalización extrapolada.")
    if sin_desliz:
        res.advertencias.append("Deslizamiento pleural bajo; revisar clip.")
    return res


def resumir_examen(zonas: list[ResultadoZona], ref: Referencia) -> dict:
    cuant = [z for z in zonas if z.estado == "cuantificado" and z.ipa is not None]
    ipas = [z.ipa for z in cuant]
    return {
        "ipa_global": float(np.mean(ipas)) if ipas else None,
        "heterogeneidad_entre_zonas": float(np.std(ipas)) if len(ipas) >= 2 else None,
        "zonas_cuantificadas": len(cuant),
        "zonas_totales": len(zonas),
        "zonas_sospecha_ntx": [z.zona for z in zonas if z.estado == "no_cuantificable_ntx"],
        "zonas_calidad_insuficiente": [z.zona for z in zonas if z.estado == "calidad_insuficiente"],
        "zonas_con_derrame": [z.zona for z in zonas if z.derrame_max_mm > 0],
        "referencia": ref.como_dict(),
        "zonas": [z.como_dict() for z in zonas],
    }
