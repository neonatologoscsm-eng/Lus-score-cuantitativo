"""Generador de clips sintéticos de ecografía pulmonar neonatal con máscaras exactas.

Sirve para (1) probar la tubería completa sin datos reales, (2) pruebas
unitarias con verdad conocida y (3) preentrenamiento. No reemplaza datos
reales: la física está simplificada a propósito.

Modelo de imagen: amplitud lineal × speckle + ruido térmico → dB → atenuación
neta con la profundidad → ganancia → compresión logarítmica a 0–255 con un
rango dinámico dado. La ganancia desplaza los grises de forma aditiva, igual
que en un ecógrafo con mapa de grises lineal en dB.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from .etiquetas import Clase
from .io import Clip

ATENUACION_NETA_DB_POR_MM = 0.25  # atenuación tras compensación TGC


def db(x: float) -> float:
    return 10.0 ** (x / 20.0)


@dataclass
class Escena:
    humedad: float = 0.0                 # 0 = aireado (líneas A); 1 = pulmón blanco confluente
    n_costillas: int = 2
    # (centro_mm, semiancho_mm, profundidad_mm) medidos desde la superficie pulmonar
    consolidaciones: list = field(default_factory=list)
    derrame_mm: float = 0.0
    neumotorax: bool = False
    higado: bool = False                 # vista de referencia hepática (receso costofrénico derecho)
    ganancia_db: float = 0.0
    rango_dinamico_db: float = 60.0
    prof_pleura_mm: float = 4.0
    n_frames: int = 16
    alto: int = 256
    ancho: int = 256
    mm_por_pixel: float = 0.15
    fps: float = 20.0
    semilla: int = 0


def _speckle(rng, forma, sigma=0.8):
    re = gaussian_filter(rng.standard_normal(forma), sigma)
    im = gaussian_filter(rng.standard_normal(forma), sigma)
    s = np.hypot(re, im)
    return s / s.mean()


def generar_clip(e: Escena) -> Clip:
    rng = np.random.default_rng(e.semilla)
    H, W, mm = e.alto, e.ancho, e.mm_por_pixel
    margen = 12                                    # px extra para el deslizamiento
    Wc = W + 2 * margen
    z = (np.arange(H) * mm)[:, None]               # mm, (H, 1)
    x = (np.arange(W) * mm)[None, :]               # mm, (1, W)

    humedad = 0.0 if e.neumotorax else float(np.clip(e.humedad, 0, 1))
    consolidaciones = [] if e.neumotorax else list(e.consolidaciones)

    # --- capa estática ------------------------------------------------------
    zp = e.prof_pleura_mm + 0.15 * np.sin(2 * np.pi * x / rng.uniform(3, 8) + rng.uniform(0, 6))
    grosor_pl = 0.45
    amp = np.zeros((H, W))
    lab = np.full((H, W), Clase.FONDO, np.uint8)

    pared = z < zp
    amp[pared] = db(-35)
    amp[(z < 0.4) & pared] = db(-18)
    amp[(np.abs(z - 0.45 * zp) < 0.15) & pared] = db(-24)
    lab[pared] = Clase.PARED_TORACICA

    pleura = (z >= zp) & (z < zp + grosor_pl)
    amp[pleura] = db(-10)
    lab[pleura] = Clase.LINEA_PLEURAL

    grosor_derrame = np.zeros_like(x)
    if e.derrame_mm > 0:
        c = rng.uniform(0.3, 0.7) * W * mm
        grosor_derrame = e.derrame_mm * np.clip(1.2 - ((x - c) / (0.6 * W * mm)) ** 2, 0.3, 1.0)
    z_sup = zp + grosor_pl + grosor_derrame        # superficie pulmonar (visceral)
    derrame = (z >= zp + grosor_pl) & (z < z_sup)
    amp[derrame] = db(-62)
    lab[derrame] = Clase.DERRAME

    region_pulmon = z >= z_sup
    higado = np.zeros((H, W), bool)
    if e.higado:
        xd = rng.uniform(0.35, 0.5) * W * mm
        borde = xd + 0.5 * (z - zp)                # diafragma oblicuo
        diafragma = (np.abs(x - borde) < 0.35) & (z >= zp)
        higado = (x > borde + 0.35) & (z >= zp)
        amp[higado] = db(-30)
        lab[higado] = Clase.HIGADO
        for _ in range(rng.integers(2, 5)):
            r = rng.uniform(0.8, 2.0)
            cz = rng.uniform(zp.mean() + 5, H * mm - 3)
            cx = rng.uniform(max(xd + 0.5 * (cz - zp.mean()) + r + 1, 0), W * mm - r - 0.5)
            vaso = ((z - cz) ** 2 + (x - cx) ** 2 < r ** 2) & higado
            amp[vaso] = db(-64)
            lab[vaso] = Clase.VASO_ANECOICO
        amp[diafragma] = db(-12)
        lab[diafragma] = Clase.DIAFRAGMA
        region_pulmon &= ~(higado | diafragma)
        pleura_sobre_higado = pleura & (x > xd)
        lab[pleura_sobre_higado] = Clase.PARED_TORACICA
        amp[pleura_sobre_higado] = db(-35)

    # --- capa pulmonar móvil (coordenadas: profundidad bajo la superficie × ancho extendido)
    u = (np.arange(H) * mm)[:, None]
    xc = ((np.arange(Wc) - margen) * mm)[None, :]
    amp_pul = np.full((H, Wc), db(-50))
    lab_pul = np.full((H, Wc), Clase.PULMON, np.uint8)
    if humedad > 0:
        n_b = rng.poisson(humedad * Wc * mm * 0.5)
        for _ in range(n_b):
            cx, sx = rng.uniform(xc.min(), xc.max()), rng.uniform(0.35, 0.7)
            amp_pul += db(-8) * rng.uniform(0.6, 1.0) * np.exp(-0.5 * ((xc - cx) / sx) ** 2) * np.exp(-u / 25.0)
        amp_pul += humedad ** 2 * db(-16) * np.exp(-u / 25.0)
    for (c_mm, hw_mm, d_mm) in consolidaciones:
        rel = np.clip(1 - ((xc - c_mm) / hw_mm) ** 2, 0, None)
        irregular = gaussian_filter(rng.standard_normal(Wc), 2) * 0.6 * min(1.0, d_mm / 3)
        prof = np.where(rel > 0, d_mm * np.sqrt(rel) + irregular[None, :], 0)
        cons = u < prof
        amp_pul[cons] = db(-31)
        lab_pul[cons] = Clase.CONSOLIDACION
        if d_mm > 5:
            for _ in range(6):
                bz, bx = rng.uniform(1, d_mm * 0.7), rng.uniform(c_mm - hw_mm * 0.5, c_mm + hw_mm * 0.5)
                amp_pul[((u - bz) ** 2 + (xc - bx) ** 2 < 0.12) & cons] = db(-12)
    speckle_pul = _speckle(rng, (H, Wc))

    # líneas A: reverberaciones estáticas a múltiplos de la distancia piel–pleura, solo en columnas aireadas
    col_cons = np.zeros(Wc, bool)
    for (c_mm, hw_mm, _) in consolidaciones:
        col_cons |= np.abs(xc[0] - c_mm) < hw_mm
    lineas_a = np.zeros((H, W))
    dp = float(zp.mean())
    for k in range(2, 8):
        lineas_a += db(-20 - 7 * (k - 2)) * np.exp(-0.5 * ((z - k * dp) / 0.2) ** 2)
    lineas_a *= (1 - humedad) ** 1.5
    lineas_a = np.broadcast_to(lineas_a, (H, W)) * (grosor_derrame < 0.05)

    # --- costillas y sombras (se superponen a todo) -----------------------------
    sombra = np.zeros((H, W), bool)
    costilla = np.zeros((H, W), bool)
    if e.n_costillas > 0:
        centros = (np.arange(e.n_costillas) + 0.5) / e.n_costillas * W * mm
        centros += rng.uniform(-0.1, 0.1, e.n_costillas) * W * mm / max(e.n_costillas, 1)
        for c in centros:
            hw = rng.uniform(1.8, 2.6)
            dentro = np.abs(x - c) < hw
            zr = dp - 1.5 + 0.6 * ((x - c) / hw) ** 2
            costilla |= dentro & (z >= zr) & (z < zr + 0.4)
            sombra |= dentro & (z >= zr + 0.4)
    speckle_est = _speckle(rng, (H, W))

    # --- composición por frame ---------------------------------------------------
    T = e.n_frames
    fr_resp = rng.uniform(0.7, 1.0)                # Hz (≈ 40–60 rpm)
    s_px = np.clip(np.rint(z_sup[0] / mm), 0, H).astype(int)   # (W,) fila de la superficie pulmonar
    filas = np.arange(H)[:, None]
    u_idx = filas - s_px[None, :]
    dentro_pul = (u_idx >= 0) & region_pulmon
    u_idx = np.clip(u_idx, 0, H - 1)
    frames = np.empty((T, H, W), np.uint8)
    etiquetas = np.empty((T, H, W), np.uint8)
    for t in range(T):
        desp = 0 if e.neumotorax else int(round(4 * np.sin(2 * np.pi * fr_resp * t / e.fps)))
        cols = np.arange(W) + margen + desp
        a_pul = np.take_along_axis(amp_pul[:, cols], u_idx, axis=0)
        l_pul = np.take_along_axis(lab_pul[:, cols], u_idx, axis=0)
        s_pul = np.take_along_axis(speckle_pul[:, cols], u_idx, axis=0)
        aireada = ~col_cons[cols][None, :]
        a = np.where(dentro_pul, a_pul * s_pul + lineas_a * aireada, amp * speckle_est)
        l = np.where(dentro_pul, l_pul, lab)
        a = np.where(costilla, db(-8) * speckle_est, a)
        a = np.where(sombra, db(-62) * speckle_est, a)
        l = np.where(costilla, Clase.COSTILLA, l)
        l = np.where(sombra & ~costilla, Clase.SOMBRA_COSTAL, l)
        ruido = np.hypot(rng.standard_normal((H, W)), rng.standard_normal((H, W))) * db(-58)
        nivel = 20 * np.log10(a + ruido + 1e-9) - ATENUACION_NETA_DB_POR_MM * z + e.ganancia_db
        gris = 255.0 * (nivel + e.rango_dinamico_db) / e.rango_dinamico_db
        frames[t] = np.clip(np.rint(gris), 0, 255).astype(np.uint8)
        etiquetas[t] = l

    meta = {"escena": {k: v for k, v in e.__dict__.items()}, "sintetico": True,
            "rango_dinamico_db": e.rango_dinamico_db}
    return Clip(frames=frames, mm_por_pixel=(mm, mm), fps=e.fps, etiquetas=etiquetas, meta=meta)


def escena_aleatoria(rng: np.random.Generator, **fijos) -> Escena:
    """Escena variada para entrenamiento: cubre todo el espectro de hallazgos."""
    mm = rng.uniform(0.12, 0.18)
    ancho_mm = 256 * mm
    cons = []
    if rng.random() < 0.4:
        for _ in range(rng.integers(1, 4)):
            grande = rng.random() < 0.35
            cons.append((rng.uniform(0.1, 0.9) * ancho_mm,
                         rng.uniform(3, 9) if grande else rng.uniform(0.8, 2.5),
                         rng.uniform(7, 16) if grande else rng.uniform(1.0, 4.0)))
    p = dict(
        humedad=float(rng.uniform(0, 1)),
        n_costillas=int(rng.integers(0, 3)),
        consolidaciones=cons,
        derrame_mm=float(rng.uniform(1.5, 5)) if rng.random() < 0.15 else 0.0,
        neumotorax=bool(rng.random() < 0.1),
        higado=bool(rng.random() < 0.25),
        ganancia_db=float(rng.uniform(-4, 4)),
        rango_dinamico_db=float(rng.choice([50.0, 60.0, 70.0])),
        prof_pleura_mm=float(rng.uniform(2.5, 6.0)),
        mm_por_pixel=float(mm),
        semilla=int(rng.integers(0, 2**31 - 1)),
    )
    p.update(fijos)
    return Escena(**p)
