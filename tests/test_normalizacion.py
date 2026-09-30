import numpy as np
import pytest

from conftest import referencia
from qlus.cuantificacion import cuantificar_clip
from qlus.normalizacion import estimar_referencia
from qlus.sintetico import ATENUACION_NETA_DB_POR_MM, Escena, generar_clip


def test_recta_de_profundidad_del_higado(ref0):
    esperado = -ATENUACION_NETA_DB_POR_MM * 255 / 60
    assert ref0.pendiente == pytest.approx(esperado, rel=0.3)
    assert ref0.modo == "un_punto"
    assert ref0.heterogeneidad_db < 3.0 and not ref0.advertencias


@pytest.mark.parametrize("humedad", [0.2, 0.6])
def test_invariante_a_la_ganancia_del_examen(humedad):
    fba = []
    for g in (-4.0, 0.0, 4.0):
        ref = referencia(ganancia_db=g)
        c = generar_clip(Escena(n_frames=6, semilla=21, humedad=humedad, ganancia_db=g))
        fba.append(cuantificar_clip(c.frames, c.etiquetas, c.mm_por_pixel, ref, fps=c.fps).fba)
    assert max(fba) - min(fba) < 1.5


def test_sin_normalizacion_la_ganancia_si_cambia_el_resultado(ref0):
    """Control: la misma referencia con clips de distinta ganancia NO es invariante (y se advierte)."""
    a = generar_clip(Escena(n_frames=6, semilla=21, humedad=0.5, ganancia_db=0))
    b = generar_clip(Escena(n_frames=6, semilla=21, humedad=0.5, ganancia_db=5))
    ra = cuantificar_clip(a.frames, a.etiquetas, a.mm_por_pixel, ref0, fps=a.fps)
    rb = cuantificar_clip(b.frames, b.etiquetas, b.mm_por_pixel, ref0, fps=b.fps)
    assert rb.fba - ra.fba > 5
    assert any("ganancia" in adv for adv in rb.advertencias)


def test_modo_dos_puntos_corrige_otro_rango_dinamico():
    """El contraste hígado–vaso (dB) medido una vez con un preset conocido permite
    recuperar la escala dB/gris de otro preset sin conocer su rango dinámico."""
    a = generar_clip(Escena(higado=True, n_frames=6, semilla=8, rango_dinamico_db=90.0))
    ref_a = estimar_referencia(a.frames, a.etiquetas, a.mm_por_pixel, rango_dinamico_db=90.0)
    z = float(np.mean(ref_a.rango_prof_mm))
    contraste = (ref_a.gris_referencia(z) - ref_a.gris_anecoico) * ref_a.db_por_gris
    b = generar_clip(Escena(higado=True, n_frames=6, semilla=8, rango_dinamico_db=70.0))
    ref_b = estimar_referencia(b.frames, b.etiquetas, b.mm_por_pixel, contraste_tejido_sangre_db=contraste)
    assert ref_b.modo == "dos_puntos"
    assert ref_b.db_por_gris == pytest.approx(70 / 255, rel=0.1)


def test_dos_puntos_rechaza_vaso_recortado():
    c = generar_clip(Escena(higado=True, n_frames=4, semilla=8, rango_dinamico_db=50.0))
    with pytest.raises(ValueError):
        estimar_referencia(c.frames, c.etiquetas, c.mm_por_pixel, contraste_tejido_sangre_db=34.0)


def test_falla_si_no_hay_higado():
    c = generar_clip(Escena(n_frames=2, semilla=9))
    with pytest.raises(ValueError):
        estimar_referencia(c.frames, c.etiquetas, c.mm_por_pixel, rango_dinamico_db=60)
    assert np.isfinite(c.frames).all()
