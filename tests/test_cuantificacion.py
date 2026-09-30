import pytest

from qlus.cuantificacion import cuantificar_clip, resumir_examen
from qlus.sintetico import Escena, generar_clip


def _zona(ref, **kw):
    c = generar_clip(Escena(n_frames=8, semilla=kw.pop("semilla", 11), **kw))
    return cuantificar_clip(c.frames, c.etiquetas, c.mm_por_pixel, ref, fps=c.fps)


def test_fba_crece_con_el_liquido(ref0):
    fba = [_zona(ref0, humedad=h).fba for h in (0.0, 0.5, 1.0)]
    assert fba[0] < 15 and fba[2] > 80
    assert fba[0] < fba[1] < fba[2]


def test_consolidacion_subpleural_vs_extensa(ref0):
    sub = _zona(ref0, humedad=0.3, consolidaciones=[(19, 1.5, 2.5)])
    ext = _zona(ref0, humedad=0.3, consolidaciones=[(19, 5.0, 12.0)])
    assert sub.cons_subpleural_pct > 0 and sub.cons_extensa_pct == 0
    assert ext.cons_extensa_pct > 0 and ext.cons_prof_max_mm > 10
    # la consolidación cuenta como tejido sin aire: IPA ≥ FBA
    assert ext.ipa >= ext.fba and ext.ipa == pytest.approx(
        ext.cons_extensa_pct + ext.cons_subpleural_pct
        + (100 - ext.cons_extensa_pct - ext.cons_subpleural_pct) * ext.fba / 100, abs=3)


def test_derrame_se_reporta_y_no_entra(ref0):
    r = _zona(ref0, humedad=0.3, derrame_mm=3.0)
    assert r.estado == "cuantificado"
    assert r.derrame_max_mm == pytest.approx(3.0, abs=0.8)
    assert r.indice_deslizamiento > 0.05   # el pulmón se mueve bajo el líquido


def test_neumotorax_no_se_cuantifica(ref0):
    r = _zona(ref0, neumotorax=True)
    assert r.estado == "no_cuantificable_ntx" and r.ipa is None
    normal = _zona(ref0, humedad=0.0)
    assert normal.estado == "cuantificado" and normal.indice_deslizamiento > 0.05


def test_resumen_examen(ref0):
    zonas = [_zona(ref0, humedad=0.2), _zona(ref0, humedad=0.8), _zona(ref0, neumotorax=True)]
    for z, n in zip(zonas, ("a", "b", "c")):
        z.zona = n
    res = resumir_examen(zonas, ref0)
    assert res["zonas_cuantificadas"] == 2 and res["zonas_sospecha_ntx"] == ["c"]
    assert res["ipa_global"] == pytest.approx((zonas[0].ipa + zonas[1].ipa) / 2)
