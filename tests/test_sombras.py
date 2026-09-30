import numpy as np

from qlus.etiquetas import Clase
from qlus.postproceso import columnas_excluidas
from qlus.sintetico import Escena, generar_clip


def _columnas_sombra_reales(etiquetas):
    return (etiquetas == Clase.SOMBRA_COSTAL).mean(axis=1) > 0.5   # (T, W)


def test_excluye_sombras_segmentadas():
    c = generar_clip(Escena(n_frames=6, n_costillas=2, humedad=0.4, semilla=4))
    excl, info = columnas_excluidas(c.frames, c.etiquetas, c.mm_por_pixel)
    reales = _columnas_sombra_reales(c.etiquetas)
    assert excl[reales].mean() > 0.99
    assert info["frac_columnas_excluidas"] < 0.45


def test_red_de_seguridad_si_la_ia_no_ve_la_sombra():
    """La IA 'alucina' pulmón y pleura continua sobre la costilla: la intensidad igual la excluye."""
    c = generar_clip(Escena(n_frames=6, n_costillas=2, humedad=0.4, semilla=5))
    reales = _columnas_sombra_reales(c.etiquetas)
    lab = c.etiquetas.copy()
    lab[np.isin(lab, (Clase.COSTILLA, Clase.SOMBRA_COSTAL))] = Clase.PULMON
    for t in range(lab.shape[0]):
        filas_pl = np.nonzero((c.etiquetas[t] == Clase.LINEA_PLEURAL).any(axis=1))[0]
        banda = slice(filas_pl.min(), filas_pl.max() + 1)
        cols = reales[t]
        lab[t, banda][:, cols] = Clase.LINEA_PLEURAL
    excl, info = columnas_excluidas(c.frames, lab, c.mm_por_pixel)
    assert excl[reales].mean() > 0.9
    assert info["frac_por_red_seguridad"] > 0.1


def test_sin_costillas_no_excluye():
    c = generar_clip(Escena(n_frames=4, n_costillas=0, humedad=0.5, semilla=6))
    excl, _ = columnas_excluidas(c.frames, c.etiquetas, c.mm_por_pixel)
    assert excl.mean() < 0.02
