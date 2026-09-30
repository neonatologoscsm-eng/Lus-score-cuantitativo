import pytest

from qlus.normalizacion import estimar_referencia
from qlus.sintetico import Escena, generar_clip


def referencia(ganancia_db=0.0, rango_dinamico_db=60.0, semilla=6):
    c = generar_clip(Escena(higado=True, humedad=0.2, n_frames=6, semilla=semilla,
                            ganancia_db=ganancia_db, rango_dinamico_db=rango_dinamico_db))
    return estimar_referencia(c.frames, c.etiquetas, c.mm_por_pixel, rango_dinamico_db=rango_dinamico_db)


@pytest.fixture(scope="session")
def ref0():
    return referencia()
