"""Ontología de segmentación: una clase por píxel, mutuamente excluyentes.

Estos identificadores son el contrato entre la anotación (CVAT, 3D Slicer),
el entrenamiento, la exportación a nnU-Net y la cuantificación. No reordenar:
agregar clases nuevas solo al final.
"""

from enum import IntEnum


class Clase(IntEnum):
    FONDO = 0            # fuera del campo o sin contacto
    PARED_TORACICA = 1   # piel, subcutáneo, músculo
    COSTILLA = 2         # cortical/cartílago costal
    SOMBRA_COSTAL = 3    # sombra acústica posterior a la costilla
    LINEA_PLEURAL = 4
    PULMON = 5           # parénquima con artefactos (líneas A, líneas B, pulmón blanco)
    CONSOLIDACION = 6    # tejido sin aire; subpleural/extensa se decide por profundidad
    DERRAME = 7
    HIGADO = 8
    BAZO = 9
    CORAZON = 10
    TIMO = 11
    DIAFRAGMA = 12
    VASO_ANECOICO = 13   # venas hepáticas, porta, cava: referencia anecoica


N_CLASES = len(Clase)

# Píxel (o frame completo) sin anotar: se ignora en el entrenamiento.
SIN_ANOTAR = 255

NOMBRES = {c.value: c.name.lower() for c in Clase}

# Colores RGB para superposiciones y para configurar la herramienta de anotación.
COLORES = {
    Clase.FONDO: (0, 0, 0),
    Clase.PARED_TORACICA: (120, 90, 60),
    Clase.COSTILLA: (255, 255, 255),
    Clase.SOMBRA_COSTAL: (90, 90, 90),
    Clase.LINEA_PLEURAL: (255, 220, 0),
    Clase.PULMON: (0, 170, 255),
    Clase.CONSOLIDACION: (230, 30, 30),
    Clase.DERRAME: (40, 60, 255),
    Clase.HIGADO: (150, 60, 20),
    Clase.BAZO: (170, 80, 170),
    Clase.CORAZON: (255, 120, 160),
    Clase.TIMO: (255, 160, 60),
    Clase.DIAFRAGMA: (220, 220, 140),
    Clase.VASO_ANECOICO: (0, 90, 90),
}

# Clases que nunca forman parte del cálculo de aireación.
NO_PULMONARES = (
    Clase.FONDO, Clase.PARED_TORACICA, Clase.COSTILLA, Clase.SOMBRA_COSTAL,
    Clase.DERRAME, Clase.HIGADO, Clase.BAZO, Clase.CORAZON, Clase.TIMO,
    Clase.DIAFRAGMA, Clase.VASO_ANECOICO,
)
