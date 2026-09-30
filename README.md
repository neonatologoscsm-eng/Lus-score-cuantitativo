# qLUS-Neo — LUS score neonatal cuantitativo

Algoritmo para obtener un score de ecografía pulmonar neonatal **continuo, objetivo y puramente
ecográfico**, en lugar del score visual semicuantitativo (0–3 por zona) tipo Brat.

- **La IA segmenta** cada píxel: pared, costillas, sombras costales, pleura, pulmón, consolidación,
  derrame, hígado, timo, corazón, etc.
- **La cuantificación es determinística**: mide la "blancura" del pulmón en una banda fija bajo la
  pleura, cuenta la consolidación como tejido sin aire, excluye derrame y sombras, y marca el
  neumotórax como no cuantificable.
- **Las sombras costales se eliminan de forma autónoma** (IA + red de seguridad por intensidad +
  consistencia temporal).
- **La intensidad se normaliza con el hígado del propio paciente** (clip hepático del mismo examen),
  sin fantoma externo.

Salida principal por zona y examen: **IPA — Índice de Pérdida de Aireación (0–100)**, con FBA
(fracción de blanco aparente), CPB (cobertura pleural por patrón B), consolidación subpleural y
extensa, derrame, deslizamiento pleural y controles de calidad.

## Documentación

- [Propuesta técnica](docs/propuesta-tecnica.md): física, arquitectura, normalización hepática,
  exclusión de sombras, fórmulas, validación y hoja de ruta.
- [Guía de anotación](docs/guia-anotacion.md) y [etiquetas para CVAT](docs/etiquetas_cvat.json).
- [Entrenamiento local con una biblioteca de clips sin anotar](docs/entrenamiento-local.md)
  (Windows, paso a paso).

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Uso

```bash
# 1. Datos sintéticos para probar la tubería y preentrenar
qlus sintetico --salida datos/sinteticos --n 200
qlus sintetico --tipo examen --salida datos/examen_demo

# 2. Preentrenamiento autosupervisado con clips reales SIN anotar (DICOM/video, sin hígado)
qlus biblioteca --origen "ruta/a/la/biblioteca" --salida datos/biblioteca
qlus preentrenar --datos datos/biblioteca --salida modelos/preentrenado.pt

# 3. Entrenar la segmentación (clips .npz "PACIENTE__zona.npz" con máscaras)
qlus entrenar --datos datos/sinteticos --salida modelos/unet.pt --epocas 20 \
  --inicial modelos/preentrenado.pt   # opcional

# 4. Analizar un examen: clip hepático + zonas, con el rango dinámico del preset
qlus analizar --modelo modelos/unet.pt --rango-dinamico 60 \
  --higado datos/examen_demo/DEMO__higado.npz \
  --zona anterior_superior_der=datos/examen_demo/DEMO__anterior_superior_der.npz \
  --zona lateral_der=datos/examen_demo/DEMO__lateral_der.npz \
  --figuras salida/figuras --salida salida/resultado.json

# Datos reales: clip DICOM + máscaras exportadas de CVAT → .npz
qlus importar --clip clip.dcm --mascaras export/SegmentationClass/ --salida datos/P001__lateral_der.npz

# Exportar a nnU-Net v2 (folds agrupados por paciente)
qlus exportar-nnunet --datos datos/reales --salida nnUNet_raw
```

Con `--comparar` (y clips anotados), `analizar` informa el IPA calculado con las máscaras de la IA
junto al calculado con las del experto: es la métrica clave para aceptar el modelo.

## Demostración con datos sintéticos

U-Net liviana (1,9 M parámetros) entrenada 10 épocas en CPU con 160 clips sintéticos: Dice medio
de validación 0,94 (sombra costal 0,99, pleura 0,87, consolidación 0,94, derrame 0,94, hígado 0,99).
En un examen sintético nuevo (hígado + 6 zonas), el IPA calculado con la segmentación de la IA
difiere menos de 1 punto del calculado con las máscaras exactas en todas las zonas, y la zona con
neumotórax queda "no cuantificable" en ambos casos.

![Láminas de control: consolidación subpleural, consolidación extensa y neumotórax; columnas
excluidas por sombra costal en rayado](docs/img/ejemplo_ia_sintetico.png)

Esto demuestra que la tubería funciona de punta a punta; **no** demuestra desempeño en clips reales.

## Estado

Tubería completa probada con datos sintéticos. **Aún no hay validación con clips reales**: las anclas
de la escala (−20/+15 dB respecto al hígado) y los umbrales son provisionales y se fijarán con los
datos de desarrollo antes de cualquier uso clínico. Uso exclusivo de investigación.
