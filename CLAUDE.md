# qLUS-Neo — contexto para Claude

LUS score neonatal cuantitativo: una U-Net segmenta cada píxel y una cuantificación determinística
calcula el IPA (0–100). Diseño en `docs/propuesta-tecnica.md`; comandos en `README.md`.

## Entorno (computador del usuario, Windows)

- Python del proyecto: `.venv\Scripts\python.exe` (CLI: `.venv\Scripts\qlus.exe`). Lo crea
  `scripts\preentrenar_biblioteca.ps1`. Pruebas: `.venv\Scripts\python.exe -m pytest` (requiere `pip install -e ".[dev]"`).
- Sin GPU salvo que se indique: el preentrenamiento tarda horas en CPU; lanzarlo en segundo plano.

## Datos de pacientes — reglas

- `datos/` y `modelos/` nunca se suben a git (están en `.gitignore`).
- No abrir los clips originales de la biblioteca (OneDrive) ni las imágenes de
  `datos\biblioteca\mosaicos\`, y no mostrar la columna `origen` de `datos\biblioteca\inventario.csv`:
  pueden contener nombres de pacientes. Para contar estados basta agrupar la columna `estado`.
- Para verificar el entrenamiento alcanzan `modelos\` y la salida de los comandos.

## Estado (1-oct-2026)

1. Hecho: `qlus biblioteca` procesó la biblioteca local de clips sin anotar → `datos\biblioteca\`.
2. En curso o interrumpido: `qlus preentrenar` (autosupervisado, sin anotaciones ni hígado).
   Existe `modelos\preentrenado.pt` pero faltaban `preentrenado.historial.json` y
   `preentrenado.ejemplos.png`, que solo se escriben al terminar. Primero verificar si sigue
   corriendo (fecha de `preentrenado.pt`, otras terminales); si se cortó, relanzar
   `qlus preentrenar --datos datos\biblioteca --salida modelos\preentrenado.pt` (parte de cero;
   `--epocas 10` para acortar). Éxito: pérdida de validación claramente bajo "sin modelo".
3. Siguiente: anotar 10–20 clips variados en CVAT (`docs/guia-anotacion.md`), importarlos con
   `qlus importar` a `datos\reales\` (`PACIENTE__zona.npz`), agregar sintéticos
   (`qlus sintetico --salida datos\reales --n 100`) y entrenar con
   `qlus entrenar --datos datos\reales --salida modelos\unet.pt --inicial modelos\preentrenado.pt`.
   Comparar contra el mismo entrenamiento sin `--inicial`.
4. Pendiente: un comando que exporte los `Bnnnnn.npz` anonimizados a PNG por frame para subirlos
   a CVAT (así no se suben los DICOM originales) con numeración compatible con `qlus importar`.

## Convenciones

- Código, comentarios y mensajes en español; seguir el estilo de los módulos existentes.
- `src/qlus/etiquetas.py` es el contrato de clases: no reordenar, solo agregar al final.
- La partición entrenamiento/validación es siempre por paciente (prefijo antes de `__`).
