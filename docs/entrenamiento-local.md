# Entrenamiento en el computador local con una biblioteca de clips sin anotar

Los clips de pacientes se quedan en el computador donde están: todo se ejecuta ahí y nada se sube.
`datos/` y `modelos/` están excluidos de git.

## Qué se puede entrenar sin anotaciones (y qué no)

| Paso | Necesita | Qué aprende |
|---|---|---|
| **Preentrenamiento autosupervisado** (`qlus preentrenar`) | Solo clips, **sin anotar y sin hígado** | Cómo se ven en clips reales la pleura, las costillas y sus sombras, las líneas A y B, el pulmón blanco y las consolidaciones. La red reconstruye parches borrados del frame t a partir del resto de la imagen y de los frames t−1 y t+1 |
| **Segmentación** (`qlus entrenar`) | Frames anotados por píxel ([guía de anotación](guia-anotacion.md)) | Qué es cada píxel. Parte de los pesos preentrenados (`--inicial`), así que necesita menos anotación |
| **Cuantificación** (`qlus analizar`) | Modelo de segmentación + **clip hepático** del examen | El IPA. El hígado solo se necesita aquí, para normalizar la intensidad |

El preentrenamiento **no** produce un modelo que segmente ni que puntúe: es el punto de partida para
que la segmentación aprenda con pocos clips anotados.

## Pasos en Windows

1. Instalar [Python 3.11 o 3.12](https://www.python.org/downloads/) (marcar *Add python.exe to PATH*)
   y [Git](https://git-scm.com/download/win).
2. En PowerShell, descargar el repositorio (una vez):

   ```powershell
   cd $HOME\Documents
   git clone https://github.com/neonatologoscsm-eng/Lus-score-cuantitativo.git
   cd Lus-score-cuantitativo
   ```

3. Ejecutar todo (entorno, instalación, biblioteca y preentrenamiento):

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\preentrenar_biblioteca.ps1 `
     -Biblioteca "C:\Users\<usuario>\OneDrive\...\Biblioteca Ecografias pulmonares"
   ```

   Con `-SoloIngesta` se prepara la biblioteca sin entrenar, para revisarla antes.

Equivalente sin el script (con el entorno activado):

```powershell
qlus biblioteca --origen "C:\...\Biblioteca Ecografias pulmonares" --salida datos\biblioteca
qlus preentrenar --datos datos\biblioteca --salida modelos\preentrenado.pt
```

## Qué hace `qlus biblioteca`

- Recorre la carpeta **con subcarpetas** y toma DICOM (con o sin extensión) y videos
  (mp4, avi, mov, wmv, mkv, m4v, mpg, gif). Las imágenes fijas (jpg/png) no se usan.
- **Anonimiza**: guarda solo los píxeles de la región de imagen, con nombres `B00001.npz`,
  `B00002.npz`, etc. No copia el encabezado DICOM ni el nombre del archivo. En los videos, la región
  es el rectángulo que cambia en el tiempo, lo que deja fuera textos y menús fijos alrededor.
- **Descarta** Doppler color, duplicados y archivos dañados, y anota el motivo en `inventario.csv`.
- Guarda hasta 64 frames consecutivos por clip, con el lado mayor reducido a 384 px.
- **Se puede reanudar**: lo ya procesado no se repite, y los archivos con error se reintentan.

Revisar después:

1. **`datos\biblioteca\mosaicos\`**: miniaturas de cada clip. Verificar que no quede texto con datos
   del paciente dentro de la imagen (el texto *sobre* la imagen no se puede recortar). Si aparece,
   borrar ese `Bnnnnn.npz`.
2. **`datos\biblioteca\inventario.csv`** (se abre en Excel): omitidos y errores. Contiene las rutas
   originales, así que no debe compartirse. Si hay errores del tipo *Unable to decompress*, instalar
   `pip install -e ".[jpeg]"` (el script ya lo hace) y volver a ejecutar.

OneDrive: los archivos "solo en línea" se descargan al leerlos, lo que puede tardar. Para evitarlo,
usar *Mantener siempre en este dispositivo* en la carpeta.

## Qué hace `qlus preentrenar` y cómo leer el resultado

- Por cada época imprime la pérdida de validación y la de referencia **sin modelo** (rellenar lo
  borrado con gris). Que la pérdida de validación quede claramente por debajo de la referencia
  indica que la red aprende la estructura de los clips.
- Guarda el mejor modelo en `modelos\preentrenado.pt` en cada época que mejora, así que se puede
  interrumpir con Ctrl+C sin perder lo avanzado.
- `modelos\preentrenado.ejemplos.png`: en cada fila, el frame con parches borrados, la
  reconstrucción y el original.
- Duración: en CPU, cerca de 2,5 s por iteración en 4 núcleos. Con los valores por defecto
  (30 épocas × 200 iteraciones) son de 2 a 4 horas en un portátil; con `--epocas 10` es un tercio.
  Con una GPU NVIDIA son minutos (el script indica cómo instalar PyTorch con CUDA).
- Cada época carga 100 clips al azar (`--clips-en-memoria`, cerca de 1 GB de RAM), así el tamaño
  de la biblioteca no está limitado por la memoria.

## Siguiente paso: segmentación con los pesos preentrenados

1. Elegir 10–20 clips variados de la biblioteca (normal, líneas B, pulmón blanco, consolidación,
   derrame, neumotórax), anotarlos en CVAT según la [guía](guia-anotacion.md) e importarlos con
   `qlus importar` a `datos\reales\` con nombres `PACIENTE__zona.npz`.
2. Mientras no haya clips hepáticos anotados, agregar clips sintéticos a la misma carpeta para que
   la red no olvide las clases que faltan (hígado, vasos):
   `qlus sintetico --salida datos\reales --n 100` (quedan como pacientes `S00000`… y la partición
   sigue siendo por paciente).
3. Entrenar partiendo del preentrenamiento:

   ```powershell
   qlus entrenar --datos datos\reales --salida modelos\unet.pt --inicial modelos\preentrenado.pt
   ```

   `--inicial` copia todos los pesos salvo la capa final y fija la arquitectura (`base`, `niveles`)
   del modelo preentrenado.
