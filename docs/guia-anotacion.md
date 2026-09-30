# Guía de anotación — qLUS-Neo

Objetivo: máscaras por píxel para entrenar la segmentación. Cada píxel anotado recibe **una** clase.
La calidad de estas máscaras define el techo de la IA: ante la duda, marcar **ignorar**.

## Preparación

1. Crear un proyecto en CVAT con las etiquetas de `docs/etiquetas_cvat.json`
   (Proyecto → Constructor "Raw" → pegar el contenido). Los colores deben coincidir exactamente.
2. Subir los clips **anonimizados** (sin nombre, RUT ni fecha en la imagen).
3. Nombre de cada tarea/clip: `PACIENTE__zona`, p. ej. `P007__lateral_der`.

## Qué frames anotar

- Anotación **dispersa**: 1 de cada 5 frames (6–10 por clip), incluyendo inspiración y espiración.
  Los frames no anotados se usan igual como contexto temporal.
- Cada frame anotado se anota **completo** (todo el campo), no solo el hallazgo.

## Reglas por clase

| Clase | Qué incluir | Errores frecuentes |
|---|---|---|
| `pared_toracica` | Piel, subcutáneo y músculo, hasta justo sobre la pleura o la costilla | Incluir la pleura |
| `costilla` | Arco cortical hiperecogénico y el cartílago costal visible (en neonatos puede ser hipoecoico) | Olvidar el cartílago |
| `sombra_costal` | Todo lo que está bajo la costilla **hasta el fondo de la imagen**, entre sus bordes laterales, incluida la penumbra sin pleura visible | Cortarla antes del fondo |
| `linea_pleural` | Solo la línea brillante, con su grosor real, donde sea visible. Con derrame: la pleura parietal | Dibujarla bajo la sombra |
| `pulmon` | Bajo la pleura, entre sombras, hasta el fondo: patrón A, líneas B, pulmón blanco | Marcar líneas B como otra clase |
| `consolidacion` | Tejido de aspecto sólido (gris, tipo hígado) desde la pleura hasta su borde profundo irregular; incluye broncogramas. Grandes y **también las subpleurales pequeñas** | Decidir "subpleural o extensa": no se decide, lo mide el algoritmo |
| `derrame` | Espacio anecoico entre pleura parietal y superficie pulmonar | Confundir con sombra costal |
| `higado` / `bazo` | Parénquima completo | Incluir vasos grandes |
| `vaso_anecoico` | Venas hepáticas, porta o cava dentro del hígado (≥ 1 mm) | — |
| `corazon` | Corazón en zonas anteriores inferiores izquierdas | — |
| `timo` | Zonas anteriores superiores: homogéneo con puntos ecogénicos | **Marcarlo como consolidación** |
| `diafragma` | Línea curva brillante entre pulmón y víscera | — |
| `ignorar` | Regiones dudosas o con artefacto no clasificable | Usarlo en exceso |
| (sin etiqueta) | Fuera del campo o sin contacto de la sonda | — |

## Control de calidad

- Un segundo anotador repite el 20 % de los frames; se calcula el Dice entre anotadores por clase.
  Las clases con Dice < 0,7 se discuten y se ajusta esta guía.
- Revisar especialmente: sombras (deben llegar al fondo), timo y consolidaciones pequeñas.

## Exportación e importación

1. En CVAT: *Export task dataset* → formato **Segmentation mask 1.1**.
2. Importar cada clip con sus máscaras (carpeta `SegmentationClass/`):

```bash
qlus importar --clip P007_lateral_der.dcm --mascaras export/SegmentationClass/ \
              --salida datos/P007__lateral_der.npz
```

Si los archivos de máscara numeran los frames desde 1, agregar `--desfase-frame 1`.
