# qLUS-Neo — LUS score neonatal cuantitativo

Proyecto para desarrollar un algoritmo que entregue un score de ecografía pulmonar neonatal
**continuo, objetivo y reproducible**, en lugar del score visual semicuantitativo (0–3 por zona)
tipo Brat.

Idea central: medir la **pérdida de aireación** de cada zona pulmonar a partir de la "blancura"
calibrada de la imagen, después de:

- identificar la línea pleural y el parénquima pulmonar,
- excluir las sombras acústicas costales y los órganos no pulmonares (timo, corazón, hígado, bazo),
- ponderar de forma distinta las consolidaciones subpleurales y extensas,
- separar el derrame pleural y detectar el neumotórax (que no debe leerse como "pulmón normal").

## Documentación

- [Propuesta técnica](docs/propuesta-tecnica.md): premisa física, arquitectura, fórmulas,
  estrategia IA/reglas, datos, validación, estado del arte y hoja de ruta.
