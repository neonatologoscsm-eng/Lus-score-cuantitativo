"""Línea de comandos de qLUS-Neo.

  qlus sintetico      genera clips sintéticos (entrenamiento o un examen de demostración)
  qlus biblioteca     prepara una biblioteca de clips SIN anotar (DICOM/video) → .npz anonimizados
  qlus preentrenar    preentrenamiento autosupervisado con esos clips (no requiere anotación)
  qlus entrenar       entrena la segmentación con clips .npz anotados
  qlus analizar       segmenta (IA o anotación) y cuantifica un examen completo
  qlus exportar-nnunet  exporta clips anotados a formato nnU-Net v2
  qlus importar       une un clip DICOM/video con máscaras PNG anotadas → .npz
  qlus etiquetas-cvat escribe la configuración de etiquetas para CVAT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .cuantificacion import Parametros, cuantificar_clip, resumir_examen
from .etiquetas import SIN_ANOTAR
from .io import cargar_clip, guardar_npz
from .normalizacion import estimar_referencia
from .postproceso import columnas_excluidas
from .sintetico import Escena, escena_aleatoria, generar_clip
from .visualizacion import lamina_resumen

ZONAS_DEMO = {
    "anterior_superior_der": dict(humedad=0.35),
    "anterior_inferior_der": dict(humedad=0.55, consolidaciones=[(17.0, 1.5, 3.0), (24.0, 1.0, 2.5)]),
    "lateral_der": dict(humedad=0.8, consolidaciones=[(20.0, 6.0, 12.0)]),
    "anterior_superior_izq": dict(humedad=0.3),
    "anterior_inferior_izq": dict(neumotorax=True),
    "lateral_izq": dict(humedad=0.6, derrame_mm=3.0),
}


def _cmd_sintetico(a):
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.semilla)
    if a.tipo == "examen":
        comunes = dict(ganancia_db=float(rng.uniform(-3, 3)), rango_dinamico_db=a.rango_dinamico,
                       n_frames=a.frames, prof_pleura_mm=float(rng.uniform(3, 5)))
        e = Escena(higado=True, humedad=0.2, semilla=int(rng.integers(1 << 30)), **comunes)
        guardar_npz(generar_clip(e), salida / "DEMO__higado.npz")
        for zona, kw in ZONAS_DEMO.items():
            e = Escena(semilla=int(rng.integers(1 << 30)), **comunes, **kw)
            guardar_npz(generar_clip(e), salida / f"DEMO__{zona}.npz")
        print(f"Examen sintético (hígado + {len(ZONAS_DEMO)} zonas) en {salida}")
        return
    for i in range(a.n):
        e = escena_aleatoria(rng, n_frames=a.frames)
        guardar_npz(generar_clip(e), salida / f"S{i:05d}__clip.npz")
    print(f"{a.n} clips sintéticos en {salida}")


def _cmd_biblioteca(a):
    from .biblioteca import ingerir

    r = ingerir(a.origen, a.salida, max_frames=a.max_frames, lado_max=a.lado_max, umbral_color=a.umbral_color,
                mosaicos=not a.sin_mosaicos)
    print(f"→ {a.salida}  (detalle por archivo en {r['inventario']})")


def _cmd_preentrenar(a):
    from .preentrenamiento import preentrenar

    r = preentrenar(a.datos, a.salida, epocas=a.epocas, tamano=a.tamano, lote=a.lote, base=a.base,
                    niveles=a.niveles, iter_por_epoca=a.iter_por_epoca, lr=a.lr,
                    clips_en_memoria=a.clips_en_memoria, semilla=a.semilla, hilos=a.hilos)
    print(f"Mejor pérdida de reconstrucción en validación: {r['mejor_perdida_val']:.4f} "
          f"(sin modelo {r['perdida_val_sin_modelo']:.4f})  → {a.salida}")


def _cmd_entrenar(a):
    from .entrenamiento import entrenar

    r = entrenar(a.datos, a.salida, epocas=a.epocas, tamano=a.tamano, lote=a.lote, base=a.base,
                 iter_por_epoca=a.iter_por_epoca, lr=a.lr, semilla=a.semilla, hilos=a.hilos, inicial=a.inicial)
    print(f"Mejor Dice medio en validación: {r['mejor_dice_medio']:.3f}  → {a.salida}")


def _segmentador(a):
    if a.modelo:
        from .inferencia import cargar_modelo, segmentar_clip

        modelo, meta = cargar_modelo(a.modelo)
        return lambda clip: segmentar_clip(modelo, meta, clip.frames)

    def usar_anotacion(clip):
        if clip.etiquetas is None or (clip.etiquetas == SIN_ANOTAR).all(axis=(1, 2)).any():
            raise SystemExit(f"{clip.meta.get('fuente')}: sin anotación completa de todos los frames; use --modelo")
        return clip.etiquetas
    return usar_anotacion


def _cmd_analizar(a):
    mm = tuple(a.mm_por_pixel) if a.mm_por_pixel else None
    segmentar = _segmentador(a)
    p = Parametros()
    hig = cargar_clip(a.higado, mm)
    lab_hig = segmentar(hig)
    ref = estimar_referencia(hig.frames, lab_hig, hig.mm_por_pixel, rango_dinamico_db=a.rango_dinamico,
                             contraste_tejido_sangre_db=a.contraste_higado_sangre)
    zonas, comparacion = [], {}
    figuras = Path(a.figuras) if a.figuras else None
    if figuras:
        figuras.mkdir(parents=True, exist_ok=True)
    for spec in a.zona:
        nombre, ruta = spec.split("=", 1)
        clip = cargar_clip(ruta, mm)
        lab = segmentar(clip)
        r = cuantificar_clip(clip.frames, lab, clip.mm_por_pixel, ref, zona=nombre, fps=clip.fps, p=p)
        zonas.append(r)
        if a.modelo and a.comparar and clip.etiquetas is not None:
            r_gt = cuantificar_clip(clip.frames, clip.etiquetas, clip.mm_por_pixel, ref, zona=nombre, fps=clip.fps, p=p)
            comparacion[nombre] = {"ipa_ia": r.ipa, "ipa_anotacion": r_gt.ipa,
                                   "estado_ia": r.estado, "estado_anotacion": r_gt.estado}
        if figuras:
            t = clip.frames.shape[0] // 2
            excl, _ = columnas_excluidas(clip.frames, lab, clip.mm_por_pixel, p.sombras)
            fmt = lambda v: "-" if v is None else f"{v:.1f}"
            texto = [f"{nombre}  estado: {r.estado}",
                     f"IPA {fmt(r.ipa)}  FBA {fmt(r.fba)}  CPB {fmt(r.cpb)}  "
                     f"cons sub {fmt(r.cons_subpleural_pct)}%  ext {fmt(r.cons_extensa_pct)}%",
                     f"derrame {r.derrame_max_mm:.1f} mm  deslizamiento {fmt(r.indice_deslizamiento)}  "
                     f"columnas excluidas {r.sombras['frac_columnas_excluidas']:.0%}"]
            cv2.imwrite(str(figuras / f"{nombre}.png"), lamina_resumen(clip.frames[t], lab[t], excl[t], texto))
    resumen = resumir_examen(zonas, ref)
    if comparacion:
        resumen["comparacion_ia_vs_anotacion"] = comparacion
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    with open(a.salida, "w") as f:
        json.dump(resumen, f, indent=1, ensure_ascii=False, default=float)
    print(f"IPA global: {resumen['ipa_global'] if resumen['ipa_global'] is None else round(resumen['ipa_global'], 1)}"
          f"  ({resumen['zonas_cuantificadas']}/{resumen['zonas_totales']} zonas)")
    for z in zonas:
        ipa = "—" if z.ipa is None else f"{z.ipa:5.1f}"
        print(f"  {z.zona:24s} {z.estado:22s} IPA {ipa}")
        for adv in z.advertencias:
            print(f"      ! {adv}")
    for adv in ref.advertencias:
        print(f"  ! referencia: {adv}")
    print(f"→ {a.salida}")


def _cmd_exportar(a):
    from .nnunet import exportar

    raiz = exportar(a.datos, a.salida, id_dataset=a.id, nombre=a.nombre, cada_n_frames=a.cada_n)
    print(f"Dataset nnU-Net en {raiz} (copiar splits_final.json a nnUNet_preprocessed/{raiz.name}/)")


def _cmd_importar(a):
    from .anotacion import importar_mascaras

    clip = cargar_clip(a.clip, tuple(a.mm_por_pixel) if a.mm_por_pixel else None)
    clip = importar_mascaras(clip, a.mascaras, desfase_frame=a.desfase_frame)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    guardar_npz(clip, a.salida)
    print(f"{clip.meta['frames_anotados']}/{clip.frames.shape[0]} frames anotados → {a.salida}")


def _cmd_etiquetas_cvat(a):
    from .anotacion import etiquetas_cvat

    with open(a.salida, "w") as f:
        json.dump(etiquetas_cvat(), f, indent=1, ensure_ascii=False)
    print(f"Etiquetas CVAT → {a.salida}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qlus", description="LUS score neonatal cuantitativo (qLUS-Neo)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sintetico", help="genera clips sintéticos")
    s.add_argument("--salida", required=True)
    s.add_argument("--tipo", choices=["entrenamiento", "examen"], default="entrenamiento")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--frames", type=int, default=16)
    s.add_argument("--rango-dinamico", type=float, default=60.0)
    s.add_argument("--semilla", type=int, default=0)
    s.set_defaults(func=_cmd_sintetico)

    s = sub.add_parser("biblioteca", help="prepara clips sin anotar para el preentrenamiento")
    s.add_argument("--origen", required=True, help="carpeta con los clips (se recorre con subcarpetas)")
    s.add_argument("--salida", required=True)
    s.add_argument("--max-frames", type=int, default=64, help="frames consecutivos por clip (tramo central)")
    s.add_argument("--lado-max", type=int, default=384, help="px del lado mayor tras reducir")
    s.add_argument("--umbral-color", type=float, default=0.003, help="fracción de píxeles en color para descartar")
    s.add_argument("--sin-mosaicos", action="store_true", help="no generar las hojas de miniaturas")
    s.set_defaults(func=_cmd_biblioteca)

    s = sub.add_parser("preentrenar", help="preentrenamiento autosupervisado (sin anotación)")
    s.add_argument("--datos", required=True, help="carpeta generada por 'qlus biblioteca'")
    s.add_argument("--salida", required=True)
    s.add_argument("--epocas", type=int, default=30)
    s.add_argument("--tamano", type=int, default=256)
    s.add_argument("--lote", type=int, default=8)
    s.add_argument("--base", type=int, default=32)
    s.add_argument("--niveles", type=int, default=4)
    s.add_argument("--iter-por-epoca", type=int, default=200)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--clips-en-memoria", type=int, default=100, help="clips cargados por época (~7 MB c/u)")
    s.add_argument("--hilos", type=int, default=None)
    s.add_argument("--semilla", type=int, default=0)
    s.set_defaults(func=_cmd_preentrenar)

    s = sub.add_parser("entrenar", help="entrena la segmentación")
    s.add_argument("--datos", required=True)
    s.add_argument("--salida", required=True)
    s.add_argument("--epocas", type=int, default=20)
    s.add_argument("--tamano", type=int, default=256)
    s.add_argument("--lote", type=int, default=8)
    s.add_argument("--base", type=int, default=32)
    s.add_argument("--iter-por-epoca", type=int, default=200)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--hilos", type=int, default=None)
    s.add_argument("--semilla", type=int, default=0)
    s.add_argument("--inicial", help="pesos iniciales (p. ej. de 'qlus preentrenar'); fija base y niveles")
    s.set_defaults(func=_cmd_entrenar)

    s = sub.add_parser("analizar", help="cuantifica un examen")
    s.add_argument("--higado", required=True, help="clip de referencia hepática del mismo examen y preset")
    s.add_argument("--zona", action="append", required=True, help="NOMBRE=ruta_clip (repetible)")
    s.add_argument("--modelo", help="modelo .pt; sin él se usan las etiquetas anotadas del .npz")
    s.add_argument("--rango-dinamico", type=float, default=None, help="rango dinámico del preset en dB")
    s.add_argument("--contraste-higado-sangre", type=float, default=None,
                   help="modo dos puntos: contraste hígado–vaso en dB")
    s.add_argument("--mm-por-pixel", type=float, nargs=2, help="solo para videos sin metadatos: fila columna")
    s.add_argument("--comparar", action="store_true", help="con --modelo, compara contra la anotación del .npz")
    s.add_argument("--figuras", help="carpeta para láminas de control")
    s.add_argument("--salida", default="resultado_qlus.json")
    s.set_defaults(func=_cmd_analizar)

    s = sub.add_parser("exportar-nnunet", help="exporta a formato nnU-Net v2")
    s.add_argument("--datos", required=True)
    s.add_argument("--salida", required=True)
    s.add_argument("--id", type=int, default=501)
    s.add_argument("--nombre", default="LUSNeo")
    s.add_argument("--cada-n", type=int, default=1)
    s.set_defaults(func=_cmd_exportar)

    s = sub.add_parser("importar", help="clip + máscaras PNG anotadas → .npz")
    s.add_argument("--clip", required=True)
    s.add_argument("--mascaras", required=True, help="carpeta con un PNG por frame anotado (número de frame en el nombre)")
    s.add_argument("--salida", required=True, help="PACIENTE__zona.npz")
    s.add_argument("--mm-por-pixel", type=float, nargs=2)
    s.add_argument("--desfase-frame", type=int, default=0, help="restar al número del archivo (p. ej. 1 si parte en 1)")
    s.set_defaults(func=_cmd_importar)

    s = sub.add_parser("etiquetas-cvat", help="configuración de etiquetas para CVAT")
    s.add_argument("--salida", default="etiquetas_cvat.json")
    s.set_defaults(func=_cmd_etiquetas_cvat)

    a = ap.parse_args(argv)
    if a.cmd == "analizar" and a.rango_dinamico is None and a.contraste_higado_sangre is None:
        ap.error("analizar requiere --rango-dinamico (preset) o --contraste-higado-sangre")
    a.func(a)


if __name__ == "__main__":
    main()
