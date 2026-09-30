"""Ingesta de una biblioteca de clips SIN anotar (DICOM y video) para el preentrenamiento.

Recorre la carpeta de forma recursiva y convierte cada clip en `Bnnnnn.npz`:

- **Anonimiza**: guarda solo los píxeles de la región de imagen, sin el encabezado
  DICOM ni el nombre del archivo original. La región sale de
  SequenceOfUltrasoundRegions (DICOM) o, si no está, del rectángulo que cambia en
  el tiempo, lo que deja fuera textos, menús y reglas fijas alrededor de la imagen.
- **Descarta Doppler color**: el color alteraría los grises al convertir a escala
  de grises. Un mapa de grises teñido (casi todo el campo con el mismo matiz) sí se
  acepta.
- **Omite duplicados** (misma huella de archivo) y archivos que no son clips.
- Guarda un tramo central de hasta `max_frames` frames consecutivos, reducido para
  que el lado mayor no pase de `lado_max` px. El tamaño del píxel se guarda si el
  DICOM lo informa; si no, queda NaN (sirve para preentrenar, no para cuantificar).

`inventario.csv` registra origen → salida y el motivo de cada omisión. Contiene las
rutas originales: queda solo en el computador local (la carpeta datos/ no se sube
al repositorio). Se puede reanudar: los archivos ya procesados no se repiten.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .io import Clip, _a_gris, guardar_npz, leer_dicom_crudo

EXT_VIDEO = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".m4v", ".mpg", ".mpeg", ".gif"}
EXT_DICOM = {".dcm", ".dicom"}
EXT_IMAGEN = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
COLUMNAS = ["id", "origen", "estado", "frames", "fps", "alto", "ancho", "mm_fila", "mm_columna",
            "recorte", "frac_color", "fabricante", "modelo", "huella"]


def es_dicom(ruta: Path) -> bool:
    if ruta.suffix.lower() in EXT_DICOM:
        return True
    if ruta.suffix:
        return False
    try:
        with open(ruta, "rb") as f:
            return f.read(132)[128:] == b"DICM"
    except OSError:
        return False


def huella(ruta: Path, bloque: int = 1 << 20) -> str:
    """Tamaño + SHA-1 del primer y el último MB: detecta copias sin leer archivos enteros."""
    n = ruta.stat().st_size
    h = hashlib.sha1(str(n).encode())
    with open(ruta, "rb") as f:
        h.update(f.read(bloque))
        if n > 2 * bloque:
            f.seek(-bloque, 2)
            h.update(f.read(bloque))
    return h.hexdigest()[:16]


def escribir_png(ruta: Path, img: np.ndarray) -> None:
    """cv2.imwrite no acepta rutas con tildes o ñ en Windows; se codifica en memoria."""
    Path(ruta).write_bytes(cv2.imencode(".png", img)[1].tobytes())


def leer_video_rgb(ruta: Path, max_frames: int = 64, lado_lectura: int = 1024,
                   centrar: bool = True) -> tuple[np.ndarray, float]:
    """Tramo central de hasta `max_frames` frames RGB, reducidos al leer si el lado mayor pasa de
    `lado_lectura` (una grabación de pantalla en 1080p no necesita más para recortar la imagen)."""
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        if not str(ruta).isascii():  # OpenCV en Windows puede no abrir rutas con tildes o ñ
            with tempfile.TemporaryDirectory() as d:
                copia = Path(d) / f"clip{ruta.suffix}"
                shutil.copyfile(ruta, copia)
                return leer_video_rgb(copia, max_frames, lado_lectura, centrar)
        raise ValueError("no se pudo abrir el video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for _ in range(tramo_central(total, max_frames).start if centrar else 0):  # saltar sin guardar
        if not cap.grab():
            break
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        f = lado_lectura / max(fr.shape[:2])
        if f < 1:
            fr = cv2.resize(fr, (round(fr.shape[1] * f), round(fr.shape[0] * f)), interpolation=cv2.INTER_AREA)
        frames.append(fr[..., ::-1] if fr.ndim == 3 else fr)
    cap.release()
    if not frames:
        if centrar and total > max_frames:  # el recuento de frames del contenedor era erróneo
            return leer_video_rgb(ruta, max_frames, lado_lectura, centrar=False)
        raise ValueError("el video no contiene frames")
    return np.stack(frames), float(fps)


def fraccion_color(rgb: np.ndarray, umbral: int = 40) -> tuple[float, float]:
    """(fracción de píxeles con color, fracción entre los píxeles con señal), en hasta 16 frames."""
    if rgb.ndim != 4:
        return 0.0, 0.0
    idx = np.linspace(0, len(rgb) - 1, min(len(rgb), 16)).astype(int)
    sub = rgb[idx].astype(np.int16)
    croma = sub.max(-1) - sub.min(-1)
    senal = sub.max(-1) > 20
    coloreado = croma > umbral
    return float(coloreado.mean()), float(coloreado.sum() / max(senal.sum(), 1))


def region_imagen(frames: np.ndarray, umbral_movimiento: float = 2.0) -> tuple[tuple[int, int, int, int], str]:
    """Rectángulo (y0, y1, x0, x1) de la imagen ecográfica: el mayor componente que cambia en el
    tiempo (o, en imágenes fijas, el mayor componente con señal)."""
    T, H, W = frames.shape
    sub = frames[np.linspace(0, T - 1, min(T, 32)).astype(int)].astype(np.float32)
    mapa, fuente = None, "contenido"
    if T >= 3:
        mapa = sub.std(0) > umbral_movimiento
        fuente = "movimiento"
        if mapa.mean() < 0.01:
            mapa = None
    if mapa is None:
        mapa, fuente = sub.max(0) > 10, "contenido"
    k = max(3, (min(H, W) // 40) | 1)
    cerrado = cv2.morphologyEx(mapa.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cerrado, connectivity=8)
    if n <= 1:
        return (0, H, 0, W), "completo"
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[i, :4])
    # El texto pegado a la imagen (que "parpadea" por la compresión del video) puede quedar unido
    # al componente: se recortan las columnas y filas del borde con poca señal respecto de la imagen.
    x0, x1 = _bordes_con_senal(mapa[y:y + h, x:x + w].mean(0))
    y0, y1 = _bordes_con_senal(mapa[y:y + h, x + x0:x + x1].mean(1))
    return (y + y0, y + y1, x + x0, x + x1), fuente


def _bordes_con_senal(perfil: np.ndarray, frac: float = 0.3) -> tuple[int, int]:
    ok = np.nonzero(perfil >= frac * np.median(perfil))[0]
    return (int(ok[0]), int(ok[-1]) + 1) if len(ok) else (0, len(perfil))


def tramo_central(T: int, max_frames: int) -> slice:
    if T <= max_frames:
        return slice(0, T)
    t0 = (T - max_frames) // 2
    return slice(t0, t0 + max_frames)


def reducir(frames: np.ndarray, lado_max: int) -> tuple[np.ndarray, float]:
    H, W = frames.shape[1:]
    f = min(1.0, lado_max / max(H, W))
    if f == 1.0:
        return frames, 1.0
    h, w = max(1, round(H * f)), max(1, round(W * f))
    return np.stack([cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA) for fr in frames]), f


def procesar(ruta: Path, max_frames: int, lado_max: int, umbral_color: float,
             tamano_min: int = 64) -> tuple[Clip | None, dict]:
    """Devuelve (clip o None, fila del inventario sin id/origen/huella)."""
    fila: dict = {}
    if es_dicom(ruta):
        arr, mm, region, fps, meta = leer_dicom_crudo(ruta)
        fila.update(fabricante=meta["fabricante"], modelo=meta["modelo"])
    else:
        arr, fps = leer_video_rgb(ruta, max_frames)
        mm, region = None, None
    arr = arr[tramo_central(len(arr), max_frames)]
    fila["fps"] = round(fps, 2)

    frames = _a_gris(arr)
    if region is not None:
        y0, y1, x0, x1 = region
        fila["recorte"] = "dicom_region"
    else:
        (y0, y1, x0, x1), fila["recorte"] = region_imagen(frames)
    frames = frames[:, y0:y1, x0:x1]

    # el color se mide solo dentro de la imagen: la interfaz del equipo suele tener textos e íconos en color
    frac, frac_senal = fraccion_color(arr[:, y0:y1, x0:x1])
    fila["frac_color"] = round(frac, 4)
    if frac > umbral_color and frac_senal < 0.6:
        return None, {**fila, "estado": "omitido: Doppler o superposición en color"}
    if min(frames.shape[1:]) < tamano_min:
        return None, {**fila, "estado": f"omitido: región de imagen muy pequeña ({frames.shape[1]}x{frames.shape[2]})"}

    frames, f = reducir(frames, lado_max)
    mm = (mm[0] / f, mm[1] / f) if mm else (float("nan"), float("nan"))
    fila.update(estado="ok", frames=frames.shape[0], alto=frames.shape[1], ancho=frames.shape[2],
                mm_fila=round(mm[0], 5), mm_columna=round(mm[1], 5))
    return Clip(frames=np.ascontiguousarray(frames), mm_por_pixel=mm, fps=fps), fila


def _leer_inventario(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _mosaicos(salida: Path, ids: list[str], por_hoja: int = 100, lado: int = 128) -> None:
    """Hojas de miniaturas (frame central) para revisar recortes y que no quede texto identificable."""
    carpeta = salida / "mosaicos"
    carpeta.mkdir(exist_ok=True)
    cols = 10
    for h in range(0, len(ids), por_hoja):
        grupo = ids[h:h + por_hoja]
        filas = (len(grupo) + cols - 1) // cols
        hoja = np.zeros((filas * (lado + 14), cols * lado), np.uint8)
        for k, i in enumerate(grupo):
            fr = np.load(salida / f"{i}.npz")["frames"]
            fr = fr[len(fr) // 2]
            f = lado / max(fr.shape)
            fr = cv2.resize(fr, (max(1, int(fr.shape[1] * f)), max(1, int(fr.shape[0] * f))))
            y, x = (k // cols) * (lado + 14), (k % cols) * lado
            hoja[y + 14:y + 14 + fr.shape[0], x:x + fr.shape[1]] = fr
            cv2.putText(hoja, i, (x + 2, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1, cv2.LINE_AA)
        escribir_png(carpeta / f"mosaico_{h // por_hoja + 1:03d}.png", hoja)


def ingerir(origen: str | Path, salida: str | Path, max_frames: int = 64, lado_max: int = 384,
            umbral_color: float = 0.003, mosaicos: bool = True, registro=print) -> dict:
    origen, salida = Path(origen), Path(salida)
    if not origen.is_dir():
        raise ValueError(f"No existe la carpeta {origen}")
    tmp = salida / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    ruta_inv = salida / "inventario.csv"
    # los errores se reintentan (p. ej. tras instalar un decodificador que faltaba)
    previas = [r for r in _leer_inventario(ruta_inv) if not r["estado"].startswith("error")]
    hechos = {r["origen"] for r in previas}
    huellas = {r["huella"]: r["id"] for r in previas if r["estado"] == "ok"}
    siguiente = 1 + max((int(r["id"][1:]) for r in previas if r["id"]), default=0)

    archivos = sorted(p for p in origen.rglob("*") if p.is_file())
    candidatos = [p for p in archivos if p.suffix.lower() in EXT_VIDEO or es_dicom(p)]
    n_imagenes = sum(p.suffix.lower() in EXT_IMAGEN for p in archivos)
    registro(f"{len(archivos)} archivos: {len(candidatos)} clips candidatos (DICOM/video), "
             f"{n_imagenes} imágenes fijas no incluidas, {len(archivos) - len(candidatos) - n_imagenes} otros")

    cuenta: dict[str, int] = {}
    with open(ruta_inv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(previas)
        for k, p in enumerate(candidatos, 1):
            rel = str(p.relative_to(origen))
            if rel in hechos:
                continue
            fila = {"origen": rel, "id": ""}
            try:
                fila["huella"] = hx = huella(p)
                if hx in huellas:
                    fila["estado"] = f"omitido: duplicado ({huellas[hx]})"
                else:
                    clip, extra = procesar(p, max_frames, lado_max, umbral_color)
                    fila.update(extra)
                    if clip is not None:
                        fila["id"] = f"B{siguiente:05d}"
                        siguiente += 1
                        # escritura atómica: una interrupción no deja un .npz a medias en la carpeta
                        guardar_npz(clip, tmp / f"{fila['id']}.npz")
                        (tmp / f"{fila['id']}.npz").replace(salida / f"{fila['id']}.npz")
                        huellas[hx] = fila["id"]
            except Exception as e:  # un archivo dañado no detiene la ingesta
                fila["estado"] = f"error ({type(e).__name__}: {str(e)[:150]})"
            w.writerow(fila)
            f.flush()
            clave = fila["estado"].split(" (")[0]
            cuenta[clave] = cuenta.get(clave, 0) + 1
            if k % 25 == 0:
                registro(f"  {k}/{len(candidatos)} …")

    shutil.rmtree(tmp, ignore_errors=True)
    ids = sorted(r["id"] for r in _leer_inventario(ruta_inv) if r["estado"] == "ok")
    if mosaicos and ids:
        _mosaicos(salida, ids)
    registro(f"Clips listos para preentrenar: {len(ids)}  (esta vez: {cuenta or 'nada nuevo'})")
    return {"ok": len(ids), "esta_vez": cuenta, "inventario": str(ruta_inv)}
