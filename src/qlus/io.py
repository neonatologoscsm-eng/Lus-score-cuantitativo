"""Lectura de clips: DICOM de ultrasonido, video (mp4/avi) y el formato .npz propio.

Todo clip se entrega como `Clip`: frames en escala de grises (T, H, W) uint8 y el
tamaño físico del píxel en mm (fila, columna). Sin el tamaño del píxel no se
puede definir la banda de profundidad fija bajo la pleura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Clip:
    frames: np.ndarray                      # (T, H, W) uint8
    mm_por_pixel: tuple[float, float]       # (fila = profundidad, columna = lateral)
    fps: float = 20.0
    etiquetas: np.ndarray | None = None     # (T, H, W) uint8, opcional (anotación o sintético)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.frames.ndim == 2:
            self.frames = self.frames[None]
        if self.frames.ndim != 3:
            raise ValueError(f"frames debe ser (T, H, W); recibido {self.frames.shape}")
        if self.etiquetas is not None and self.etiquetas.shape != self.frames.shape:
            raise ValueError("etiquetas y frames deben tener la misma forma")


def _a_gris(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 or (arr.ndim == 3 and arr.shape[-1] == 3):
        arr = arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1e-6) * 255.0
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def cargar_dicom(ruta: str | Path) -> Clip:
    import pydicom

    ds = pydicom.dcmread(str(ruta))
    try:
        from pydicom.pixels import pixel_array  # pydicom >= 3: convierte YBR a RGB

        arr = pixel_array(ds)
    except ImportError:
        arr = ds.pixel_array
        if str(ds.PhotometricInterpretation).startswith("YBR"):
            from pydicom.pixel_data_handlers.util import convert_color_space

            arr = convert_color_space(arr, ds.PhotometricInterpretation, "RGB")

    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if n_frames == 1:
        arr = arr[None]
    frames = _a_gris(arr)

    mm = None
    region = None
    for r in getattr(ds, "SequenceOfUltrasoundRegions", []) or []:
        # RegionSpatialFormat 1 = imagen 2D; unidades 3 = cm
        if int(getattr(r, "RegionSpatialFormat", 0)) == 1 and int(getattr(r, "PhysicalUnitsXDirection", 0)) == 3:
            mm = (float(r.PhysicalDeltaY) * 10.0, float(r.PhysicalDeltaX) * 10.0)
            region = (int(r.RegionLocationMinY0), int(r.RegionLocationMaxY1) + 1,
                      int(r.RegionLocationMinX0), int(r.RegionLocationMaxX1) + 1)
            break
    if mm is None and hasattr(ds, "PixelSpacing"):
        mm = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
    if mm is None:
        raise ValueError(f"{ruta}: el DICOM no informa el tamaño del píxel (SequenceOfUltrasoundRegions/PixelSpacing)")
    if region is not None:
        y0, y1, x0, x1 = region
        frames = frames[:, y0:y1, x0:x1]

    fps = 20.0
    if getattr(ds, "CineRate", None):
        fps = float(ds.CineRate)
    elif getattr(ds, "FrameTime", None):
        fps = 1000.0 / float(ds.FrameTime)

    meta = {"fuente": str(ruta), "fabricante": str(getattr(ds, "Manufacturer", "")),
            "modelo": str(getattr(ds, "ManufacturerModelName", ""))}
    return Clip(frames=np.ascontiguousarray(frames), mm_por_pixel=mm, fps=fps, meta=meta)


def cargar_video(ruta: str | Path, mm_por_pixel: tuple[float, float]) -> Clip:
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir {ruta}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise ValueError(f"{ruta} no contiene frames")
    return Clip(frames=np.stack(frames), mm_por_pixel=tuple(mm_por_pixel), fps=float(fps),
                meta={"fuente": str(ruta)})


def cargar_npz(ruta: str | Path) -> Clip:
    d = np.load(str(ruta), allow_pickle=False)
    etiquetas = d["etiquetas"] if "etiquetas" in d.files else None
    fps = float(d["fps"]) if "fps" in d.files else 20.0
    return Clip(frames=d["frames"], mm_por_pixel=tuple(float(v) for v in d["mm_por_pixel"]),
                fps=fps, etiquetas=etiquetas, meta={"fuente": str(ruta)})


def guardar_npz(clip: Clip, ruta: str | Path) -> None:
    datos = {"frames": clip.frames, "mm_por_pixel": np.asarray(clip.mm_por_pixel, np.float64),
             "fps": np.float64(clip.fps)}
    if clip.etiquetas is not None:
        datos["etiquetas"] = clip.etiquetas
    np.savez_compressed(str(ruta), **datos)


def cargar_clip(ruta: str | Path, mm_por_pixel: tuple[float, float] | None = None) -> Clip:
    ruta = Path(ruta)
    ext = ruta.suffix.lower()
    if ext == ".npz":
        return cargar_npz(ruta)
    if ext in (".dcm", ".dicom", ""):
        return cargar_dicom(ruta)
    if mm_por_pixel is None:
        raise ValueError(f"{ruta}: para video se requiere mm_por_pixel")
    return cargar_video(ruta, mm_por_pixel)
