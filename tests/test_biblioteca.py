import csv
import shutil

import cv2
import numpy as np

from qlus.biblioteca import ingerir, region_imagen
from qlus.io import cargar_npz
from qlus.sintetico import Escena, generar_clip


def _pantalla(frames, doppler=False):
    """Clip incrustado en una "pantalla" con texto fijo: nombre arriba y parámetros pegados a la derecha."""
    T, H, W = frames.shape
    out = np.zeros((T, H + 60, W + 90, 3), np.uint8)
    for t in range(T):
        fr = cv2.cvtColor(frames[t], cv2.COLOR_GRAY2BGR)
        if doppler:
            fr[20:50, 30:60] = (0, 0, 200) if t % 2 else (200, 30, 0)
        out[t, 40:40 + H, 20:20 + W] = fr
        cv2.putText(out[t], "PEREZ JUAN", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(out[t], "L12 30Hz", (W + 26, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    return out


def _video(ruta, frames):
    T, H, W, _ = frames.shape
    vw = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), 20, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()


def _dicom_sin_extension(ruta, frames):
    import pydicom  # noqa: F401
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    T, H, W = frames.shape
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.PatientName = "US", "PEREZ^JUAN"
    ds.Rows, ds.Columns, ds.NumberOfFrames = H, W, T
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    r = Dataset()
    r.RegionSpatialFormat, r.PhysicalUnitsXDirection, r.PhysicalUnitsYDirection = 1, 3, 3
    r.PhysicalDeltaX, r.PhysicalDeltaY = 0.03, 0.03
    r.RegionLocationMinX0, r.RegionLocationMinY0, r.RegionLocationMaxX1, r.RegionLocationMaxY1 = 10, 5, W - 11, H - 6
    ds.SequenceOfUltrasoundRegions = [r]
    ds.PixelData = frames.tobytes()
    ds.save_as(ruta, enforce_file_format=True)


def _clip(semilla, n_frames=12):
    return generar_clip(Escena(n_frames=n_frames, humedad=0.5, alto=96, ancho=96, mm_por_pixel=0.3,
                               semilla=semilla)).frames


def test_region_imagen_excluye_texto_fijo():
    gris = _pantalla(_clip(1))[..., 1]
    (y0, y1, x0, x1), fuente = region_imagen(gris)
    assert fuente == "movimiento"
    # las primeras filas (piel) del clip sintético son estáticas: pueden quedar fuera
    assert 0 <= y0 - 40 <= 4 and abs(y1 - 136) <= 2 and abs(x0 - 20) <= 2 and abs(x1 - 116) <= 3


def test_ingerir_biblioteca(tmp_path):
    bib, sal = tmp_path / "bib", tmp_path / "sal"
    (bib / "SDR" / "paciente X").mkdir(parents=True)
    _video(bib / "SDR" / "PEREZ_JUAN.avi", _pantalla(_clip(1, n_frames=90)))
    shutil.copy(bib / "SDR" / "PEREZ_JUAN.avi", bib / "copia.avi")
    _video(bib / "doppler.avi", _pantalla(_clip(2), doppler=True))
    _dicom_sin_extension(bib / "SDR" / "paciente X" / "IM0001", _clip(3))
    (bib / "notas.txt").write_text("no es un clip")

    r = ingerir(bib, sal, max_frames=64, lado_max=80, registro=lambda *_: None)
    assert r["ok"] == 2
    with open(sal / "inventario.csv", encoding="utf-8-sig") as f:
        filas = {row["origen"].replace("\\", "/"): row for row in csv.DictReader(f)}
    assert set(filas) == {"SDR/PEREZ_JUAN.avi", "copia.avi", "doppler.avi", "SDR/paciente X/IM0001"}
    assert filas["copia.avi"]["estado"] == "omitido: duplicado (B00001)"
    assert filas["doppler.avi"]["estado"].startswith("omitido: Doppler")

    video = cargar_npz(sal / "B00001.npz")
    assert video.frames.shape[0] == 64 and max(video.frames.shape[1:]) <= 80   # tramo central y reducido
    assert np.isnan(video.mm_por_pixel).all()                                   # video: tamaño desconocido
    dicom = cargar_npz(sal / "B00002.npz")
    assert filas["SDR/paciente X/IM0001"]["recorte"] == "dicom_region"
    assert dicom.frames.shape[1:] == (80, 71)                                   # región 86x76 → lado ≤ 80
    assert np.allclose(dicom.mm_por_pixel, 0.3 * 86 / 80, rtol=0.02)
    assert (sal / "mosaicos" / "mosaico_001.png").exists()

    r = ingerir(bib, sal, registro=lambda *_: None)                             # reanudar: nada nuevo
    assert r["ok"] == 2 and r["esta_vez"] == {}


def test_ingerir_dicom_jpeg(tmp_path):
    import io

    from PIL import Image
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.encaps import encapsulate
    from pydicom.uid import JPEGBaseline8Bit, generate_uid

    frames = _clip(4, n_frames=4)
    partes = []
    for f in frames:
        b = io.BytesIO()
        Image.fromarray(np.repeat(f[..., None], 3, -1)).save(b, format="JPEG", quality=90)
        partes.append(b.getvalue())
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = JPEGBaseline8Bit
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.Rows, ds.Columns, ds.NumberOfFrames = "US", 96, 96, 4
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "YBR_FULL_422", 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelSpacing = [0.3, 0.3]
    ds.PixelData = encapsulate(partes)
    ds["PixelData"].VR = "OB"
    (tmp_path / "bib").mkdir()
    ds.save_as(tmp_path / "bib" / "clip.dcm", enforce_file_format=True)

    r = ingerir(tmp_path / "bib", tmp_path / "sal", mosaicos=False, registro=lambda *_: None)
    assert r["ok"] == 1
    c = cargar_npz(tmp_path / "sal" / "B00001.npz")
    assert c.frames.shape[0] == 4 and np.allclose(c.mm_por_pixel, 0.3)
