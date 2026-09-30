import numpy as np

from qlus.etiquetas import Clase
from qlus.io import cargar_clip, guardar_npz
from qlus.sintetico import Escena, generar_clip


def test_formas_y_clases():
    c = generar_clip(Escena(n_frames=4, consolidaciones=[(19, 1.5, 3)], higado=True, semilla=1))
    assert c.frames.shape == c.etiquetas.shape == (4, 256, 256)
    presentes = set(np.unique(c.etiquetas))
    for clase in (Clase.PARED_TORACICA, Clase.COSTILLA, Clase.SOMBRA_COSTAL, Clase.LINEA_PLEURAL,
                  Clase.PULMON, Clase.HIGADO, Clase.VASO_ANECOICO, Clase.DIAFRAGMA):
        assert clase in presentes


def test_ganancia_es_aditiva_en_gris():
    a = generar_clip(Escena(n_frames=1, semilla=3, ganancia_db=0)).frames[0].astype(float)
    b = generar_clip(Escena(n_frames=1, semilla=3, ganancia_db=3)).frames[0].astype(float)
    m = (a > 20) & (a < 220)
    assert abs(np.median(b[m] - a[m]) - 255 * 3 / 60) < 1.0


def test_npz_ida_y_vuelta(tmp_path):
    c = generar_clip(Escena(n_frames=2, semilla=2))
    guardar_npz(c, tmp_path / "P1__x.npz")
    d = cargar_clip(tmp_path / "P1__x.npz")
    assert np.array_equal(c.frames, d.frames) and np.array_equal(c.etiquetas, d.etiquetas)
    assert d.mm_por_pixel == c.mm_por_pixel


def test_dicom_multiframe_con_regiones(tmp_path):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    frames = np.zeros((3, 80, 100), np.uint8)
    frames[:, 10:70, 20:90] = 120
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality = "US"
    ds.Rows, ds.Columns, ds.NumberOfFrames = 80, 100, 3
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.FrameTime = 50.0
    r = Dataset()
    r.RegionSpatialFormat, r.PhysicalUnitsXDirection, r.PhysicalUnitsYDirection = 1, 3, 3
    r.PhysicalDeltaX, r.PhysicalDeltaY = 0.015, 0.015
    r.RegionLocationMinX0, r.RegionLocationMinY0, r.RegionLocationMaxX1, r.RegionLocationMaxY1 = 20, 10, 89, 69
    ds.SequenceOfUltrasoundRegions = [r]
    ds.PixelData = frames.tobytes()
    ruta = tmp_path / "clip.dcm"
    ds.save_as(ruta, enforce_file_format=True)

    c = cargar_clip(ruta)
    assert c.frames.shape == (3, 60, 70)          # recortado a la región de imagen
    assert np.allclose(c.mm_por_pixel, (0.15, 0.15))
    assert abs(c.fps - 20.0) < 1e-6
    assert (c.frames == 120).all()
