import json

import numpy as np
import torch

from qlus.entrenamiento import entrenar, particionar
from qlus.etiquetas import N_CLASES
from qlus.inferencia import cargar_modelo, segmentar_clip
from qlus.io import guardar_npz
from qlus.modelos import UNet
from qlus.nnunet import exportar
from qlus.sintetico import Escena, generar_clip


def _datos(carpeta, n=4):
    for i in range(n):
        c = generar_clip(Escena(n_frames=3, humedad=0.1 * i, alto=64, ancho=64, mm_por_pixel=0.6, semilla=i))
        guardar_npz(c, carpeta / f"P{i // 2}__z{i}.npz")


def test_unet_forma():
    y = UNet(3, N_CLASES, base=4, niveles=3)(torch.zeros(2, 3, 64, 64))
    assert y.shape == (2, N_CLASES, 64, 64)


def test_particion_por_paciente(tmp_path):
    _datos(tmp_path, n=6)
    tr, va = particionar(sorted(tmp_path.glob("*.npz")), 0.34, 0)
    pac = lambda rs: {r.stem.split("__")[0] for r in rs}
    assert tr and va and not (pac(tr) & pac(va))


def test_entrenar_e_inferir(tmp_path):
    _datos(tmp_path)
    ruta = tmp_path / "m.pt"
    r = entrenar(tmp_path, ruta, epocas=1, tamano=32, lote=2, base=4, iter_por_epoca=2, registro=lambda *_: None)
    assert ruta.exists() and 0 <= r["mejor_dice_medio"] <= 1
    modelo, meta = cargar_modelo(ruta)
    frames = np.random.default_rng(0).integers(0, 255, (5, 50, 70), dtype=np.uint8)
    lab = segmentar_clip(modelo, meta, frames)
    assert lab.shape == frames.shape and lab.max() < N_CLASES


def test_exportar_nnunet(tmp_path):
    datos, salida = tmp_path / "d", tmp_path / "o"
    datos.mkdir()
    _datos(datos)
    raiz = exportar(datos, salida, n_folds=2)
    ds = json.loads((raiz / "dataset.json").read_text())
    assert ds["labels"]["background"] == 0 and ds["numTraining"] == 12
    assert len(list((raiz / "imagesTr").glob("*_0002.png"))) == 12
    for fold in json.loads((raiz / "splits_final.json").read_text()):
        pac = lambda casos: {c.split("__")[0] for c in casos}
        assert not (pac(fold["train"]) & pac(fold["val"]))


def test_importar_mascaras_dispersas_y_entrenar(tmp_path):
    import cv2

    from qlus.anotacion import importar_mascaras
    from qlus.etiquetas import COLORES, SIN_ANOTAR

    c = generar_clip(Escena(n_frames=6, alto=64, ancho=64, mm_por_pixel=0.6, semilla=3))
    carpeta = tmp_path / "mascaras"
    carpeta.mkdir()
    for t in (1, 4):   # solo 2 de 6 frames anotados; uno en color, otro en índices
        if t == 1:
            rgb = np.zeros(c.etiquetas[t].shape + (3,), np.uint8)
            for k, col in COLORES.items():
                rgb[c.etiquetas[t] == k] = col
            cv2.imwrite(str(carpeta / f"frame_{t:06d}.png"), rgb[..., ::-1])
        else:
            cv2.imwrite(str(carpeta / f"frame_{t:06d}.png"), c.etiquetas[t])
    c.etiquetas = None
    d = importar_mascaras(c, carpeta)
    assert d.meta["frames_anotados"] == 2
    assert (d.etiquetas[0] == SIN_ANOTAR).all() and (d.etiquetas[1] != SIN_ANOTAR).all()
    datos = tmp_path / "d"
    datos.mkdir()
    guardar_npz(d, datos / "P9__z.npz")
    guardar_npz(d, datos / "P8__z.npz")
    r = entrenar(datos, tmp_path / "m.pt", epocas=1, tamano=32, lote=2, base=4, iter_por_epoca=2,
                 registro=lambda *_: None)
    assert 0 <= r["mejor_dice_medio"] <= 1
    raiz = exportar(datos, tmp_path / "nn")
    assert json.loads((raiz / "dataset.json").read_text())["numTraining"] == 4
