import numpy as np
import torch

from qlus.entrenamiento import cargar_pesos_iniciales, entrenar
from qlus.etiquetas import N_CLASES
from qlus.inferencia import cargar_modelo
from qlus.io import guardar_npz
from qlus.modelos import UNet
from qlus.preentrenamiento import TIPO, enmascarar, preentrenar
from qlus.sintetico import Escena, generar_clip


def test_enmascarar_modos():
    x = np.random.default_rng(0).random((3, 64, 64)).astype(np.float32)
    e, o, m = enmascarar(x, np.random.default_rng(1), temporal=False)
    assert np.array_equal(o, x[1]) and 0.3 < m.mean() < 0.7
    assert all((e[c][m] == e[1][m][0]).all() for c in range(3)) and np.array_equal(e[:, ~m], x[:, ~m])
    e, o, m = enmascarar(x, np.random.default_rng(1), temporal=True)
    assert np.array_equal(e[[0, 2]], x[[0, 2]]) and not np.array_equal(e[1], x[1])


def test_preentrenar_y_ajuste_fino(tmp_path):
    sin_anotar, anotados = tmp_path / "sin", tmp_path / "con"
    sin_anotar.mkdir(), anotados.mkdir()
    for i in range(4):
        c = generar_clip(Escena(n_frames=3, humedad=0.2 * i, alto=64, ancho=64, mm_por_pixel=0.6, semilla=i))
        guardar_npz(c, anotados / f"P{i // 2}__z{i}.npz")
        c.etiquetas = None
        c.mm_por_pixel = (float("nan"), float("nan"))
        guardar_npz(c, sin_anotar / f"B{i:05d}.npz")

    pre = tmp_path / "pre.pt"
    r = preentrenar(sin_anotar, pre, epocas=1, tamano=32, lote=2, base=4, niveles=3, iter_por_epoca=2,
                    clips_en_memoria=2, registro=lambda *_: None)
    assert np.isfinite(r["mejor_perdida_val"]) and r["perdida_val_sin_modelo"] > 0
    assert pre.with_suffix(".ejemplos.png").exists() and pre.with_suffix(".historial.json").exists()
    ck = torch.load(pre, weights_only=False)
    assert ck["tipo"] == TIPO and ck["config"]["n_clases"] == 1

    seg = UNet(3, N_CLASES, base=4, niveles=3)
    assert cargar_pesos_iniciales(seg, pre) == len(seg.state_dict()) - 2    # todo menos la capa final
    assert torch.equal(seg.state_dict()["bajada.0.0.weight"], ck["estado"]["bajada.0.0.weight"])

    # base=32 se ignora: la arquitectura la fija el modelo inicial
    fin = tmp_path / "seg.pt"
    entrenar(anotados, fin, epocas=1, tamano=32, lote=2, base=32, iter_por_epoca=2, inicial=pre,
             registro=lambda *_: None)
    modelo, meta = cargar_modelo(fin)
    assert meta["config"] == dict(canales_entrada=3, n_clases=N_CLASES, base=4, niveles=3)
