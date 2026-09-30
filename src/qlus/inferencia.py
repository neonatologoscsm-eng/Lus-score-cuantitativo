"""Segmentación de un clip completo con el modelo entrenado.

Las probabilidades se suavizan en el tiempo (ventana móvil) antes del argmax:
costillas, sombras e hígado no cambian entre frames consecutivos y así se evita
el parpadeo de la segmentación.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from .entrenamiento import pila_temporal
from .modelos import UNet


def cargar_modelo(ruta: str | Path) -> tuple[UNet, dict]:
    ck = torch.load(str(ruta), map_location="cpu", weights_only=False)
    modelo = UNet(**ck["config"])
    modelo.load_state_dict(ck["estado"])
    modelo.eval()
    return modelo, ck


@torch.no_grad()
def segmentar_clip(modelo: UNet, meta: dict, frames: np.ndarray, ventana_temporal: int = 3,
                   lote: int = 8) -> np.ndarray:
    T, H, W = frames.shape
    n = meta["tamano"]
    disp = next(modelo.parameters()).device
    probs = []
    for i in range(0, T, lote):
        x = np.stack([
            np.stack([cv2.resize(ch, (n, n), interpolation=cv2.INTER_LINEAR) for ch in pila_temporal(frames, t)])
            for t in range(i, min(i + lote, T))
        ]).astype(np.float32) / 255.0
        probs.append(modelo(torch.from_numpy(x).to(disp)).softmax(1).cpu().numpy().astype(np.float16))
    probs = np.concatenate(probs)                                   # (T, C, n, n)
    if ventana_temporal > 1 and T > 1:
        probs = uniform_filter1d(probs.astype(np.float32), size=ventana_temporal, axis=0, mode="nearest")
    salida = np.empty((T, H, W), np.uint8)
    for t in range(T):
        p = cv2.resize(np.ascontiguousarray(probs[t].transpose(1, 2, 0), dtype=np.float32), (W, H),
                       interpolation=cv2.INTER_LINEAR)
        salida[t] = p.argmax(-1).astype(np.uint8)
    return salida
