"""U-Net 2D liviana para la segmentación por píxel.

Entrada: (N, canales_entrada, H, W) con H y W divisibles por 2**(niveles-1);
los canales son los frames t−1, t y t+1. Salida: logits (N, n_clases, H, W).
Con base=32 y niveles=4 tiene ~1,9 M parámetros (entrena en CPU).

`config` guarda los argumentos del constructor para reconstruir el modelo desde
el checkpoint (`UNet(**ck["config"])`).
"""

from __future__ import annotations

import torch
from torch import nn


def _bloque(entrada: int, salida: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(entrada, salida, 3, padding=1, bias=False), nn.BatchNorm2d(salida), nn.ReLU(inplace=True),
        nn.Conv2d(salida, salida, 3, padding=1, bias=False), nn.BatchNorm2d(salida), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, canales_entrada: int = 3, n_clases: int = 14, base: int = 32, niveles: int = 4):
        super().__init__()
        self.config = dict(canales_entrada=canales_entrada, n_clases=n_clases, base=base, niveles=niveles)
        anchos = [base * 2 ** i for i in range(niveles)]
        self.bajada = nn.ModuleList()
        c = canales_entrada
        for a in anchos:
            self.bajada.append(_bloque(c, a))
            c = a
        self.subida = nn.ModuleList()
        self.fusion = nn.ModuleList()
        for a in reversed(anchos[:-1]):
            self.subida.append(nn.ConvTranspose2d(c, a, 2, stride=2))
            self.fusion.append(_bloque(2 * a, a))
            c = a
        self.cabeza = nn.Conv2d(c, n_clases, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        saltos = []
        for i, bloque in enumerate(self.bajada):
            if i:
                x = nn.functional.max_pool2d(x, 2)
            x = bloque(x)
            saltos.append(x)
        for sube, fusiona, salto in zip(self.subida, self.fusion, reversed(saltos[:-1])):
            x = fusiona(torch.cat([sube(x), salto], 1))
        return self.cabeza(x)
