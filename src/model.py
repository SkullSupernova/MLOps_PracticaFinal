# =============================================================================
# model.py
# Definición de la arquitectura CNN_ResNet para clasificación de caracteres
# Kuzushiji (KMNIST).
#
# Contiene:
#   - ResidualBlock : bloque básico con skip connection.
#   - CNN_ResNet    : arquitectura residual adaptada a imágenes 28x28 monocanal.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Bloque residual fundamental de la arquitectura ResNet.

    Aplica dos capas convolucionales con normalización por lotes y conecta
    la entrada a la salida mediante una skip connection proyectada si las
    dimensiones no coinciden.

    Argumentos:
        in_channels  (int): Canales del tensor de entrada.
        out_channels (int): Canales del tensor de salida.
        stride       (int): Stride de la primera convolución. Un valor > 1
                            reduce la resolución espacial.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CNN_ResNet(nn.Module):
    """
    Red neuronal convolucional residual adaptada a imágenes de un canal y
    resolución 28x28 píxeles (formato KMNIST).

    Arquitectura:
        - Capa convolucional inicial: 1 → 16 canales.
        - Capa residual 1: 16 → 16 canales, stride 1.
        - Capa residual 2: 16 → 32 canales, stride 2 (reducción espacial).
        - Capa residual 3: 32 → 64 canales, stride 2 (reducción espacial).
        - Pooling global adaptativo → vector de 64 dimensiones.
        - Clasificador lineal: 64 → 10 clases.

    Entrada:  Tensor (B, 1, 28, 28).
    Salida:   Logits (B, 10).
    """

    def __init__(self):
        super().__init__()
        self._in_channels = 16

        self.conv_initial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(16, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, num_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, out_channels: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self._in_channels, out_channels, s))
            self._in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_initial(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
