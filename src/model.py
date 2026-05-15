# =========================================================
# 1. Librerías estándar de Python
# =========================================================
import sys
import os
import time
import random
import shutil
import copy
from collections import Counter
from pathlib import Path
from typing import Optional

# =========================================================
# 2. Manipulación de datos y matemáticas
# =========================================================
import numpy as np
import pandas as pd
import yaml

# =========================================================
# 3. Deep Learning: PyTorch y torchvision
# =========================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms

# =========================================================
# 4. Machine Learning clásico y métricas
# =========================================================
import sklearn
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# 5. Visualización
# =========================================================
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 6. Procesamiento de imágenes y tipografía
# =========================================================
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import matplotlib.font_manager as fm

# =========================================================
# 7. Entorno interactivo / notebooks
# =========================================================
import IPython
from IPython.display import display, Markdown


class ResidualBlock(nn.Module):
    """
    Define el bloque de construcción fundamental para la arquitectura ResNet,
    implementando una conexión de salto (skip connection) sobre dos capas convolucionales consecutivas.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Inicializa las capas y parámetros del bloque residual.

        Argumentos:
            in_channels (int): Número de canales o mapas de características del tensor de entrada.
            out_channels (int): Número de mapas de características generados en la salida.
            stride (int, opcional): Tamaño del salto (stride) aplicado en la primera convolución.
                Un valor > 1 efectúa una reducción de resolución espacial. Por defecto es 1.
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta la propagación hacia adelante (forward pass) del bloque residual aplicando la ecuación F(x) + x.

        Argumentos:
            x (torch.Tensor): Tensor de entrada al bloque de procesamiento.

        Retorna:
            torch.Tensor: Tensor resultante tras la suma residual y la activación final (ReLU).
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNN_ResNet(nn.Module):
    """
    Arquitectura de Red Neuronal Convolucional Residual (ResNet) simplificada y adaptada
    específicamente para procesar conjuntos de datos con imágenes de un solo canal y
    baja resolución espacial (28x28 píxeles).
    """
    def __init__(self):
        """
        Inicializa las etapas convolucionales y el clasificador lineal de la topología ResNet.
        El modelo consta de una convolución base, tres etapas residuales de extracción de características
        y una proyección lineal configurada explícitamente para 10 clases de salida.
        """
        super(CNN_ResNet, self).__init__()
        
        self.in_channels = 16
        self.conv_initial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Construye secuencialmente una etapa completa compuesta por iteraciones de bloques residuales.

        Argumentos:
            out_channels (int): Filtros de salida exigidos para todos los bloques de la etapa.
            num_blocks (int): Cantidad de bloques residuales independientes a concatenar.
            stride (int): Salto de la convolución aplicado de forma exclusiva en el primer bloque de la secuencia.

        Retorna:
            torch.nn.Sequential: Módulo contenedor secuencial ejecutable.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define y ejecuta el flujo de inferencia de la red desde el procesamiento base hasta la capa final.

        Argumentos:
            x (torch.Tensor): Tensor de entrada asumiendo formato y dimensiones estandarizadas (Batch, 1, 28, 28).

        Retorna:
            torch.Tensor: Tensor de logits no normalizados equivalentes a las clases de decisión (Batch, 10).
        """
        out = self.conv_initial(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return logits

