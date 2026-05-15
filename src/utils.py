# =============================================================================
# utils.py
# Utilidades del proyecto KMNIST: carga de datos, normalización, DataLoaders,
# métricas y visualización.
#
# No contiene definiciones de arquitecturas de modelo (véase model.py).
# No depende de IPython ni de entornos interactivos.
# =============================================================================

import copy
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.Image as Image
import matplotlib.font_manager as fm
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Subset, random_split

from src.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Utilidades de entorno
# =============================================================================

def get_project_root() -> Path:
    """
    Retorna la ruta absoluta del directorio raíz del proyecto.

    Asume que utils.py se encuentra en <raíz>/src/utils.py.
    """
    return Path(__file__).parent.parent.resolve()


def load_config(nombre: str) -> dict:
    """
    Lee y devuelve la configuración YAML ubicada en <raíz>/config/<nombre>.

    Argumentos:
        nombre (str): Nombre del archivo YAML, incluyendo extensión.

    Retorna:
        dict: Parámetros de configuración parseados.

    Excepciones:
        FileNotFoundError: Si el archivo no existe.
        yaml.YAMLError: Si el archivo tiene sintaxis YAML inválida.
    """
    config_path = get_project_root() / 'config' / nombre
    with open(config_path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def setup_environment() -> tuple:
    """
    Detecta el hardware disponible y determina num_workers óptimo.

    Retorna:
        tuple: (device: torch.device, num_workers: int)
    """
    cpu_cores = os.cpu_count()
    logger.info("Núcleos lógicos CPU: %s", cpu_cores)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info("Dispositivo: GPU (%s) | VRAM: %.2f GB", gpu_name, vram_gb)
    else:
        device = torch.device("cpu")
        logger.info("Dispositivo: CPU")

    num_workers = 0 if os.name == 'nt' else 2
    logger.info("num_workers configurado en %d (OS: %s)", num_workers, os.name)

    return device, num_workers


def set_seed(seed: int = 42) -> torch.Generator:
    """
    Fija las semillas globales en Python, NumPy, PyTorch (CPU y GPU) y
    habilita el modo determinista de cuDNN.

    Argumentos:
        seed (int): Semilla global. Por defecto: 42.

    Retorna:
        torch.Generator: Generador CPU inicializado con la semilla indicada.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)
    logger.info("Semilla global fijada en %d.", seed)
    return g


# =============================================================================
# Carga y validación de datos
# =============================================================================

def cargar_y_validar_kmnist(
    data_path: str,
    is_train: bool = True,
    download: bool = False,
) -> torchvision.datasets.KMNIST:
    """
    Instancia y valida estructuralmente el dataset KMNIST.

    Argumentos:
        data_path (str): Directorio raíz donde residen los datos.
        is_train  (bool): True para la partición de entrenamiento.
        download  (bool): Descargar si no existen los archivos.

    Retorna:
        torchvision.datasets.KMNIST: Instancia validada.

    Excepciones:
        RuntimeError: Error interno de PyTorch durante la carga.
        ValueError:   Las dimensiones del tensor no coinciden con (1, 28, 28).
    """
    raw_folder = Path(data_path) / 'KMNIST' / 'raw'
    logger.info("Verificando directorio: %s", raw_folder.resolve())

    expected = {
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz",
    }
    if raw_folder.exists():
        missing = expected - {f.name for f in raw_folder.iterdir() if f.is_file()}
        if missing:
            logger.warning("Archivos faltantes: %s", missing)
    else:
        logger.warning("Directorio raw no encontrado: %s", raw_folder)

    dataset = torchvision.datasets.KMNIST(
        root=data_path, train=is_train,
        download=download, transform=transforms.ToTensor(),
    )
    img_sample, _ = dataset[0]
    expected_shape = (1, 28, 28)
    if img_sample.shape != expected_shape:
        raise ValueError(
            f"Discrepancia dimensional: esperado {expected_shape}, "
            f"obtenido {img_sample.shape}."
        )
    logger.info(
        "KMNIST cargado. Partición: %s | Muestras: %d | "
        "Rango de valores: [%.4f, %.4f]",
        "train" if is_train else "test",
        len(dataset), img_sample.min().item(), img_sample.max().item(),
    )
    return dataset


def calculate_dataset_statistics(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 60000,
) -> tuple:
    """
    Calcula la media y la desviación estándar global del dataset.

    Argumentos:
        dataset    (Dataset): Instancia del dataset.
        batch_size (int):     Muestras cargadas simultáneamente.

    Retorna:
        tuple: (mean_val: float, std_val: float)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_stack, _ = next(iter(loader))
    return data_stack.mean().item(), data_stack.std().item()


# =============================================================================
# Transformaciones y DataLoaders
# =============================================================================

def get_transforms(
    mean_val: float,
    std_val: float,
    is_train: bool = True,
) -> transforms.Compose:
    """
    Construye el pipeline de transformaciones de imagen.

    En entrenamiento aplica RandomRotation y RandomAffine como aumento.
    En evaluación solo normaliza.
    """
    ops = []
    if is_train:
        ops += [
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((mean_val,), (std_val,)),
    ]
    return transforms.Compose(ops)


def prepare_dataloaders(
    ruta_base: str,
    mean_val: float,
    std_val: float,
    batch_size_train: int,
    batch_size_eval: int,
    num_workers: int,
    generator: torch.Generator,
) -> tuple:
    """
    Construye DataLoaders para entrenamiento, validación y prueba.

    La división train/val (80/20) se realiza a nivel de índices para
    garantizar la separación estricta de transformaciones sin contaminación.

    Retorna:
        tuple: (train_loader, val_loader, test_loader,
                train_size, val_size, test_size)
    """
    transform_train = get_transforms(mean_val, std_val, is_train=True)
    transform_eval  = get_transforms(mean_val, std_val, is_train=False)

    full_aug  = torchvision.datasets.KMNIST(
        root=ruta_base, train=True, download=False, transform=transform_train
    )
    full_eval = torchvision.datasets.KMNIST(
        root=ruta_base, train=True, download=False, transform=transform_eval
    )
    test_ds = torchvision.datasets.KMNIST(
        root=ruta_base, train=False, download=False, transform=transform_eval
    )

    n = len(full_aug)
    train_size = int(0.8 * n)
    val_size   = n - train_size

    # Índices shuffled una sola vez — sin consumir el generador dos veces
    perm = torch.randperm(n, generator=generator).tolist()
    train_idx = perm[:train_size]
    val_idx   = perm[train_size:]

    train_ds = Subset(full_aug,  train_idx)   # con augmentación
    val_ds   = Subset(full_eval, val_idx)     # sin augmentación

    common_dl_kwargs = dict(num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size_train,
        shuffle=True, generator=generator, **common_dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size_eval,
        shuffle=False, **common_dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size_eval,
        shuffle=False, **common_dl_kwargs,
    )

    logger.info(
        "DataLoaders construidos — Train: %d | Val: %d | Test: %d",
        train_size, val_size, len(test_ds),
    )
    return train_loader, val_loader, test_loader, train_size, val_size, len(test_ds)


# =============================================================================
# Callbacks de entrenamiento
# =============================================================================

class EarlyStopping:
    """Detiene el entrenamiento si la pérdida de validación no mejora."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ModelCheckpoint:
    """Conserva en memoria el estado del modelo con menor pérdida de validación."""

    def __init__(self):
        self.best_val_loss   = float('inf')
        self.best_model_state = None

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss    = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            return True
        return False


# =============================================================================
# Entrenamiento y evaluación
# =============================================================================

def calculate_metrics(y_true, y_pred) -> dict:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
    }

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int,
    device: torch.device,
    scheduler=None,
) -> tuple:
    """
    Bucle principal de entrenamiento con Early Stopping y Model Checkpointing.

    Retorna:
        tuple: (history: dict, model: nn.Module con pesos óptimos restaurados)
    """
    early_stopping    = EarlyStopping(patience=5)
    model_checkpoint  = ModelCheckpoint()
    history           = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    model     = model.to(device)
    start     = time.time()
    logger.info("Entrenamiento iniciado en: %s", device)

    for epoch in range(num_epochs):
        # --- Fase de entrenamiento ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        # --- Fase de validación ---
        model.eval()
        val_loss_accum = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss_accum += criterion(outputs, labels).item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_loss_accum / len(val_loader.dataset)
        metrics        = calculate_metrics(all_labels, all_preds)

        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['f1_macro'])

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_epoch_loss)

        saved    = model_checkpoint(model, val_epoch_loss)
        save_tag = " [*]" if saved else ""

        logger.info(
            "Época %03d/%d | LR: %.2e | Loss: %.4f | "
            "Val Loss: %.4f | Val Acc: %.4f | F1: %.4f%s",
            epoch + 1, num_epochs, current_lr,
            epoch_loss, val_epoch_loss,
            metrics['accuracy'], metrics['f1_macro'], save_tag,
        )

        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            logger.info("Early Stopping activado en época %d.", epoch + 1)
            break

    elapsed = time.time() - start
    logger.info("Entrenamiento finalizado en %.2f s.", elapsed)

    if model_checkpoint.best_model_state:
        model.load_state_dict(model_checkpoint.best_model_state)
        logger.info("Pesos restaurados al mínimo de pérdida de validación.")

    return history, model

# def load_checkpoint(
#     save_path: str,
#     model: nn.Module,
#     device: torch.device,
# ) -> tuple:
#     """
#     Carga el estado del modelo desde un archivo .pth.

#     Usa weights_only=True para evitar la deserialización de objetos Python
#     arbitrarios (mitigación de pickle injection).

#     Retorna:
#         tuple: (success: bool, state_dict: dict | None, metrics: dict)
#     """
#     try:
#         ckpt = torch.load(save_path, map_location=device, weights_only=True)
#         model.load_state_dict(ckpt['model_state_dict'])
#         metrics = ckpt.get('metrics', {})
#         logger.info("Checkpoint cargado desde: %s | Métricas: %s", save_path, metrics)
#         return True, ckpt['model_state_dict'], metrics
#     except Exception as exc:
#         logger.error("Error al cargar checkpoint '%s': %s", save_path, exc)
#         return False, None, {}

def load_checkpoint(
    save_path: str,
    model: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Carga el estado del modelo desde un archivo .pth.

    Usa weights_only=True junto con autorización explícita de NumPy scalar.
    """
    try:
        # Autorización de la variable global restringida por PyTorch 2.6+
        import numpy as np
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        
        ckpt = torch.load(save_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        metrics = ckpt.get('metrics', {})
        logger.info("Checkpoint cargado desde: %s | Métricas: %s", save_path, metrics)
        return True, ckpt['model_state_dict'], metrics
    except Exception as exc:
        logger.error("Error al cargar checkpoint '%s': %s", save_path, exc)
        return False, None, {}

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Evalúa el modelo sobre el conjunto de prueba.

    Retorna:
        tuple: (accuracy: float, f1_macro: float)
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, predicted   = torch.max(model(inputs).data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = float(np.sum(all_preds == all_labels) / len(all_labels))
    f1  = float(f1_score(all_labels, all_preds, average='macro'))
    return acc, f1


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Extrae predicciones, probabilidades Softmax e imágenes de un DataLoader.

    Retorna:
        tuple: (all_labels, all_preds, all_probs, all_images)
    """
    model.eval()
    all_preds, all_labels, all_probs, all_images = [], [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs        = model(inputs)
            probs          = F.softmax(outputs, dim=1)
            _, preds       = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(inputs.cpu())

    return (
        np.array(all_labels), np.array(all_preds),
        np.array(all_probs), all_images,
    )


# =============================================================================
# Visualización
# =============================================================================

def plot_training_history(history: dict) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Pérdida de entrenamiento', linewidth=2)
    plt.plot(history['val_loss'],   label='Pérdida de validación',    linewidth=2, linestyle='--')
    plt.title('Dinámica de convergencia')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (Cross Entropy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
) -> None:
    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matriz de confusión (Exactitud: {acc:.2%})")
    plt.ylabel("Etiqueta real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.show()


def plot_error_gallery(
    images: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    mean_val: float,
    std_val: float,
    num_images: int = 10,
) -> None:
    """
    Muestra imágenes clasificadas incorrectamente.

    Los parámetros mean_val y std_val se usan para la desnormalización,
    evitando el uso de constantes hardcodeadas.
    """
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        logger.info("No se registraron errores de clasificación.")
        return

    indices = np.random.choice(wrong, size=min(num_images, len(wrong)), replace=False)
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        img_disp = images[idx].squeeze().numpy() * std_val + mean_val
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_disp, cmap='gray')
        plt.title(
            f"Real: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}",
            color='red', fontsize=11, fontweight='bold',
        )
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Utilidades OOD
# =============================================================================

def get_japanese_font(size: int = 22):
    """Localiza una fuente CJK del sistema o retorna la fuente por defecto."""
    candidates = ['NotoSansCJK-Regular.ttc', 'ipag.ttf',
                  'msgothic.ttc', 'Arial Unicode.ttf']
    font_path = None
    for font in fm.findSystemFonts():
        for cand in candidates:
            if cand.lower() in font.lower():
                font_path = font
                break
        if font_path:
            break
    try:
        return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    except IOError:
        return ImageFont.load_default()


def create_image_from_text(
    char: str,
    mean_val: float,
    std_val: float,
    font_size: int = 22,
) -> torch.Tensor:
    """
    Genera un tensor 28x28 normalizado a partir de un carácter de texto.

    Retorna:
        torch.Tensor: Tensor (1, 1, 28, 28).
    """
    img  = Image.new('L', (28, 28), color=0)
    draw = ImageDraw.Draw(img)
    font = get_japanese_font(size=font_size)
    try:
        left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
        w, h = right - left, bottom - top
    except AttributeError:
        w, h = draw.textsize(char, font=font)   # compatibilidad con Pillow < 9.2

    draw.text(((28 - w) / 2, (28 - h) / 2), char, font=font, fill=255)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean_val,), (std_val,)),
    ])
    return tf(img).unsqueeze(0)


def visualize_ood_confidence(
    model: nn.Module,
    input_tensor: torch.Tensor,
    input_label: str,
    class_names: list,
    device: torch.device,
    mean_val: float,
    std_val: float,
) -> None:
    """
    Visualiza la distribución de confianza Softmax ante una entrada OOD.

    Los parámetros mean_val y std_val se usan para desnormalizar
    la imagen de visualización.
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs  = model(input_tensor)
        probs    = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(torch.argmax(outputs, 1).item())

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 2, 1)
    img_disp = input_tensor.cpu().squeeze().numpy() * std_val + mean_val
    plt.imshow(img_disp, cmap='gray')
    plt.title(f"Entrada OOD: '{input_label}'")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    colors = ['#e0e0e0'] * 10
    colors[pred_idx] = '#3498db'
    plt.bar(class_names, probs, color=colors)
    plt.title(
        f"Predicción: {class_names[pred_idx]} "
        f"({probs[pred_idx] * 100:.1f}%)"
    )
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()