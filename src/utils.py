# =========================================================
# 1. Librerías estándar de Python
# =========================================================
import os
import time
import random
import copy
from collections import Counter
from pathlib import Path

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms

# =========================================================
# 4. Machine Learning clásico y métricas
# =========================================================
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix
)

# =========================================================
# 5. Visualización
# =========================================================
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 6. Procesamiento de imágenes y tipografía
# =========================================================
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import matplotlib.font_manager as fm

def get_project_root() -> Path:
    """
    Obtiene la ruta absoluta del directorio raíz del proyecto mediante la resolución relativa
    desde la ubicación del script actual.

    Retorna:
        Path: Objeto que representa la ruta absoluta al directorio raíz del proyecto.

    Notas:
        La profundidad de jerarquía ascendente (parents[3]) asume una estructura de carpetas
        estricta, cuya compatibilidad y correcto funcionamiento en entornos de producción 
        arbitrarios se considera [NO VERIFICABLE].
    """
    # return Path(__file__).parent.parent.resolve().parents[3]
    return Path(__file__).parent.parent.resolve()

def load_data(path: str) -> pd.DataFrame:
    """
    Carga un conjunto de datos en formato CSV desde una ruta relativa al directorio raíz del proyecto.

    Argumentos:
        path (str): Ruta relativa del archivo de datos respecto a la raíz del proyecto.

    Retorna:
        pd.DataFrame: Estructura de datos tabular con el contenido del archivo cargado.

    Excepciones:
        FileNotFoundError: Si la ruta combinada no corresponde a un archivo accesible en disco.
    """
    raiz_proyecto = get_project_root()
    ruta_completa = raiz_proyecto / path
    return pd.read_csv(ruta_completa)

def load_config(nombre: str) -> dict:
    """
    Lee y procesa un archivo de configuración en formato YAML ubicado en el directorio de configuración del proyecto.

    Argumentos:
        nombre (str): Nombre del archivo YAML a procesar, incluyendo su extensión.

    Retorna:
        dict: Diccionario estructurado con los parámetros de configuración extraídos.

    Excepciones:
        FileNotFoundError: Si el archivo especificado no existe en el directorio de configuración.
        yaml.YAMLError: Si el archivo presenta errores de sintaxis o formato YAML inválido.
    """
    raiz_proyecto = get_project_root()
    ruta_config = raiz_proyecto / 'config' / nombre
    with open(ruta_config, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_environment():
    """
    Configura el dispositivo de aceleración de hardware y los parámetros de concurrencia para el procesamiento de datos.

    Evalúa la disponibilidad de GPU compatibles con CUDA y determina el número óptimo de subprocesos 
    para la carga de datos en función del sistema operativo subyacente, priorizando la estabilidad 
    frente a la paralelización en entornos Windows.

    Retorna:
        tuple: Tupla que contiene:
            - device (torch.device): Dispositivo asignado para el cómputo ('cuda' o 'cpu').
            - num_workers (int): Número de procesos paralelos recomendados para objetos DataLoader.
    """
    print("\n=== Detección de Hardware ===")
    
    cpu_cores = os.cpu_count()
    if cpu_cores is not None:
        print(f"Núcleos lógicos CPU  : {cpu_cores}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Dispositivo          : GPU ({gpu_name})")
        print(f"VRAM Disponible      : {vram_gb:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Dispositivo          : CPU")

    if os.name == 'nt':
        num_workers = 0
        print("Configuración OS     : Windows (num_workers=0)")
    else:
        num_workers = 2
        print("Configuración OS     : Unix/Linux (num_workers=2)")

    return device, num_workers

def set_seed(seed=42):
    """
    Establece la semilla de inicialización para múltiples generadores de números pseudoaleatorios,
    garantizando la reproducibilidad metodológica de los experimentos.

    Fija las semillas en las librerías integradas de Python, NumPy y PyTorch, además de forzar 
    comportamiento determinista en los algoritmos de convolución de cuDNN, desactivando métricas heurísticas.

    Argumentos:
        seed (int, opcional): Valor numérico utilizado como semilla global. Su valor predeterminado es 42.

    Retorna:
        torch.Generator: Objeto generador de PyTorch instanciado y configurado con la semilla especificada,
        empleado para inyectar determinismo en operaciones estocásticas como el particionado de conjuntos de datos.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    
    print(f"\nSemilla global y generador fijados en: {seed}")
    
    return g_cpu

def calculate_metrics(y_true, y_pred):
    """
    Calcula las métricas de evaluación de clasificación multiclase: exactitud (accuracy) y F1-Score (macro).

    Argumentos:
        y_true (array-like): Secuencia de etiquetas reales (ground truth).
        y_pred (array-like): Secuencia de etiquetas predichas por el modelo.

    Retorna:
        dict: Diccionario que contiene las métricas calculadas bajo las claves 'accuracy' y 'f1_macro'.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return {'accuracy': acc, 'f1_macro': f1}

def cargar_y_validar_kmnist(data_path: str, is_train: bool = True, download: bool = False):
    """
    Instancia y valida la integridad estructural del conjunto de datos KMNIST.

    Verifica la existencia de los archivos comprimidos en el directorio especificado, 
    instancia el objeto del conjunto de datos mediante torchvision y comprueba que 
    los tensores resultantes cumplan con las dimensiones (1, 28, 28) y el rango de valores esperado.

    Argumentos:
        data_path (str): Ruta base del directorio donde se alojan o descargarán los datos.
        is_train (bool, opcional): Indica si se carga la partición de entrenamiento (True) o de prueba (False). Por defecto es True.
        download (bool, opcional): Habilita la descarga automática desde internet si los archivos requeridos no se detectan. Por defecto es False.

    Retorna:
        torchvision.datasets.KMNIST: Objeto de conjunto de datos instanciado y validado.

    Excepciones:
        RuntimeError: Si ocurre un error interno en PyTorch durante la carga (ej. corrupción de datos o fallo de validación MD5).
        ValueError: Si las dimensiones espaciales del tensor extraído no coinciden con la estructura esperada (1, 28, 28).
    """
    data_root = Path(data_path)
    raw_folder = data_root / 'KMNIST' / 'raw'

    print(f"Verificando directorio: {raw_folder.resolve()}")

    expected_files = {
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    }

    if raw_folder.exists():
        existing_files = {f.name for f in raw_folder.iterdir() if f.is_file()}
        missing_files = expected_files - existing_files
        
        if not missing_files:
            print("Archivos de origen detectados y validados por nomenclatura.")
        else:
            print(f"Advertencia: Faltan los siguientes archivos requeridos: {missing_files}")
    else:
        print("Advertencia: El directorio raw no existe.")

    try:
        print(f"Iniciando procesamiento de KMNIST (train={is_train})...")
        dataset = torchvision.datasets.KMNIST(
            root=data_root, 
            train=is_train, 
            download=download, 
            transform=transforms.ToTensor()
        )
        
        img_sample, _ = dataset[0]

        print("\n=== Verificación dimensional ===")
        print(f"Formato del tensor  : {img_sample.shape}")
        print(f"Rango de valores    : [{img_sample.min():.4f}, {img_sample.max():.4f}]")

        expected_shape = (1, 28, 28)
        if img_sample.shape != expected_shape:
            raise ValueError(f"Discrepancia dimensional. Esperado {expected_shape}, obtenido {img_sample.shape}")
            
        print("Dataset cargado, instanciado y verificado correctamente.")
        return dataset

    except RuntimeError as e:
        print(f"\nERROR DE EJECUCIÓN (PyTorch): {e}")
        print("Posible causa: Corrupción de datos o discrepancia en la validación del checksum MD5.")
        raise
        
    except ValueError as e:
        print(f"\nERROR DE VALIDACIÓN (Datos): {e}")
        print("Posible causa: Transformación incorrecta o alteración de la estructura del dataset base.")
        raise

def calculate_dataset_statistics(dataset: torch.utils.data.Dataset, batch_size: int = 60000) -> tuple:
    """
    Calcula la media y la desviación estándar global de un conjunto de datos de imágenes.

    Argumentos:
        dataset (torch.utils.data.Dataset): Instancia del conjunto de datos de PyTorch a procesar.
        batch_size (int, opcional): Número de muestras a cargar simultáneamente en memoria para el cálculo. Por defecto es 60000.

    Retorna:
        tuple: Tupla compuesta por dos valores de coma flotante que representan la media y la desviación estándar (mean_val, std_val).

    Notas:
        La función requiere que el sistema disponga de memoria RAM suficiente para apilar el tamaño 
        del lote especificado en un único tensor computacional.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_stack, _ = next(iter(loader))
    
    mean_val = data_stack.mean().item()
    std_val = data_stack.std().item()
    
    return mean_val, std_val

def plot_class_distribution(targets: list, labels_map: dict) -> None:
    """
    Genera y despliega un gráfico de barras que representa la distribución de frecuencias absolutas por clase.

    Argumentos:
        targets (list): Colección de etiquetas numéricas correspondientes a las muestras del conjunto de datos.
        labels_map (dict): Diccionario que vincula los identificadores numéricos de clase con sus representaciones textuales.

    Retorna:
        None
    """
    counts = Counter(targets)
    sorted_indices = sorted(counts.keys())
    classes = [labels_map[i] for i in sorted_indices]
    values = [counts[i] for i in sorted_indices]

    plt.figure(figsize=(10, 4))
    plt.bar(classes, values, color='#4a90e2', alpha=0.9, edgecolor='black')
    plt.title('Distribución de Clases (Training Set)')
    plt.ylabel('Número de muestras')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

def plot_sample_images(dataset: torch.utils.data.Dataset, labels_map: dict, num_classes: int = 10) -> None:
    """
    Genera una cuadrícula visual mostrando una muestra representativa (la primera ocurrencia) para cada clase del conjunto de datos.

    Argumentos:
        dataset (torch.utils.data.Dataset): Instancia del conjunto de datos de PyTorch instanciada previamente.
        labels_map (dict): Diccionario que vincula los identificadores numéricos de clase con sus representaciones textuales.
        num_classes (int, opcional): Número total de clases independientes a visualizar. Por defecto es 10.

    Retorna:
        None

    Notas:
        La función asume que el objeto `dataset` expone el atributo `targets` con la totalidad de las etiquetas. 
        Esta propiedad es estándar en conjuntos de datos nativos de torchvision, pero su disponibilidad en 
        implementaciones personalizadas es [NO VERIFICABLE].
    """
    all_labels = dataset.targets.tolist()
    fig = plt.figure(figsize=(12, 5))

    for i in range(num_classes): 
        idx = (torch.tensor(all_labels) == i).nonzero(as_tuple=True)[0][0].item()
        img, label = dataset[idx]
        
        ax_img = plt.subplot(2, 5, i + 1)
        ax_img.imshow(img.squeeze(), cmap='gray')
        ax_img.set_title(f"{labels_map[label]} (ID:{label})")
        ax_img.axis('off')

    plt.tight_layout()
    plt.show()

def get_transforms(mean_val: float, std_val: float, is_train: bool = True) -> transforms.Compose:
    """
    Construye el pipeline de transformaciones de imágenes mediante torchvision.
    Aplica aumento de datos geométrico (rotación y traslación afín) exclusivamente
    para el conjunto de entrenamiento con el objetivo de reducir el sobreajuste.

    Argumentos:
        mean_val (float): Valor de la media calculada del conjunto de datos para la normalización.
        std_val (float): Valor de la desviación estándar calculada del conjunto de datos.
        is_train (bool, opcional): Bandera que determina si se inyectan transformaciones
            de aumento de datos (True) o solo transformaciones base (False). Por defecto es True.

    Retorna:
        torchvision.transforms.Compose: Composición secuencial de transformaciones aplicables a tensores.
    """
    transform_list = []
    
    if is_train:
        transform_list.extend([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((mean_val,), (std_val,))
    ])
    
    return transforms.Compose(transform_list)


def prepare_dataloaders(ruta_base: str, mean_val: float, std_val: float, 
                        batch_size_train: int, batch_size_eval: int, 
                        num_workers: int, generator: torch.Generator) -> tuple:
    """
    Instancia los conjuntos de datos KMNIST, aplica un particionado riguroso (80/20) y construye
    los iteradores (DataLoaders). Previene la filtración de aumento de datos aislando las
    transformaciones del conjunto de validación.

    Argumentos:
        ruta_base (str): Directorio raíz donde se alojan o descargarán los datos.
        mean_val (float): Media del conjunto de datos para la normalización.
        std_val (float): Desviación estándar del conjunto de datos para la normalización.
        batch_size_train (int): Tamaño del lote para el iterador de entrenamiento.
        batch_size_eval (int): Tamaño del lote para los iteradores de validación y prueba.
        num_workers (int): Número de subprocesos paralelos para la carga de datos.
        generator (torch.Generator): Generador de números pseudoaleatorios para asegurar
            la reproducibilidad estricta del particionado.

    Retorna:
        tuple: Tupla estructurada conteniendo:
            - train_loader (DataLoader): Iterador para el conjunto de entrenamiento.
            - val_loader (DataLoader): Iterador para el conjunto de validación.
            - test_loader (DataLoader): Iterador para el conjunto de prueba.
            - train_size (int): Magnitud escalar del conjunto de entrenamiento.
            - val_size (int): Magnitud escalar del conjunto de validación.
            - test_size (int): Magnitud escalar del conjunto de prueba (total del test_dataset).
    """
    transform_train = get_transforms(mean_val, std_val, is_train=True)
    transform_eval = get_transforms(mean_val, std_val, is_train=False)

    full_train_dataset_aug = torchvision.datasets.KMNIST(
        root=ruta_base, train=True, download=False, transform=transform_train
    )
    full_train_dataset_eval = torchvision.datasets.KMNIST(
        root=ruta_base, train=True, download=False, transform=transform_eval
    )
    
    test_dataset = torchvision.datasets.KMNIST(
        root=ruta_base, train=False, download=False, transform=transform_eval
    )

    train_size = int(0.8 * len(full_train_dataset_aug))
    val_size = len(full_train_dataset_aug) - train_size

    train_dataset, _ = random_split(
        full_train_dataset_aug, [train_size, val_size], generator=generator
    )
    _, val_dataset = random_split(
        full_train_dataset_eval, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, 
                              num_workers=num_workers, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False, 
                            num_workers=num_workers, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False, 
                             num_workers=num_workers, generator=generator)

    return train_loader, val_loader, test_loader, train_size, val_size, len(test_dataset)

def plot_normalized_batch(images_batch: torch.Tensor, labels_batch: torch.Tensor) -> None:
    """
    Despliega visualmente un lote de imágenes previamente sometido al pipeline de transformación y normalización.

    Argumentos:
        images_batch (torch.Tensor): Tensor multidimensional que contiene el lote de imágenes (B, C, H, W).
        labels_batch (torch.Tensor): Tensor unidimensional con las etiquetas numéricas correspondientes al lote.

    Retorna:
        None
    """
    plt.figure(figsize=(12, 2))
    grid_img = torchvision.utils.make_grid(images_batch, nrow=8, padding=2, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f"Visualización del Batch (Aumentado y Normalizado) - Etiquetas: {labels_batch.tolist()}")
    plt.show()


class EarlyStopping:
    """
    Mecanismo de control (callback) que detiene el entrenamiento de la red neuronal si la
    métrica de validación evaluada no experimenta una mejora sustancial tras un número definido de épocas.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Inicializa el controlador de Early Stopping.

        Argumentos:
            patience (int, opcional): Número de épocas consecutivas sin mejora permitidas antes
                de detener la ejecución. Por defecto es 5.
            min_delta (float, opcional): Umbral mínimo de cambio requerido para que se considere
                una mejora real. Por defecto es 0.0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        """
        Evalúa la pérdida de validación actual frente al mejor registro histórico.

        Argumentos:
            val_loss (float): Valor escalar de la función de pérdida sobre el conjunto de validación
                en la iteración o época en curso.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelCheckpoint:
    """
    Mecanismo de persistencia en memoria (callback) que conserva los pesos estructurales
    del modelo correspondientes a la iteración con menor error de validación empírico.
    """
    def __init__(self):
        """
        Inicializa los registros para monitorizar el estado óptimo del modelo durante la ejecución.
        """
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def __call__(self, model: torch.nn.Module, val_loss: float) -> bool:
        """
        Evalúa si el modelo actual supera el rendimiento histórico y, de ser así, actualiza su estado en memoria.

        Argumentos:
            model (torch.nn.Module): Instancia de la arquitectura de red neuronal en evaluación.
            val_loss (float): Métrica escalar de pérdida de validación actual.

        Retorna:
            bool: Devuelve True si se actualizó el estado óptimo en memoria, False en caso contrario.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            return True
        return False

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    """
    Ejecuta el bucle principal de entrenamiento y validación para un modelo de clasificación multiclase en PyTorch.
    
    Integra mecanismos de control automáticos incluyendo detención temprana (Early Stopping), 
    persistencia del mejor estado en memoria (Model Checkpointing) y ajuste dinámico de la tasa de aprendizaje.
    Al finalizar la ejecución, restaura automáticamente los pesos del modelo correspondientes 
    a la iteración con menor pérdida de validación.
    
    Argumentos:
        model (torch.nn.Module): Instancia de la arquitectura de red neuronal a entrenar.
        train_loader (torch.utils.data.DataLoader): Iterador del conjunto de datos particionado para entrenamiento.
        val_loader (torch.utils.data.DataLoader): Iterador del conjunto de datos particionado para validación.
        criterion (callable): Función de pérdida escalar a optimizar (ej. CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Algoritmo iterativo de optimización de pesos (ej. Adam).
        num_epochs (int): Límite superior de iteraciones completas sobre el conjunto de entrenamiento.
        device (torch.device): Dispositivo de aceleración de hardware asignado ('cpu' o 'cuda').
        scheduler (torch.optim.lr_scheduler._LRScheduler, opcional): Planificador para la reducción dinámica 
            de la tasa de aprendizaje ante estancamientos en la convergencia. Por defecto es None.
            
    Retorna:
        tuple: Tupla estructurada conteniendo:
            - history (dict): Diccionario con la evolución histórica de las métricas 
              ('train_loss', 'val_loss', 'val_acc', 'val_f1') recolectadas por época.
            - model (torch.nn.Module): Modelo final, restaurado estructuralmente a su estado óptimo.
    
    Notas:
        La función asume un entorno operativo de clasificación multiclase estándar, extrayendo la 
        predicción final de los logits mediante la operación computacional `torch.max(outputs, 1)`.
    """
    early_stopping = EarlyStopping(patience=5)
    model_checkpoint = ModelCheckpoint()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    model = model.to(device)
    print(f"Inicio de entrenamiento en: {device}")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        model.eval()
        val_running_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        metrics = calculate_metrics(all_labels, all_preds)
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['f1_macro'])
        
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_epoch_loss)
        
        saved = model_checkpoint(model, val_epoch_loss)
        save_msg = "(Mejor Modelo Guardado)" if saved else ""        
      
        print(f"Epoch {epoch+1:02d}/{num_epochs} | LR: {current_lr:.2e} | Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1_macro']:.4f} {save_msg}")
        
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            print(f"Early Stopping activado en la época {epoch+1}. Convergencia alcanzada.")
            break
            
    total_time = time.time() - start_time
    print(f"\nEntrenamiento finalizado en {total_time:.2f} segundos.")
    
    if model_checkpoint.best_model_state:
        model.load_state_dict(model_checkpoint.best_model_state)
        print("Pesos restaurados al punto de menor pérdida de validación.")
        
    return history, model

def load_checkpoint(save_path: str, model: nn.Module, device: torch.device) -> tuple:
    """
    Restaura los pesos estructurales y las métricas históricas de un modelo persistido en disco.
    
    Argumentos:
        save_path (str): Ruta relativa o absoluta del archivo serializado que contiene el punto de control (.pth).
        model (torch.nn.Module): Instancia base de la red neuronal sobre la que se inyectarán los pesos.
        device (torch.device): Dispositivo de cómputo de destino para mapear los tensores cargados.
        
    Retorna:
        tuple: Tupla posicional compuesta por:
            - success (bool): Bandera indicando si el proceso de carga se ejecutó sin excepciones.
            - model_state_dict (dict | None): Diccionario completo de pesos del modelo, o None si la carga falló.
            - metrics (dict): Diccionario con las métricas guardadas durante el entrenamiento original, 
              o un diccionario vacío en caso de error o inexistencia.
              
    Excepciones:
        Captura cualquier excepción genérica (`Exception`) originada por rutas inválidas, corrupción del archivo 
        o desajuste arquitectónico, imprimiendo el rastreo en la salida estándar sin interrumpir el flujo global.
        
    Notas:
        La instrucción de carga utiliza el flag `weights_only=False`. Esta parametrización permite deserializar 
        objetos Python arbitrarios por compatibilidad heredada, asumiendo que la procedencia del archivo `.pth` 
        es internamente segura.
    """
    try:
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        metrics = ckpt.get('metrics', {})
        return True, ckpt['model_state_dict'], metrics
    except Exception as e:
        print(f"Error de carga de checkpoint: {e}")
        return False, None, {}

def evaluate_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> tuple:
    """
    Ejecuta el paso de inferencia determinista de un modelo entrenado sobre un conjunto de datos 
    de prueba independiente, calculando su nivel de generalización mediante métricas de clasificación.
    
    Inhabilita dinámicamente el motor de autograd (`torch.no_grad()`) para suprimir la recopilación 
    de gradientes, reduciendo sustancialmente el impacto en la memoria VRAM y acelerando el tiempo de cómputo.
    
    Argumentos:
        model (torch.nn.Module): Arquitectura de red neuronal previamente entrenada y optimizada.
        test_loader (torch.utils.data.DataLoader): Iterador configurado que provee los tensores de imágenes 
            y sus respectivas anotaciones numéricas reales.
        device (torch.device): Dispositivo de hardware objetivo para procesar los cálculos tensoriales.
        
    Retorna:
        tuple: Tupla numérica conteniendo:
            - test_acc (float): Tasa de exactitud global (Accuracy) sobre todo el conjunto evaluado.
            - test_f1 (float): F1-Score agrupado bajo la modalidad macro (no ponderado por soporte de clases).
            
    Notas:
        La función asume explícitamente que los valores emitidos por el modelo son un mapeo de logits,
        extrayendo la clase inferida mediante `torch.max(outputs.data, 1)`.
    """
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    test_acc = np.sum(test_preds == test_labels) / len(test_labels)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    
    return test_acc, test_f1
def plot_training_history(history: dict) -> None:
    """
    Genera un gráfico lineal bidimensional que visualiza la evolución de la función de pérdida 
    tanto para el conjunto de entrenamiento como para el de validación a lo largo de las épocas.

    Argumentos:
        history (dict): Diccionario que contiene las métricas históricas del entrenamiento. 
            Debe incluir obligatoriamente las claves 'train_loss' y 'val_loss', asociadas a listas numéricas.

    Retorna:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Pérdida de Entrenamiento', linewidth=2)
    plt.plot(history['val_loss'], label='Pérdida de Validación', linewidth=2, linestyle='--')
    plt.title('Dinámica de Convergencia')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (Cross Entropy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple:
    """
    Ejecuta el paso de inferencia de un modelo sobre un iterador de datos (DataLoader) completo, 
    extrayendo exhaustivamente las predicciones, distribuciones de probabilidad y tensores originales.

    La evaluación se realiza suprimiendo dinámicamente el cálculo de gradientes (torch.no_grad()) 
    para optimizar el uso de memoria VRAM.

    Argumentos:
        model (torch.nn.Module): Arquitectura de red neuronal previamente entrenada.
        dataloader (torch.utils.data.DataLoader): Iterador que provee los lotes de imágenes y sus etiquetas.
        device (torch.device): Dispositivo de hardware objetivo para procesar la inferencia ('cpu' o 'cuda').

    Retorna:
        tuple: Tupla posicional estructurada conteniendo:
            - all_labels (numpy.ndarray): Arreglo unidimensional con las etiquetas reales de todas las muestras.
            - all_preds (numpy.ndarray): Arreglo unidimensional con las clases predichas (índices de máxima probabilidad).
            - all_probs (numpy.ndarray): Arreglo matricial bidimensional con las probabilidades Softmax por clase.
            - all_images (list): Lista de tensores originales de las imágenes procesadas, transferidos a memoria CPU.
    """
    model.eval()
    all_preds, all_labels, all_probs, all_images = [], [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(inputs.cpu())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_images

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> None:
    """
    Renderiza una matriz de confusión anotada utilizando un mapa de calor (heatmap) para evaluar 
    visualmente el rendimiento predictivo del modelo desagregado por clase.

    Calcula la exactitud global (accuracy) del conjunto y la integra dinámicamente en el título del gráfico.

    Argumentos:
        y_true (numpy.ndarray): Arreglo unidimensional con las etiquetas numéricas reales (ground truth).
        y_pred (numpy.ndarray): Arreglo unidimensional con las etiquetas numéricas predichas por el modelo.
        class_names (list): Lista de cadenas de texto (strings) con los nombres de las clases en el orden correspondiente a sus índices.

    Retorna:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matriz de Confusión Global (Exactitud: {acc:.2%})")
    plt.ylabel("Etiqueta Real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.show()

def plot_error_gallery(images: list, y_true: np.ndarray, y_pred: np.ndarray, 
                       class_names: list, num_images: int = 10) -> None:
    """
    Identifica, selecciona y despliega una muestra aleatoria de imágenes que han sido 
    clasificadas incorrectamente por el modelo, facilitando la auditoría cualitativa de errores.

    Argumentos:
        images (list): Colección de tensores de imágenes correspondientes al conjunto evaluado.
        y_true (numpy.ndarray): Arreglo unidimensional con las etiquetas numéricas reales.
        y_pred (numpy.ndarray): Arreglo unidimensional con las etiquetas numéricas predichas.
        class_names (list): Lista de cadenas con las denominaciones textuales de cada clase.
        num_images (int, opcional): Límite máximo de imágenes erróneas a representar en la cuadrícula. Por defecto es 10.

    Retorna:
        None

    Notas:
        La función aplica una transformación de desnormalización explícita sobre los tensores asumiendo 
        parámetros estadísticos rígidamente fijados (std: 0.3483, mean: 0.1918). La precisión visual de 
        esta corrección fuera del dominio específico del dataset KMNIST se considera [NO VERIFICABLE].
    """
    wrong_indices = np.where(y_true != y_pred)[0]
    
    if len(wrong_indices) == 0:
        print("No se registraron errores de clasificación.")
        return

    num_show = min(num_images, len(wrong_indices))
    indices_to_show = np.random.choice(wrong_indices, size=num_show, replace=False)

    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices_to_show):
        img_tensor = images[idx]
        label_real = y_true[idx]
        label_pred = y_pred[idx]
        
        # Desnormalización aproximada para visualización (KMNIST)
        img_disp = img_tensor.squeeze().numpy() * 0.3483 + 0.1918
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_disp, cmap='gray')
        
        title_str = f"Real: {class_names[label_real]}\nPred: {class_names[label_pred]}"
        plt.title(title_str, color='red', fontsize=11, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def get_japanese_font(size: int = 22):
    """
    Localiza de manera heurística y carga una fuente tipográfica compatible con caracteres 
    CJK (Chino, Japonés, Coreano) instalada en el sistema anfitrión.

    Realiza una búsqueda iterativa sobre las fuentes del sistema utilizando matplotlib, 
    intentando coincidir con una lista de familias tipográficas conocidas. En caso de no 
    hallar coincidencias o fallar en la carga, implementa un mecanismo de contingencia (fallback) 
    cargando la fuente por defecto de PIL.

    Argumentos:
        size (int, opcional): Tamaño de la fuente tipográfica en puntos. Por defecto es 22.

    Retorna:
        PIL.ImageFont.FreeTypeFont o PIL.ImageFont.ImageFont: Objeto de fuente tipográfica instanciado 
        y listo para ser utilizado en el renderizado de texto.

    Notas:
        La disponibilidad, ruta y correcta carga de la fuente dependen enteramente de la configuración 
        del sistema operativo subyacente y los paquetes instalados. Este comportamiento es 
        completamente dependiente del entorno e intrínsecamente [NO VERIFICABLE].
    """
    candidates = ['NotoSansCJK-Regular.ttc', 'ipag.ttf', 'msgothic.ttc', 'Arial Unicode.ttf']
    font_path = None
    system_fonts = fm.findSystemFonts()
    
    for font in system_fonts:
        for cand in candidates:
            if cand.lower() in font.lower():
                font_path = font
                break
        if font_path: break
            
    try:
        if font_path:
            return ImageFont.truetype(font_path, size)
        return ImageFont.load_default()
    except IOError:
        return ImageFont.load_default()

def create_image_from_text(char: str, mean_val: float, std_val: float, font_size: int = 22) -> torch.Tensor:
    """
    Genera sintéticamente una imagen en escala de grises de 28x28 píxeles renderizando un único 
    carácter de texto centrado, y la transforma en un tensor normalizado compatible con el 
    pipeline del conjunto de datos KMNIST.

    Argumentos:
        char (str): El carácter o cadena de texto (idealmente longitud 1) a renderizar en la imagen.
        mean_val (float): Valor de la media estadística a utilizar para la normalización del tensor.
        std_val (float): Valor de la desviación estándar a utilizar para la normalización del tensor.
        font_size (int, opcional): Tamaño de la tipografía para el renderizado. Por defecto es 22.

    Retorna:
        torch.Tensor: Tensor multidimensional de dimensiones (1, 1, 28, 28) que representa la 
        imagen renderizada y normalizada.

    Notas:
        Implementa un bloque try-except sobre los métodos `textbbox` y `textsize` para garantizar 
        retrocompatibilidad con diferentes versiones de la librería Pillow (PIL).
    """
    img = Image.new('L', (28, 28), color=0)
    draw = ImageDraw.Draw(img)
    font = get_japanese_font(size=font_size)
    
    try:
        left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
        w, h = right - left, bottom - top
    except AttributeError:
        w, h = draw.textsize(char, font=font)
        
    x, y = (28 - w) / 2, (28 - h) / 2
    draw.text((x, y), char, font=font, fill=255)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean_val,), (std_val,))
    ])
    
    return transform(img).unsqueeze(0)

def visualize_ood_confidence(model: nn.Module, input_tensor: torch.Tensor, 
                             input_label: str, class_names: list, device: torch.device) -> None:
    """
    Ejecuta el paso de inferencia de un modelo sobre un tensor de entrada fuera de la 
    distribución de entrenamiento (Out-of-Distribution, OOD) y visualiza cualitativamente 
    la certeza predictiva generada.

    Despliega un panel bidimensional que incluye la imagen de entrada desnormalizada junto 
    con un diagrama de barras que expone la distribución de probabilidades cruzada (Softmax) 
    asignada por el modelo a cada clase posible.

    Argumentos:
        model (torch.nn.Module): Arquitectura de la red neuronal preentrenada.
        input_tensor (torch.Tensor): Tensor de entrada (usualmente sintético) con dimensiones compatibles (1, 1, 28, 28).
        input_label (str): Etiqueta descriptiva de la entrada sintética para el título del gráfico.
        class_names (list): Lista de representaciones textuales de las clases para el eje de abscisas del gráfico.
        device (torch.device): Dispositivo de cómputo donde se alojan el modelo y el tensor ('cpu' o 'cuda').

    Retorna:
        None

    Notas:
        La función aplica un cálculo estático y fijo para desnormalizar la imagen y mostrarla correctamente 
        (`* 0.3483 + 0.1918`). Asumir que toda entrada futura obedecerá rígidamente a estos parámetros 
        estadísticos sin variación es una simplificación cuyo comportamiento generalizado es [NO VERIFICABLE].
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = torch.argmax(outputs, 1).item()
    
    pred_char_rom = class_names[pred_idx]
    
    plt.figure(figsize=(9, 3))
    
    plt.subplot(1, 2, 1)
    img_in = input_tensor.cpu().squeeze().numpy() * 0.3483 + 0.1918
    plt.imshow(img_in, cmap='gray')
    plt.title(f"Entrada Sintética OOD:\n'{input_label}'", fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    colors = ['#e0e0e0'] * 10
    colors[pred_idx] = '#3498db'
    plt.bar(class_names, probs, color=colors)
    plt.title(f"Clasificación: {pred_char_rom} (Certeza: {probs[pred_idx]*100:.1f}%)", fontsize=11)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()