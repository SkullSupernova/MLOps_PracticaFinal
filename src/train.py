# =============================================================================
# train.py
# Pipeline de entrenamiento supervisado para clasificación de caracteres
# Kuzushiji (KMNIST) mediante la arquitectura CNN_ResNet.
#
# Responsabilidad: configuración del entorno, carga y particionado de datos,
# instanciación del modelo, gestión de checkpoints, bucle de entrenamiento
# y persistencia del modelo resultante.
#
# Uso como script independiente:
#   python train.py [--opciones]
#
# Uso como módulo importado desde main.py:
#   from train import run_training
# =============================================================================

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Resolución de la ruta raíz del proyecto para importaciones locales.
# Se asume que train.py se encuentra en el directorio raíz del proyecto y que
# el módulo src/ existe como subdirectorio directo de dicho raíz.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import (
    CNN_ResNet,
    calculate_dataset_statistics,
    cargar_y_validar_kmnist,
    load_checkpoint,
    prepare_dataloaders,
    set_seed,
    setup_environment,
    train_model,
)

# =============================================================================
# Constantes de configuración por defecto
# =============================================================================
DEFAULT_DATA_PATH       = os.path.join(BASE_DIR, 'data')
DEFAULT_MODELS_DIR      = os.path.join(BASE_DIR, 'models')
DEFAULT_CHECKPOINT_NAME = 'ResNet_Final_Combined.pth'
DEFAULT_MAX_EPOCHS      = 200
DEFAULT_LR              = 1e-3
DEFAULT_L2              = 1e-4
DEFAULT_BATCH_TRAIN     = 8
DEFAULT_BATCH_EVAL      = 64
DEFAULT_SEED            = 42

# Valores estadísticos de respaldo (calculados previamente sobre KMNIST train)
MEAN_FALLBACK = 0.1918
STD_FALLBACK  = 0.3483


# =============================================================================
# Función auxiliar: construcción de objetos de optimización
# =============================================================================
def build_training_objects(
    model: nn.Module,
    lr: float,
    l2: float,
) -> tuple:
    """
    Instancia la función de pérdida, el optimizador y el planificador de tasa
    de aprendizaje utilizados durante el entrenamiento.

    Argumentos:
        model (nn.Module): Arquitectura de red neuronal sobre la que se aplica
            el optimizador.
        lr (float): Tasa de aprendizaje inicial para el algoritmo Adam.
        l2 (float): Factor de regularización L2 aplicado como weight decay.

    Retorna:
        tuple:
            - criterion (nn.CrossEntropyLoss): Función de pérdida multiclase.
            - optimizer (optim.Adam): Optimizador con regularización L2.
            - scheduler (optim.lr_scheduler.ReduceLROnPlateau): Planificador
              de reducción dinámica de LR ante estancamiento de la validación.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
    )
    return criterion, optimizer, scheduler


# =============================================================================
# Función principal de entrenamiento (importable)
# =============================================================================
def run_training(
    data_path:       str   = DEFAULT_DATA_PATH,
    models_dir:      str   = DEFAULT_MODELS_DIR,
    checkpoint_name: str   = DEFAULT_CHECKPOINT_NAME,
    max_epochs:      int   = DEFAULT_MAX_EPOCHS,
    lr:              float = DEFAULT_LR,
    l2:              float = DEFAULT_L2,
    batch_train:     int   = DEFAULT_BATCH_TRAIN,
    batch_eval:      int   = DEFAULT_BATCH_EVAL,
    force_train:     bool  = False,
    seed:            int   = DEFAULT_SEED,
) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento supervisado para KMNIST.

    El flujo cubre las siguientes etapas en orden secuencial:
      1. Configuración del hardware y reproducibilidad.
      2. Carga y validación estructural del dataset.
      3. Cálculo de estadísticas de normalización.
      4. Construcción de DataLoaders (train / val / test).
      5. Instanciación del modelo CNN_ResNet y objetos de optimización.
      6. Gestión de checkpoint: carga si existe y no se fuerza reentrenamiento.
      7. Entrenamiento mediante `train_model` (definido en src/utils.py).
      8. Persistencia del modelo y métricas finales.

    Argumentos:
        data_path (str): Ruta al directorio raíz de los datos KMNIST.
        models_dir (str): Directorio donde se guardará el archivo .pth.
        checkpoint_name (str): Nombre del archivo de checkpoint de salida.
        max_epochs (int): Número máximo de épocas de entrenamiento.
        lr (float): Tasa de aprendizaje inicial (Adam).
        l2 (float): Coeficiente de regularización L2 (weight decay).
        batch_train (int): Tamaño del lote para el DataLoader de entrenamiento.
        batch_eval (int): Tamaño del lote para validación y test.
        force_train (bool): Si es True, ignora checkpoints previos y reentrena.
        seed (int): Semilla global para reproducibilidad.

    Retorna:
        dict: Diccionario con las claves:
            - 'model'        : instancia nn.Module con los pesos óptimos.
            - 'device'       : torch.device utilizado durante el entrenamiento.
            - 'mean_val'     : media de normalización del dataset.
            - 'std_val'      : desviación estándar de normalización.
            - 'train_loader' : DataLoader de entrenamiento.
            - 'val_loader'   : DataLoader de validación.
            - 'test_loader'  : DataLoader de prueba.
            - 'history'      : dict con métricas por época (None si se cargó
                               checkpoint sin reentrenamiento).
            - 'save_path'    : ruta al archivo .pth persistido o cargado.
    """
    # ------------------------------------------------------------------
    # 1. Entorno y reproducibilidad
    # ------------------------------------------------------------------
    device, num_workers = setup_environment()
    g_cpu = set_seed(seed=seed)

    # ------------------------------------------------------------------
    # 2. Carga y validación del dataset (sin transformaciones de aug.)
    # ------------------------------------------------------------------
    ds_train_raw = cargar_y_validar_kmnist(
        data_path=data_path, is_train=True, download=False
    )

    # ------------------------------------------------------------------
    # 3. Estadísticas de normalización
    # ------------------------------------------------------------------
    try:
        mean_val, std_val = calculate_dataset_statistics(ds_train_raw)
        print(f"\nEstadísticas calculadas -> Media: {mean_val:.4f} | "
              f"Desv. típica: {std_val:.4f}")
    except Exception as e:
        mean_val, std_val = MEAN_FALLBACK, STD_FALLBACK
        print(f"Advertencia: Cálculo de estadísticas fallido ({e}). "
              f"Usando valores de respaldo: mean={mean_val}, std={std_val}.")

    # ------------------------------------------------------------------
    # 4. DataLoaders con separación estricta de transformaciones
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, train_sz, val_sz, test_sz = (
        prepare_dataloaders(
            ruta_base=data_path,
            mean_val=mean_val,
            std_val=std_val,
            batch_size_train=batch_train,
            batch_size_eval=batch_eval,
            num_workers=num_workers,
            generator=g_cpu,
        )
    )

    print(f"\n=== Resumen del particionado ===")
    print(f"Entrenamiento : {train_sz:,} muestras")
    print(f"Validación    : {val_sz:,} muestras")
    print(f"Test          : {test_sz:,} muestras")

    # ------------------------------------------------------------------
    # 5. Instanciación del modelo y objetos de optimización
    # ------------------------------------------------------------------
    model = CNN_ResNet().to(device)
    criterion, optimizer, scheduler = build_training_objects(model, lr=lr, l2=l2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Configuración del modelo ===")
    print(f"Arquitectura : CNN_ResNet ({n_params:,} parámetros)")
    print(f"Optimizador  : Adam (LR={lr}, L2={l2})")
    print(f"Planificador : ReduceLROnPlateau (factor=0.1, paciencia=5)")
    print(f"Épocas máx.  : {max_epochs}")

    # ------------------------------------------------------------------
    # 6. Gestión de checkpoint
    # ------------------------------------------------------------------
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, checkpoint_name)
    history = None

    if os.path.exists(save_path) and not force_train:
        print(f"\nCheckpoint previo detectado: {save_path}")
        success, _, metrics = load_checkpoint(save_path, model, device)
        if success:
            val_acc  = metrics.get('val_acc',  'N/A')
            val_loss = metrics.get('val_loss', 'N/A')
            if isinstance(val_acc, float):
                print(f"Métricas recuperadas -> Val Acc: {val_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}")
            else:
                print(f"Métricas recuperadas -> {metrics}")
            print("Carga exitosa. Entrenamiento omitido.\n")
            return {
                'model':        model,
                'device':       device,
                'mean_val':     mean_val,
                'std_val':      std_val,
                'train_loader': train_loader,
                'val_loader':   val_loader,
                'test_loader':  test_loader,
                'history':      history,
                'save_path':    save_path,
            }

    # ------------------------------------------------------------------
    # 7. Entrenamiento
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nIniciando entrenamiento...")
    history, model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=max_epochs,
        device=device,
        scheduler=scheduler,
    )

    # ------------------------------------------------------------------
    # 8. Persistencia
    # ------------------------------------------------------------------
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'metrics': {
                'val_acc':  history['val_acc'][-1],
                'val_loss': history['val_loss'][-1],
            },
        },
        save_path,
    )
    print(f"Modelo persistido en: {save_path}")

    return {
        'model':        model,
        'device':       device,
        'mean_val':     mean_val,
        'std_val':      std_val,
        'train_loader': train_loader,
        'val_loader':   val_loader,
        'test_loader':  test_loader,
        'history':      history,
        'save_path':    save_path,
    }


# =============================================================================
# Punto de entrada como script independiente
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Entrenamiento supervisado KMNIST — CNN_ResNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data-path', type=str, default=DEFAULT_DATA_PATH,
        help='Ruta al directorio raíz de los datos KMNIST.',
    )
    parser.add_argument(
        '--models-dir', type=str, default=DEFAULT_MODELS_DIR,
        help='Directorio de salida para el archivo de checkpoint .pth.',
    )
    parser.add_argument(
        '--checkpoint-name', type=str, default=DEFAULT_CHECKPOINT_NAME,
        help='Nombre del archivo de checkpoint.',
    )
    parser.add_argument(
        '--max-epochs', type=int, default=DEFAULT_MAX_EPOCHS,
        help='Número máximo de épocas de entrenamiento.',
    )
    parser.add_argument(
        '--lr', type=float, default=DEFAULT_LR,
        help='Tasa de aprendizaje inicial (Adam).',
    )
    parser.add_argument(
        '--l2', type=float, default=DEFAULT_L2,
        help='Coeficiente de regularización L2 (weight decay).',
    )
    parser.add_argument(
        '--batch-train', type=int, default=DEFAULT_BATCH_TRAIN,
        help='Tamaño del lote para el DataLoader de entrenamiento.',
    )
    parser.add_argument(
        '--batch-eval', type=int, default=DEFAULT_BATCH_EVAL,
        help='Tamaño del lote para validación y test.',
    )
    parser.add_argument(
        '--force-train', action='store_true',
        help='Ignorar checkpoints previos y reentrenar desde cero.',
    )
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_SEED,
        help='Semilla global para reproducibilidad.',
    )
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    run_training(
        data_path=args.data_path,
        models_dir=args.models_dir,
        checkpoint_name=args.checkpoint_name,
        max_epochs=args.max_epochs,
        lr=args.lr,
        l2=args.l2,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        force_train=args.force_train,
        seed=args.seed,
    )
