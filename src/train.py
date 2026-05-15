# =============================================================================
# train.py
# Pipeline de entrenamiento supervisado para clasificación KMNIST.
# =============================================================================

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from src.logging_config import get_logger
from src.model import CNN_ResNet
from src.utils import (
    calculate_dataset_statistics,
    cargar_y_validar_kmnist,
    load_checkpoint,
    prepare_dataloaders,
    set_seed,
    setup_environment,
    train_model,
)

logger = get_logger(__name__)

# =============================================================================
# Constantes de configuración por defecto
# =============================================================================
from pathlib import Path

_PROJECT_ROOT        = Path(__file__).parent.parent.resolve()
DEFAULT_DATA_PATH    = str(_PROJECT_ROOT / 'data')
DEFAULT_MODELS_DIR   = str(_PROJECT_ROOT / 'models')
DEFAULT_CHECKPOINT_NAME = 'ResNet_Final_Combined.pth'
DEFAULT_MAX_EPOCHS   = 200
DEFAULT_LR           = 1e-3
DEFAULT_L2           = 1e-4
DEFAULT_BATCH_TRAIN  = 8
DEFAULT_BATCH_EVAL   = 64
DEFAULT_SEED         = 42
MEAN_FALLBACK        = 0.1918
STD_FALLBACK         = 0.3483


def build_training_objects(model: nn.Module, lr: float, l2: float) -> tuple:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5,
    )
    return criterion, optimizer, scheduler


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
    use_wandb:       bool  = False,
) -> dict:
    """Ejecuta el pipeline completo de entrenamiento supervisado para KMNIST."""

    device, num_workers = setup_environment()
    g_cpu = set_seed(seed=seed)

    # W&B (opcional)
    run = None
    if use_wandb:
        run = wandb.init(
            project="kmnist-resnet",
            config=dict(
                max_epochs=max_epochs, lr=lr, l2=l2,
                batch_train=batch_train, batch_eval=batch_eval, seed=seed,
            ),
        )

    # Estadísticas de normalización
    ds_train_raw = cargar_y_validar_kmnist(data_path=data_path, is_train=True)
    try:
        mean_val, std_val = calculate_dataset_statistics(ds_train_raw)
    except Exception as exc:
        mean_val, std_val = MEAN_FALLBACK, STD_FALLBACK
        logger.warning("Estadísticas no calculadas (%s). Usando valores de respaldo.", exc)

    train_loader, val_loader, test_loader, train_sz, val_sz, test_sz = prepare_dataloaders(
        ruta_base=data_path, mean_val=mean_val, std_val=std_val,
        batch_size_train=batch_train, batch_size_eval=batch_eval,
        num_workers=num_workers, generator=g_cpu,
    )

    model = CNN_ResNet().to(device)
    criterion, optimizer, scheduler = build_training_objects(model, lr=lr, l2=l2)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("CNN_ResNet instanciada. Parámetros: %d", n_params)

    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, checkpoint_name)
    history   = None

    if os.path.exists(save_path) and not force_train:
        success, _, metrics = load_checkpoint(save_path, model, device)
        if success:
            logger.info("Checkpoint cargado. Entrenamiento omitido.")
            return dict(
                model=model, device=device,
                mean_val=mean_val, std_val=std_val,
                train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader, history=None, save_path=save_path,
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    history, model = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, num_epochs=max_epochs,
        device=device, scheduler=scheduler,
    )

    # Serialización segura (weights_only compatible al cargarlo con weights_only=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'metrics': {
                'val_acc':  float(history['val_acc'][-1]),
                'val_loss': float(history['val_loss'][-1]),
            },
        },
        save_path,
    )

    logger.info("Modelo persistido en: %s", save_path)

    if run is not None:
        run.log({
            'val_acc':  history['val_acc'][-1],
            'val_loss': history['val_loss'][-1],
        })
        artifact = wandb.Artifact('kmnist-model', type='model')
        artifact.add_file(save_path)
        run.log_artifact(artifact)
        run.finish()

    return dict(
        model=model, device=device,
        mean_val=mean_val, std_val=std_val,
        train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, history=history, save_path=save_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Entrenamiento KMNIST — CNN_ResNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-path',        type=str,   default=DEFAULT_DATA_PATH,    dest='data_path')
    p.add_argument('--models-dir',       type=str,   default=DEFAULT_MODELS_DIR,   dest='models_dir')
    p.add_argument('--checkpoint-name',  type=str,   default=DEFAULT_CHECKPOINT_NAME, dest='checkpoint_name')
    p.add_argument('--max-epochs',       type=int,   default=DEFAULT_MAX_EPOCHS,   dest='max_epochs')
    p.add_argument('--lr',               type=float, default=DEFAULT_LR)
    p.add_argument('--l2',               type=float, default=DEFAULT_L2)
    p.add_argument('--batch-train',      type=int,   default=DEFAULT_BATCH_TRAIN,  dest='batch_train')
    p.add_argument('--batch-eval',       type=int,   default=DEFAULT_BATCH_EVAL,   dest='batch_eval')
    p.add_argument('--force-train',      action='store_true',                      dest='force_train')
    p.add_argument('--seed',             type=int,   default=DEFAULT_SEED)
    p.add_argument('--use-wandb',        action='store_true',                      dest='use_wandb')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    run_training(**vars(args))