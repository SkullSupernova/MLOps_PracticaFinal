# =============================================================================
# main.py
# Punto de entrada principal del proyecto KMNIST CNN_ResNet.
# =============================================================================

import argparse
import sys

import matplotlib
import numpy as np
import seaborn as sns
import sklearn
import torch
import torchvision

from src.logging_config import get_logger, setup_logging
from src.model import CNN_ResNet
from src.utils import (
    calculate_dataset_statistics,
    cargar_y_validar_kmnist,
    create_image_from_text,
    evaluate_model,
    get_predictions,
    load_checkpoint,
    plot_confusion_matrix,
    plot_error_gallery,
    plot_training_history,
    prepare_dataloaders,
    set_seed,
    setup_environment,
    visualize_ood_confidence,
)
from src.train import (
    DEFAULT_BATCH_EVAL, DEFAULT_BATCH_TRAIN, DEFAULT_CHECKPOINT_NAME,
    DEFAULT_DATA_PATH, DEFAULT_L2, DEFAULT_LR, DEFAULT_MAX_EPOCHS,
    DEFAULT_MODELS_DIR, DEFAULT_SEED, MEAN_FALLBACK, STD_FALLBACK,
    run_training,
)

CLASS_NAMES  = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
BASELINE_ACC = 0.9862
OOD_TESTS    = [
    ('Dígito 2', '2'), ('Dígito 7', '7'),
    ('Letra T',  'T'), ('Letra Z',  'Z'),
]


def print_version_info() -> None:
    print("=== Configuración del entorno ===")
    print(f"Python       : {sys.version.split()[0]}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"Torchvision  : {torchvision.__version__}")
    print(f"NumPy        : {np.__version__}")
    print(f"Matplotlib   : {matplotlib.__version__}")
    print(f"Seaborn      : {sns.__version__}")
    print(f"Scikit-learn : {sklearn.__version__}")


def resolve_dataset_statistics(data_path: str) -> tuple:
    logger = get_logger(__name__)
    try:
        ds = cargar_y_validar_kmnist(data_path=data_path, is_train=True)
        mean_val, std_val = calculate_dataset_statistics(ds)
        logger.info("Estadísticas: media=%.4f | std=%.4f", mean_val, std_val)
        return mean_val, std_val
    except Exception as exc:
        logger.warning("Estadísticas no calculadas (%s). Usando respaldo.", exc)
        return MEAN_FALLBACK, STD_FALLBACK


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    model = CNN_ResNet().to(device)
    success, _, metrics = load_checkpoint(checkpoint_path, model, device)
    if not success:
        raise RuntimeError(f"No se pudo cargar el checkpoint: {checkpoint_path}")
    return model


def run_evaluate(args: argparse.Namespace) -> None:
    logger = get_logger(__name__)
    logger.info("Evaluación iniciada. Checkpoint: %s", args.checkpoint)

    device, num_workers = setup_environment()
    g_cpu               = set_seed(seed=args.seed)
    mean_val, std_val   = resolve_dataset_statistics(args.data_path)

    _, _, test_loader, _, _, _ = prepare_dataloaders(
        ruta_base=args.data_path, mean_val=mean_val, std_val=std_val,
        batch_size_train=DEFAULT_BATCH_TRAIN, batch_size_eval=DEFAULT_BATCH_EVAL,
        num_workers=num_workers, generator=g_cpu,
    )
    model = load_model_from_checkpoint(args.checkpoint, device)
    test_acc, test_f1 = evaluate_model(model, test_loader, device)

    logger.info(
        "Accuracy: %.4f | F1 (macro): %.4f | Δ baseline: %+.4f",
        test_acc, test_f1, test_acc - BASELINE_ACC,
    )

    y_true, y_pred, _, images_test = get_predictions(model, test_loader, device)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
    plot_error_gallery(
        images_test, y_true, y_pred, CLASS_NAMES,
        mean_val=mean_val, std_val=std_val, num_images=10,
    )


def run_ood(args: argparse.Namespace) -> None:
    device, _         = setup_environment()
    set_seed(seed=args.seed)
    mean_val, std_val = resolve_dataset_statistics(args.data_path)
    model             = load_model_from_checkpoint(args.checkpoint, device)
    logger            = get_logger(__name__)

    for label, char_str in OOD_TESTS:
        try:
            tensor_ood = create_image_from_text(char_str, mean_val, std_val)
            visualize_ood_confidence(
                model, tensor_ood, label, CLASS_NAMES, device,
                mean_val=mean_val, std_val=std_val,
            )
        except Exception as exc:
            logger.error("Error en prueba OOD '%s': %s", label, exc)


def run_train(args: argparse.Namespace) -> None:
    result = run_training(
        data_path=args.data_path, models_dir=args.models_dir,
        checkpoint_name=args.checkpoint_name, max_epochs=args.max_epochs,
        lr=args.lr, l2=args.l2, batch_train=args.batch_train,
        batch_eval=args.batch_eval, force_train=args.force_train,
        seed=args.seed, use_wandb=getattr(args, 'use_wandb', False),
    )
    if result.get('history') is not None:
        plot_training_history(result['history'])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Clasificación KMNIST — CNN_ResNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='store_true')

    sub = parser.add_subparsers(dest='mode')

    p_train = sub.add_parser('train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_train.add_argument('--data-path',       type=str,   default=DEFAULT_DATA_PATH,       dest='data_path')
    p_train.add_argument('--models-dir',      type=str,   default=DEFAULT_MODELS_DIR,      dest='models_dir')
    p_train.add_argument('--checkpoint-name', type=str,   default=DEFAULT_CHECKPOINT_NAME, dest='checkpoint_name')
    p_train.add_argument('--max-epochs',      type=int,   default=DEFAULT_MAX_EPOCHS,      dest='max_epochs')
    p_train.add_argument('--lr',              type=float, default=DEFAULT_LR)
    p_train.add_argument('--l2',              type=float, default=DEFAULT_L2)
    p_train.add_argument('--batch-train',     type=int,   default=DEFAULT_BATCH_TRAIN,     dest='batch_train')
    p_train.add_argument('--batch-eval',      type=int,   default=DEFAULT_BATCH_EVAL,      dest='batch_eval')
    p_train.add_argument('--force-train',     action='store_true',                         dest='force_train')
    p_train.add_argument('--seed',            type=int,   default=DEFAULT_SEED)
    p_train.add_argument('--use-wandb',       action='store_true',                         dest='use_wandb')

    for name in ('evaluate', 'ood'):
        p = sub.add_parser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--checkpoint', type=str, required=True)
        p.add_argument('--data-path',  type=str, default=DEFAULT_DATA_PATH, dest='data_path')
        p.add_argument('--seed',       type=int, default=DEFAULT_SEED)

    return parser


def main() -> None:
    setup_logging()
    logger = get_logger(__name__)
    parser = _build_parser()
    args   = parser.parse_args()

    if args.version:
        print_version_info()
        sys.exit(0)

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    logger.info("Modo: %s", args.mode)
    handlers = {'train': run_train, 'evaluate': run_evaluate, 'ood': run_ood}

    try:
        handlers[args.mode](args)
    except Exception as exc:
        logger.critical("Fallo irrecuperable en modo '%s': %s", args.mode, exc, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()