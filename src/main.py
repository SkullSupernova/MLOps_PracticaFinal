# =============================================================================
# main.py
# Punto de entrada principal del proyecto KMNIST CNN_ResNet.
#
# Responsabilidad: orquestar los modos de ejecución disponibles mediante una
# interfaz de línea de comandos (CLI) basada en subcomandos. Gestiona la
# carga del modelo entrenado, la evaluación cuantitativa y cualitativa sobre
# el conjunto de prueba, y la evaluación de robustez ante datos fuera de la
# distribución de entrenamiento (OOD).
#
# Modos de ejecución:
#   python main.py train    [--opciones]   -> Delega en train.py::run_training
#   python main.py evaluate [--opciones]   -> Evaluación sobre test set
#   python main.py ood      [--opciones]   -> Evaluación de robustez OOD
#
# Verificación de versiones de librerías:
#   python main.py --version
# =============================================================================

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torchvision
import IPython

from logging_config import get_logger, setup_logging

# ---------------------------------------------------------------------------
# Resolución de la ruta raíz del proyecto para importaciones locales.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import (
    CNN_ResNet,
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

from train import (
    DEFAULT_BATCH_EVAL,
    DEFAULT_BATCH_TRAIN,
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_DATA_PATH,
    DEFAULT_L2,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MODELS_DIR,
    DEFAULT_SEED,
    MEAN_FALLBACK,
    STD_FALLBACK,
    run_training,
)

# =============================================================================
# Constantes del dominio
# =============================================================================
CLASS_NAMES  = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
BASELINE_ACC = 0.9862

# Casos de prueba OOD: (etiqueta descriptiva, carácter a renderizar)
OOD_TESTS = [
    ('Dígito 2', '2'),
    ('Dígito 7', '7'),
    ('Letra T',  'T'),
    ('Letra Z',  'Z'),
]

# =============================================================================
# Utilidades compartidas
# =============================================================================
def print_version_info() -> None:
    """Imprime las versiones de todas las librerías externas relevantes."""
    print("=== Configuración del entorno ===")
    print(f"Python Runtime       : {sys.version.split()[0]}")
    print(f"PyTorch Version      : {torch.__version__}")
    print(f"Torchvision Version  : {torchvision.__version__}")
    print(f"Numpy Version        : {np.__version__}")
    print(f"Pandas Version       : {pd.__version__}")
    print(f"Matplotlib Version   : {matplotlib.__version__}")
    print(f"Seaborn Version      : {sns.__version__}")
    print(f"Scikit-learn Version : {sklearn.__version__}")
    print(f"IPython Version      : {IPython.__version__}")


def resolve_dataset_statistics(data_path: str) -> tuple:
    """
    Calcula las estadísticas de normalización a partir del conjunto de
    entrenamiento de KMNIST. En caso de fallo, retorna los valores de
    respaldo precalculados con advertencia explícita.

    Argumentos:
        data_path (str): Ruta al directorio raíz de los datos.

    Retorna:
        tuple: (mean_val: float, std_val: float)
    """
    try:
        ds_train_raw = cargar_y_validar_kmnist(
            data_path=data_path, is_train=True, download=False
        )
        mean_val, std_val = calculate_dataset_statistics(ds_train_raw)
        print(f"Estadísticas calculadas -> Media: {mean_val:.4f} | "
              f"Desv. típica: {std_val:.4f}")
        return mean_val, std_val
    except Exception as e:
        print(f"Advertencia: No se pudieron calcular las estadísticas del "
              f"dataset ({e}). Usando valores de respaldo: "
              f"mean={MEAN_FALLBACK}, std={STD_FALLBACK}.")
        return MEAN_FALLBACK, STD_FALLBACK


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Instancia CNN_ResNet y restaura los pesos desde un archivo de checkpoint.

    Argumentos:
        checkpoint_path (str): Ruta absoluta o relativa al archivo .pth.
        device (torch.device): Dispositivo de cómputo de destino.

    Retorna:
        torch.nn.Module: Modelo con los pesos restaurados en modo evaluación.

    Excepciones:
        RuntimeError: Si la carga del checkpoint falla.
    """
    model = CNN_ResNet().to(device)
    success, _, metrics = load_checkpoint(checkpoint_path, model, device)
    if not success:
        raise RuntimeError(
            f"No se pudo cargar el checkpoint desde: {checkpoint_path}"
        )
    val_acc = metrics.get('val_acc', 'N/A')
    val_loss = metrics.get('val_loss', 'N/A')
    if isinstance(val_acc, float):
        print(f"Modelo cargado -> Val Acc: {val_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}")
    else:
        print(f"Modelo cargado. Métricas disponibles: {metrics}")
    return model


# =============================================================================
# Lógica de los modos de ejecución
# =============================================================================
def run_evaluate(args: argparse.Namespace) -> None:
    """
    Ejecuta la evaluación cuantitativa y cualitativa del modelo sobre el
    conjunto de prueba.

    Etapas:
      1. Configuración del entorno y reproducibilidad.
      2. Resolución de estadísticas de normalización.
      3. Construcción del DataLoader de prueba.
      4. Carga del modelo desde el checkpoint especificado.
      5. Cálculo de accuracy y F1-Score (macro).
      6. Visualización: curva de convergencia (si se dispone del historial),
         matriz de confusión y galería de errores de clasificación.

    Argumentos:
        args (argparse.Namespace): Argumentos parseados por el subcomando
            'evaluate'. Campos requeridos: checkpoint, data_path, seed.
    """
    # Al inicio de run_evaluate:
    logger = get_logger(__name__)
    logger.info("Iniciando evaluación. Checkpoint: %s", args.checkpoint)

    device, num_workers = setup_environment()
    g_cpu = set_seed(seed=args.seed)

    mean_val, std_val = resolve_dataset_statistics(args.data_path)

    _, _, test_loader, _, _, _ = prepare_dataloaders(
        ruta_base=args.data_path,
        mean_val=mean_val,
        std_val=std_val,
        batch_size_train=DEFAULT_BATCH_TRAIN,
        batch_size_eval=DEFAULT_BATCH_EVAL,
        num_workers=num_workers,
        generator=g_cpu,
    )

    model = load_model_from_checkpoint(args.checkpoint, device)

    # --- Métricas cuantitativas ---
    test_acc, test_f1 = evaluate_model(model, test_loader, device)
    diff = test_acc - BASELINE_ACC

    # Tras evaluate_model:
    logger.info(
        "Evaluación completada. Accuracy: %.4f | F1 (macro): %.4f | "
        "Diferencia respecto al baseline: %+.4f",
        test_acc, test_f1, diff
    )

    # --- Análisis cualitativo ---
    y_true, y_pred, _, images_test = get_predictions(
        model, test_loader, device
    )
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)

    plot_error_gallery(
        images_test, y_true, y_pred, CLASS_NAMES, num_images=10
    )


def run_ood(args: argparse.Namespace) -> None:
    """
    Ejecuta la evaluación de robustez ante datos sintéticos fuera de la
    distribución de entrenamiento (Out-of-Distribution, OOD).

    Para cada caso definido en OOD_TESTS, genera una imagen sintética de
    28x28 píxeles renderizando el carácter especificado y visualiza la
    distribución de probabilidades Softmax asignada por el modelo.

    Argumentos:
        args (argparse.Namespace): Argumentos parseados por el subcomando
            'ood'. Campos requeridos: checkpoint, data_path, seed.
    """
    device, _ = setup_environment()
    set_seed(seed=args.seed)

    mean_val, std_val = resolve_dataset_statistics(args.data_path)
    model = load_model_from_checkpoint(args.checkpoint, device)

    print("\n=== Evaluación de robustez ante datos sintéticos (OOD) ===")
    for label, char_str in OOD_TESTS:
        try:
            tensor_ood = create_image_from_text(char_str, mean_val, std_val)
            visualize_ood_confidence(
                model, tensor_ood, label, CLASS_NAMES, device
            )
        except Exception as e:
            print(f"No se pudo procesar la prueba para '{label}': {e}")


def run_train(args: argparse.Namespace) -> None:
    """
    Invoca el pipeline de entrenamiento completo definido en train.py,
    trasladando todos los argumentos parseados por el subcomando 'train'.

    Tras el entrenamiento, si se dispone de historial de métricas, se
    visualiza la dinámica de convergencia.

    Argumentos:
        args (argparse.Namespace): Argumentos parseados por el subcomando
            'train'.
    """
    result = run_training(
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

    if result.get('history') is not None:
        print("\nVisualizando dinámica de convergencia...")
        plot_training_history(result['history'])


# =============================================================================
# Construcción de la CLI
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Punto de entrada principal: Clasificación de caracteres '
            'Kuzushiji (KMNIST) con CNN_ResNet.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--version', action='store_true',
        help='Mostrar versiones de las librerías del entorno y salir.',
    )

    subparsers = parser.add_subparsers(dest='mode')

    # ------------------------------------------------------------------
    # Subcomando: train
    # ------------------------------------------------------------------
    p_train = subparsers.add_parser(
        'train',
        help='Ejecutar el pipeline de entrenamiento completo.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_train.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH,
                         dest='data_path',
                         help='Ruta al directorio raíz de los datos KMNIST.')
    p_train.add_argument('--models-dir', type=str, default=DEFAULT_MODELS_DIR,
                         dest='models_dir',
                         help='Directorio de salida para el checkpoint .pth.')
    p_train.add_argument('--checkpoint-name', type=str,
                         default=DEFAULT_CHECKPOINT_NAME,
                         dest='checkpoint_name',
                         help='Nombre del archivo de checkpoint.')
    p_train.add_argument('--max-epochs', type=int, default=DEFAULT_MAX_EPOCHS,
                         dest='max_epochs',
                         help='Número máximo de épocas.')
    p_train.add_argument('--lr', type=float, default=DEFAULT_LR,
                         help='Tasa de aprendizaje inicial (Adam).')
    p_train.add_argument('--l2', type=float, default=DEFAULT_L2,
                         help='Coeficiente de regularización L2.')
    p_train.add_argument('--batch-train', type=int, default=DEFAULT_BATCH_TRAIN,
                         dest='batch_train',
                         help='Tamaño del lote para entrenamiento.')
    p_train.add_argument('--batch-eval', type=int, default=DEFAULT_BATCH_EVAL,
                         dest='batch_eval',
                         help='Tamaño del lote para validación y test.')
    p_train.add_argument('--force-train', action='store_true',
                         dest='force_train',
                         help='Reentrenar ignorando checkpoints previos.')
    p_train.add_argument('--seed', type=int, default=DEFAULT_SEED,
                         help='Semilla global para reproducibilidad.')

    # ------------------------------------------------------------------
    # Subcomando: evaluate
    # ------------------------------------------------------------------
    p_eval = subparsers.add_parser(
        'evaluate',
        help='Evaluar un modelo guardado sobre el conjunto de prueba.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_eval.add_argument('--checkpoint', type=str, required=True,
                        help='Ruta al archivo .pth del modelo entrenado.')
    p_eval.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH,
                        dest='data_path',
                        help='Ruta al directorio raíz de los datos KMNIST.')
    p_eval.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Semilla global para reproducibilidad.')

    # ------------------------------------------------------------------
    # Subcomando: ood
    # ------------------------------------------------------------------
    p_ood = subparsers.add_parser(
        'ood',
        help='Evaluar robustez del modelo ante datos sintéticos (OOD).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_ood.add_argument('--checkpoint', type=str, required=True,
                       help='Ruta al archivo .pth del modelo entrenado.')
    p_ood.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH,
                       dest='data_path',
                       help='Ruta al directorio raíz de los datos KMNIST.')
    p_ood.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help='Semilla global para reproducibilidad.')

    return parser


# =============================================================================
# Punto de entrada
# =============================================================================
def main() -> None:
    """
    Función principal: analiza los argumentos de la línea de comandos y
    despacha la ejecución al modo correspondiente.
    """

    setup_logging()   # única llamada en todo el proyecto
    logger = get_logger(__name__)

    parser = _build_parser()
    args = parser.parse_args()

    if args.version:
        print_version_info()
        sys.exit(0)

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    logger.info("Modo de ejecución seleccionado: %s", args.mode)

    dispatch = {
        'train':    run_train,
        'evaluate': run_evaluate,
        'ood':      run_ood,
    }

    handler = dispatch.get(args.mode)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except Exception as exc:
        logger.critical(
            "Fallo irrecuperable en el modo '%s': %s",
            args.mode, exc, exc_info=True
        )

if __name__ == '__main__':
    main()
