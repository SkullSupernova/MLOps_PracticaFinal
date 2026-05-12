# =============================================================================
# logging_config.py
# Configuración centralizada del sistema de logging para el proyecto KMNIST.
#
# Proporciona un único punto de configuración para todos los módulos del
# proyecto, garantizando formato uniforme, rotación de archivos y separación
# entre salida por consola y persistencia en disco.
# =============================================================================

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes de configuración
# ---------------------------------------------------------------------------
LOG_DIR_NAME   = 'logs'
LOG_FILE_NAME  = 'kmnist.log'
LOG_MAX_BYTES  = 5 * 1024 * 1024   # 5 MB por archivo
LOG_BACKUP_CNT = 3                  # máximo 3 archivos de rotación
LOG_LEVEL_FILE = logging.DEBUG
LOG_LEVEL_CON  = logging.INFO

# Nombre del logger raíz del proyecto: todos los subloggers lo heredan.
PROJECT_LOGGER_NAME = 'kmnist'

# Formato detallado para archivo (incluye módulo y línea para trazabilidad)
FMT_FILE = (
    '%(asctime)s | %(levelname)-8s | %(name)s | '
    '%(module)s:%(lineno)d | %(message)s'
)

# Formato compacto para consola
FMT_CONSOLE = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'

DATE_FMT = '%Y-%m-%d %H:%M:%S'


def _resolve_log_dir() -> Path:
    """
    Determina la ruta absoluta del directorio de logs.

    Busca el directorio en la raíz del proyecto (un nivel por encima de src/).
    Si no puede resolverlo, crea el directorio logs/ junto al propio módulo
    como mecanismo de contingencia.

    Retorna:
        Path: Ruta absoluta al directorio de logs, creado si no existía.
    """
    module_dir  = Path(__file__).parent.resolve()
    project_root = module_dir.parent
    log_dir = project_root / LOG_DIR_NAME

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Contingencia: logs junto al módulo
        log_dir = module_dir / LOG_DIR_NAME
        log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir


def setup_logging(
    level_file:    int = LOG_LEVEL_FILE,
    level_console: int = LOG_LEVEL_CON,
    log_dir:       Path | None = None,
) -> logging.Logger:
    """
    Configura el logger raíz del proyecto con dos handlers: consola y archivo.

    Debe invocarse una única vez, en el punto de entrada de la aplicación
    (main.py o el script que inicie la ejecución). Los módulos secundarios
    obtienen su logger mediante get_logger(), que hereda esta configuración.

    Argumentos:
        level_file (int): Nivel mínimo de los mensajes escritos en archivo.
            Por defecto: logging.DEBUG.
        level_console (int): Nivel mínimo de los mensajes emitidos por consola.
            Por defecto: logging.INFO.
        log_dir (Path | None): Directorio donde se crea el archivo de log.
            Si es None, se determina automáticamente mediante _resolve_log_dir().

    Retorna:
        logging.Logger: Logger raíz del proyecto completamente configurado.
    """
    logger = logging.getLogger(PROJECT_LOGGER_NAME)

    # Idempotencia: evitar duplicación de handlers si se llama varias veces.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # el nivel más bajo; los handlers filtran
    logger.propagate = False

    fmt_file    = logging.Formatter(FMT_FILE,    datefmt=DATE_FMT)
    fmt_console = logging.Formatter(FMT_CONSOLE, datefmt=DATE_FMT)

    # --- Handler de consola ---
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level_console)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)

    # --- Handler de archivo con rotación ---
    resolved_log_dir = log_dir if log_dir is not None else _resolve_log_dir()
    log_path = resolved_log_dir / LOG_FILE_NAME

    fh = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_CNT,
        encoding='utf-8',
    )
    fh.setLevel(level_file)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)

    logger.info("Sistema de logging inicializado. Archivo: %s", log_path)
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger jerárquico subordinado al logger raíz del proyecto.

    Todos los loggers obtenidos mediante esta función heredan los handlers
    y el nivel configurado por setup_logging(), sin necesidad de configuración
    adicional en cada módulo.

    Argumentos:
        name (str): Identificador del sublogger. Convenio recomendado:
            usar el nombre del módulo, que puede obtenerse como __name__
            desde cualquier módulo Python.

    Retorna:
        logging.Logger: Logger configurado y listo para su uso.

    Ejemplo:
        logger = get_logger(__name__)
        logger.info("Módulo inicializado.")
    """
    return logging.getLogger(f'{PROJECT_LOGGER_NAME}.{name}')