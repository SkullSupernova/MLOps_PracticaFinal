# 1. Librerías estándar de Python
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# 2. Librerías de terceros
import pandas as pd
import pytest
import torch

# 3. Importaciones locales del proyecto
from src.utils import (
    get_project_root,
    load_config,
    load_data,
    set_seed,
    setup_environment
)

# Lógica de script (siempre después de los imports)
print(f"Pytest version: {pytest.__version__}")


@pytest.fixture(scope="module")
def mock_project_root():
    """Proporciona una ruta raíz simulada para evitar dependencias del sistema de archivos real."""
    return Path("/mock/project/root")

# =====================================================================
# Fixtures
# =====================================================================
@pytest.fixture(scope="session")
def mock_dataset() -> pd.DataFrame:
    """Proporciona un DataFrame simulado en memoria para las pruebas."""
    return pd.DataFrame({
        "risk": [1, 0, 1, 0],
        "age": [35, 42, 28, 55],
        "income": [50000, 80000, 30000, 120000]
    })

# =====================================================================
# Tests refactorizados con simulación (Mocking)
# =====================================================================
@patch("pandas.read_csv")
def test_carga_dataset(mock_read_csv, mock_dataset):
    """
    Verifica que la invocación directa a pandas.read_csv procese el archivo 
    y retorne un DataFrame no vacío.
    """
    mock_read_csv.return_value = mock_dataset
    
    df = pd.read_csv("ruta_simulada.csv")
    
    mock_read_csv.assert_called_once_with("ruta_simulada.csv")
    assert len(df) > 0, "El dataset simulado no debe estar vacío."

@patch("src.utils.get_project_root")
@patch("pandas.read_csv")
def test_carga_dataset_2(mock_read_csv, mock_get_root, mock_dataset):
    """
    Verifica que la función personalizada load_data procese correctamente 
    la carga y retorne datos válidos sin depender del sistema de archivos.
    """
    mock_get_root.return_value = Path("/mock/root")
    mock_read_csv.return_value = mock_dataset
    
    df = load_data("ruta_simulada.csv")
    
    assert len(df) > 0, "El dataset cargado mediante load_data no debe estar vacío."

@patch("src.utils.get_project_root")
@patch("pandas.read_csv")
def test_dataset_etiqueta(mock_read_csv, mock_get_root, mock_dataset):
    """
    Verifica que el DataFrame devuelto contenga la columna objetivo requerida ('risk').
    """
    mock_get_root.return_value = Path("/mock/root")
    mock_read_csv.return_value = mock_dataset
    
    df = load_data("ruta_simulada.csv")
    
    assert "risk" in df.columns, "La columna objetivo 'risk' no se encuentra en el esquema del dataset."

# =====================================================================
# Tests para get_project_root
# =====================================================================
def test_get_project_root_returns_path():
    result = get_project_root()
    assert isinstance(result, Path), "La función debe retornar un objeto de tipo pathlib.Path."

# =====================================================================
# Tests para load_data
# =====================================================================
@patch("src.utils.get_project_root")
@patch("pandas.read_csv")
def test_load_data_success(mock_read_csv, mock_get_root, mock_project_root):
    mock_get_root.return_value = mock_project_root
    mock_df = pd.DataFrame({"risk": [1, 0, 1], "age": [25, 40, 35]})
    mock_read_csv.return_value = mock_df

    relative_path = "data/dummy_dataset.csv"
    result = load_data(relative_path)

    expected_full_path = mock_project_root / relative_path
    mock_read_csv.assert_called_once_with(expected_full_path)
    assert len(result) == 3, "El DataFrame retornado no contiene el número de filas esperado."
    assert "risk" in result.columns, "La columna 'risk' no se encuentra en el DataFrame cargado."

@patch("src.utils.get_project_root")
@patch("pandas.read_csv")
def test_load_data_file_not_found(mock_read_csv, mock_get_root, mock_project_root):
    mock_get_root.return_value = mock_project_root
    mock_read_csv.side_effect = FileNotFoundError("Archivo no encontrado")

    with pytest.raises(FileNotFoundError):
        load_data("ruta/inexistente.csv")

# =====================================================================
# Tests para load_config
# =====================================================================
@patch("src.utils.get_project_root")
@patch("builtins.open", new_callable=mock_open, read_data="batch_size: 128\nlearning_rate: 0.001")
@patch("yaml.safe_load")
def test_load_config_success(mock_yaml_load, mock_file, mock_get_root, mock_project_root):
    mock_get_root.return_value = mock_project_root
    mock_yaml_load.return_value = {"batch_size": 128, "learning_rate": 0.001}

    config_name = "config.yaml"
    result = load_config(config_name)

    expected_full_path = mock_project_root / 'config' / config_name
    mock_file.assert_called_once_with(expected_full_path, 'r')
    assert isinstance(result, dict), "La configuración cargada debe ser un diccionario."
    assert result.get("batch_size") == 128, "Los valores leídos no coinciden con la configuración esperada."

# =====================================================================
# Tests para setup_environment
# =====================================================================
@patch("torch.cuda.is_available", return_value=False)
@patch("os.cpu_count", return_value=8)
@patch("os.name", "posix")
def test_setup_environment_cpu_posix(mock_is_available, mock_cpu_count):
    device, num_workers = setup_environment()
    
    assert device.type == "cpu", "El dispositivo asignado debe ser CPU si CUDA no está disponible."
    assert num_workers == 2, "En sistemas Unix/Linux, num_workers debe ser 2."

@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.get_device_name", return_value="Mock GPU")
@patch("torch.cuda.get_device_properties")
@patch("os.name", "nt")
def test_setup_environment_gpu_windows(mock_properties, mock_name, mock_is_available):
    mock_prop = MagicMock()
    mock_prop.total_memory = 8 * 1024**3
    mock_properties.return_value = mock_prop

    device, num_workers = setup_environment()
    
    assert device.type == "cuda", "El dispositivo asignado debe ser CUDA si está disponible."
    assert num_workers == 0, "En sistemas Windows, num_workers debe ser 0 para evitar bloqueos."

# =====================================================================
# Tests para set_seed
# =====================================================================
def test_set_seed_determinism():
    seed_value = 42
    generator = set_seed(seed_value)
    
    assert isinstance(generator, torch.Generator), "Debe retornar una instancia de torch.Generator."
    assert torch.initial_seed() == seed_value, "La semilla global de PyTorch no se configuró correctamente."
    assert os.environ.get('PYTHONHASHSEED') == str(seed_value), "La variable de entorno PYTHONHASHSEED no se estableció."