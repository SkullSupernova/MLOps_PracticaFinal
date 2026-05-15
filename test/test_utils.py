# =============================================================================
# test_utils.py — Tests de utilidades de datos, semillas y entorno.
# =============================================================================

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.utils import (
    EarlyStopping,
    ModelCheckpoint,
    calculate_metrics,
    get_project_root,
    set_seed,
    setup_environment,
)


class TestGetProjectRoot:

    def test_returns_path(self):
        assert isinstance(get_project_root(), Path)

    def test_is_absolute(self):
        assert get_project_root().is_absolute()

    def test_contains_src(self):
        """La raíz debe contener el directorio src/."""
        assert (get_project_root() / 'src').exists()


class TestSetSeed:

    def test_returns_generator(self):
        g = set_seed(42)
        assert isinstance(g, torch.Generator)

    def test_torch_seed_matches(self):
        set_seed(42)
        assert torch.initial_seed() == 42

    def test_pythonhashseed_env_var(self):
        set_seed(123)
        assert os.environ.get('PYTHONHASHSEED') == '123'

    def test_deterministic_numpy(self):
        import numpy as np
        set_seed(0)
        a = np.random.rand(10)
        set_seed(0)
        b = np.random.rand(10)
        assert (a == b).all()


class TestSetupEnvironment:

    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.name", "posix")
    def test_cpu_unix(self, _):
        device, num_workers = setup_environment()
        assert device.type == "cpu"
        assert num_workers == 2

    @patch("torch.cuda.is_available", return_value=False)
    @patch("os.name", "nt")
    def test_cpu_windows(self, _):
        device, num_workers = setup_environment()
        assert device.type == "cpu"
        assert num_workers == 0


class TestCalculateMetrics:

    def test_perfect_prediction(self):
        y = [0, 1, 2, 3]
        m = calculate_metrics(y, y)
        assert m['accuracy'] == pytest.approx(1.0)
        assert m['f1_macro'] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        m = calculate_metrics(y_true, y_pred)
        assert m['accuracy'] == pytest.approx(0.0)


class TestEarlyStopping:

    def test_triggers_after_patience(self):
        es = EarlyStopping(patience=3)
        # Se requieren 4 evaluaciones: 1 basal + 3 sin mejora paramétrica
        for _ in range(4):
            es(1.0)
        assert es.early_stop

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        es(0.9)   # mejora
        assert es.counter == 0
        assert not es.early_stop


class TestModelCheckpoint:

    def test_saves_best_state(self):
        from src.model import CNN_ResNet
        model = CNN_ResNet()
        ckpt  = ModelCheckpoint()
        saved = ckpt(model, val_loss=0.5)
        assert saved
        assert ckpt.best_model_state is not None

    def test_does_not_save_worse(self):
        from src.model import CNN_ResNet
        model = CNN_ResNet()
        ckpt  = ModelCheckpoint()
        ckpt(model, val_loss=0.5)
        saved = ckpt(model, val_loss=0.8)
        assert not saved
