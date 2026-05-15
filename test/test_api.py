# =============================================================================
# test_api.py — Tests de los endpoints FastAPI.
# =============================================================================

import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.api import CLASS_NAMES, _state, app
from src.model import CNN_ResNet


@pytest.fixture(autouse=True)
def mock_model_state():
    """Inyecta un modelo mock en el estado global para todos los tests de la API."""
    model = CNN_ResNet().eval()
    original = _state.copy()
    _state.update({
        'model':   model,
        'device':  torch.device('cpu'),
        'metrics': {'val_acc': 0.986, 'val_loss': 0.045},
        'transform': __import__(
            'torchvision.transforms', fromlist=['Compose']
        ).Compose([
            __import__('torchvision.transforms', fromlist=['Grayscale']).Grayscale(1),
            __import__('torchvision.transforms', fromlist=['Resize']).Resize((28, 28)),
            __import__('torchvision.transforms', fromlist=['ToTensor']).ToTensor(),
            __import__('torchvision.transforms', fromlist=['Normalize']).Normalize(
                (0.1918,), (0.3483,)
            ),
        ]),
    })
    yield
    _state.update(original)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dummy_png_bytes():
    """Genera una imagen PNG 28x28 en escala de grises en memoria."""
    img    = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode='L')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


class TestHealthEndpoint:

    def test_status_ok_when_model_loaded(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True
        assert r.json()["status"] == "ok"

    def test_status_degraded_when_no_model(self, client):
        _state['model'] = None
        r = client.get("/health")
        assert r.json()["status"] == "degraded"
        assert r.json()["model_loaded"] is False


class TestModelInfoEndpoint:

    def test_returns_architecture_info(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        data = r.json()
        assert data["architecture"] == "CNN_ResNet"
        assert data["num_classes"] == 10
        assert len(data["class_names"]) == 10

    def test_unavailable_when_no_model(self, client):
        _state['model'] = None
        r = client.get("/model/info")
        assert r.status_code == 503


class TestPredictEndpoint:

    def test_predict_returns_valid_class(self, client, dummy_png_bytes):
        r = client.post(
            "/predict",
            files={"file": ("test.png", dummy_png_bytes, "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["predicted_class"] in CLASS_NAMES
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["probabilities"]) == 10
        assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-5

    def test_predict_rejects_unsupported_type(self, client):
        r = client.post(
            "/predict",
            files={"file": ("file.txt", b"not an image", "text/plain")},
        )
        assert r.status_code == 415

    def test_predict_unavailable_when_no_model(self, client, dummy_png_bytes):
        _state['model'] = None
        r = client.post(
            "/predict",
            files={"file": ("test.png", dummy_png_bytes, "image/png")},
        )
        assert r.status_code == 503
