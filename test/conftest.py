# =============================================================================
# conftest.py — Fixtures compartidas para toda la suite de tests.
# =============================================================================


import pytest
import torch
import torchvision.transforms as transforms


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model(device):
    """Instancia CNN_ResNet no entrenada para tests de arquitectura."""
    from src.model import CNN_ResNet
    m = CNN_ResNet().to(device)
    m.eval()
    return m


@pytest.fixture(scope="session")
def dummy_batch(device):
    """Lote sintético compatible con el formato KMNIST."""
    images = torch.randn(4, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (4,)).to(device)
    return images, labels


@pytest.fixture(scope="session")
def transform_eval():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1918,), (0.3483,)),
    ])
