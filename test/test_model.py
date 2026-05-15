# =============================================================================
# test_model.py — Tests de arquitectura y forward pass.
# =============================================================================

import torch
import pytest
from src.model import CNN_ResNet, ResidualBlock


class TestResidualBlock:

    def test_output_shape_same_channels(self):
        block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
        x   = torch.randn(2, 16, 28, 28)
        out = block(x)
        assert out.shape == (2, 16, 28, 28), (
            "El shape de salida con stride=1 y mismos canales debe ser idéntico al de entrada."
        )

    def test_output_shape_downsampled(self):
        block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
        x   = torch.randn(2, 16, 28, 28)
        out = block(x)
        assert out.shape == (2, 32, 14, 14), (
            "Con stride=2 la resolución espacial debe reducirse a la mitad."
        )

    def test_shortcut_is_identity_when_dims_match(self):
        block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
        # La shortcut debe ser un Sequential vacío
        assert len(list(block.shortcut.parameters())) == 0


class TestCNNResNet:

    def test_output_shape(self, model, dummy_batch):
        images, _ = dummy_batch
        logits = model(images)
        assert logits.shape == (4, 10), (
            "La salida del modelo debe tener shape (batch, 10)."
        )

    def test_num_classes(self, model):
        assert model.fc.out_features == 10

    def test_parameter_count(self, model):
        n = sum(p.numel() for p in model.parameters())
        # La arquitectura tiene aproximadamente 110.000 parámetros
        assert 50_000 < n < 500_000, (
            f"Número de parámetros fuera del rango esperado: {n}"
        )

    def test_output_is_finite(self, model, dummy_batch):
        images, _ = dummy_batch
        logits = model(images)
        assert torch.isfinite(logits).all(), (
            "La salida del modelo contiene NaN o infinitos."
        )

    def test_gradient_flows(self):
        """Verifica que los gradientes se propagan correctamente en entrenamiento."""
        m = CNN_ResNet()
        m.train()
        x       = torch.randn(2, 1, 28, 28, requires_grad=False)
        targets = torch.randint(0, 10, (2,))
        loss    = torch.nn.CrossEntropyLoss()(m(x), targets)
        loss.backward()
        for name, param in m.named_parameters():
            assert param.grad is not None, (
                f"El parámetro '{name}' no recibió gradiente."
            )

    def test_deterministic_output_under_seed(self):
        """Con semilla fija, la misma entrada produce la misma salida."""
        torch.manual_seed(0)
        m1 = CNN_ResNet().eval()
        torch.manual_seed(0)
        m2 = CNN_ResNet().eval()
        x = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            out1 = m1(x)
            out2 = m2(x)
        assert torch.allclose(out1, out2), (
            "Bajo la misma semilla, dos instancias idénticas deben producir la misma salida."
        )