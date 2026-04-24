import pytest
import torch

from src.nsgp.kernel.deep_kernel import FeatureExtractor


class TestFeatureExtractor:
    """Unit tests for the DKL feature extractor phi."""

    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        return FeatureExtractor(input_dim=1, output_dim=3, hidden_dims=[8, 8])

    def test_output_shape(self, mlp):
        x = torch.linspace(-2, 2, 16).unsqueeze(-1)
        phi = mlp(x)
        assert phi.shape == (16, 3)

    def test_batch_shape_preserved(self, mlp):
        x = torch.randn(4, 10, 1)
        phi = mlp(x)
        assert phi.shape == (4, 10, 3)

    def test_gradient_flows(self, mlp):
        x = torch.randn(5, 1)
        phi = mlp(x)
        loss = phi.pow(2).sum()
        loss.backward()
        for name, p in mlp.named_parameters():
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"

    def test_empty_hidden_dims_gives_linear_map(self):
        torch.manual_seed(0)
        mlp = FeatureExtractor(input_dim=2, output_dim=4, hidden_dims=[])
        x = torch.randn(7, 2)
        phi = mlp(x)
        assert phi.shape == (7, 4)

    def test_zero_bias_init(self):
        torch.manual_seed(0)
        mlp = FeatureExtractor(input_dim=2, output_dim=3, hidden_dims=[8])
        for m in mlp.net.modules():
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                assert torch.all(m.bias == 0.0)
