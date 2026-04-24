import pytest
import torch

from src.nsgp.kernel.deep_kernel import DeepKernel, FeatureExtractor


class TestFeatureExtractor:
    """Unit tests for FeatureExtractor MLP."""

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


class TestDeepKernel:
    """Unit tests for DeepKernel."""

    @pytest.fixture
    def kernel(self):
        torch.manual_seed(0)
        return DeepKernel(input_dim=1, feature_dim=2, hidden_dims=[8])

    @pytest.fixture
    def x(self):
        torch.manual_seed(1)
        return torch.linspace(-3, 3, 12).unsqueeze(-1)

    def test_full_kernel_shape(self, kernel, x):
        K = kernel(x, x).to_dense()
        assert K.shape == (12, 12)

    def test_cross_kernel_shape(self, kernel, x):
        x2 = torch.linspace(-1, 1, 5).unsqueeze(-1)
        K = kernel(x, x2).to_dense()
        assert K.shape == (12, 5)

    def test_diag_matches_full_diagonal(self, kernel, x):
        K_full = kernel(x, x).to_dense()
        K_diag = kernel(x, x, diag=True)
        assert K_diag.shape == (12,)
        assert torch.allclose(K_diag, torch.diagonal(K_full), atol=1e-5)

    def test_symmetry(self, kernel, x):
        K = kernel(x, x).to_dense()
        assert torch.allclose(K, K.transpose(-1, -2), atol=1e-5)

    def test_positive_semidefinite(self, kernel, x):
        K = kernel(x, x).to_dense()
        # Symmetrize to remove numerical asymmetry, then check eigenvalues.
        K_sym = 0.5 * (K + K.transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(K_sym)
        assert eigvals.min() > -1e-5, f"min eigenvalue {eigvals.min().item()} < 0"

    def test_gradient_flows_through_feature_extractor(self, kernel, x):
        K = kernel(x, x).to_dense()
        K.sum().backward()
        for name, p in kernel.feature_extractor.named_parameters():
            assert p.grad is not None, f"no gradient for feature_extractor.{name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    def test_x1_eq_x2_path_equivalent_to_general(self, kernel, x):
        """The x1_eq_x2 shortcut must give the same result as recomputing phi(x2)."""
        K_shortcut = kernel(x, x).to_dense()
        # Force the general path by passing a distinct (but equal-valued) tensor.
        x_copy = x.clone() + 0.0
        K_general = kernel(x, x_copy).to_dense()
        assert torch.allclose(K_shortcut, K_general, atol=1e-5)
