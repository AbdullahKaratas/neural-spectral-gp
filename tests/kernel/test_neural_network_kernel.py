import pytest
import torch

from nsgp.kernel.neural_network_kernel import NeuralNetworkKernel


class TestNeuralNetworkKernel:
    """Unit tests for the Williams (1996) neural network kernel."""

    @pytest.fixture
    def kernel(self):
        torch.set_default_dtype(torch.float64)
        k = NeuralNetworkKernel(aug_dim=3)  # d = 2
        with torch.no_grad():
            k.variance = torch.tensor([0.5, 2.0, 1.0])
        return k

    def test_matches_network_definition(self, kernel):
        """Eq. (11) is the closed form of the covariance of an infinite-width
        single-layer erf network. Validate against that definition directly via
        Monte Carlo: k(x, x') = E_u[ h(x,u) h(x',u) ], u sim N(0, Sigma).
        """
        torch.manual_seed(0)
        x = torch.randn(3, 2)
        K = kernel(x, x).to_dense()

        x_aug = torch.cat([torch.ones(x.shape[0], 1), x], dim=-1)
        std = kernel.variance.detach().sqrt()
        n_samples = 1000000
        u = torch.randn(n_samples, 3) * std
        h = torch.erf(x_aug @ u.T)
        K_mc = (h @ h.T) / n_samples

        assert torch.allclose(K, K_mc, atol=3e-3)

    def test_diag_matches_full_diagonal(self, kernel):
        x = torch.randn(3, 2)
        full = kernel(x, x).to_dense().diagonal()
        diag = kernel(x, x, diag=True)
        assert torch.allclose(diag, full, atol=1e-10)

    def test_gradient_flows(self, kernel):
        x = torch.randn(3, 2)
        loss = kernel(x, x).to_dense().sum()
        loss.backward()
        assert kernel.raw_variance.grad is not None
        assert torch.isfinite(kernel.raw_variance.grad).all()

    def test_dim_mismatch_raises(self, kernel):
        # aug_dim=3 expects d=2; feeding d=4 must raise on evaluation
        with pytest.raises(ValueError):
            kernel(torch.randn(3, 4), torch.randn(3, 4)).to_dense()
