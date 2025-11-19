import pytest
import torch
import numpy as np
from src.nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork


class TestFactorizedSpectralDensityNetwork:
    """Unit tests for FactorizedSpectralDensityNetwork low-rank methods."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return FactorizedSpectralDensityNetwork(
            input_dim=1,
            hidden_dims=[32],
            rank=5,
            n_features=10,
            omega_max=4.0,
            enforce_symmetry=True
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        X = torch.linspace(-3, 3, 20).unsqueeze(-1)
        y = torch.sin(X).squeeze() + 0.1 * torch.randn(20)
        return X, y

    def test_safe_cholesky_basic(self, model):
        """Test _safe_cholesky with a valid positive definite matrix."""
        A = torch.tensor([[2.0, 0.5], [0.5, 2.0]])
        L = model._safe_cholesky(A, jitter=1e-6, max_attempts=4)

        # Check shape
        assert L.shape == A.shape

        # Check that L @ L.T approximates A
        reconstructed = L @ L.T
        assert torch.allclose(reconstructed, A, atol=1e-4)

    def test_safe_cholesky_near_singular(self, model):
        """Test _safe_cholesky with nearly singular matrix (should add jitter)."""
        A = torch.tensor([[1.0, 0.9999], [0.9999, 1.0]])

        # Should succeed with adaptive jittering
        L = model._safe_cholesky(A, jitter=1e-6, max_attempts=4)
        assert L.shape == A.shape

    def test_safe_cholesky_failure(self, model):
        """Test _safe_cholesky raises error when matrix is not PD."""
        A = torch.tensor([[1.0, 2.0], [2.0, 1.0]])  # Negative eigenvalue

        with pytest.raises(RuntimeError, match="Cholesky failed"):
            model._safe_cholesky(A, jitter=1e-12, max_attempts=2)

    def test_compute_spectral_density_matrix(self, model):
        """Test _compute_spectral_density_matrix returns correct shape and symmetry."""
        omega_grid = torch.linspace(0, 4.0, 10).unsqueeze(-1)
        S = model._compute_spectral_density_matrix(omega_grid)

        # Should be (M, M) where M is number of frequencies
        assert S.shape == (10, 10)

        # S[i,j] = s(omega_i, omega_j) should equal S[j,i] (symmetric)
        assert torch.allclose(S, S.T, atol=1e-5)

    def test_compute_lowrank_features_shape(self, model, sample_data):
        """Test compute_lowrank_features returns correct shape."""
        X, _ = sample_data
        omega_grid = torch.linspace(0, 4.0, 10).unsqueeze(-1)

        L = model.compute_lowrank_features(X, omega_grid)

        # Should be (n, num_freqs)
        n = X.shape[0]
        num_freqs = omega_grid.shape[0]
        assert L.shape == (n, num_freqs)

    def test_compute_lowrank_features_zero_frequency(self, model):
        """Test zero frequency correction is applied."""
        X = torch.randn(10, 1)
        # Grid with zero frequency
        omega_grid = torch.tensor([[0.0], [1.0], [2.0]])

        L = model.compute_lowrank_features(X, omega_grid)

        # Should not raise error and return valid features
        assert not torch.isnan(L).any()
        assert not torch.isinf(L).any()

    def test_log_marginal_likelihood_shape(self, model, sample_data):
        """Test log_marginal_likelihood returns scalar."""
        X, y = sample_data
        omega_grid = torch.linspace(0, 4.0, 10).unsqueeze(-1)
        L = model.compute_lowrank_features(X, omega_grid)

        sigma2 = torch.tensor(0.01)
        nll = model.log_marginal_likelihood(L, y, sigma2)

        # Should be scalar
        assert nll.shape == torch.Size([])
        assert nll.ndim == 0

    def test_log_marginal_likelihood_rank_warning(self, model):
        """Test log_marginal_likelihood warns when r > n."""
        # r=5, but n=3 < r
        L = torch.randn(3, 5)
        y = torch.randn(3)
        sigma2 = torch.tensor(0.01)

        with pytest.warns(UserWarning, match="Rank r=5 exceeds number of data points"):
            model.log_marginal_likelihood(L, y, sigma2)

    def test_enforce_symmetry_flag(self):
        """Test enforce_symmetry parameter affects spectral density."""
        model_sym = FactorizedSpectralDensityNetwork(
            input_dim=1, hidden_dims=[32], rank=5,
            n_features=10, enforce_symmetry=True
        )
        model_no_sym = FactorizedSpectralDensityNetwork(
            input_dim=1, hidden_dims=[32], rank=5,
            n_features=10, enforce_symmetry=False
        )

        # Copy weights to make them identical
        model_no_sym.load_state_dict(model_sym.state_dict())

        omega_grid = torch.linspace(0, 4.0, 10).unsqueeze(-1)

        # With symmetry enforced
        f_sym = model_sym.compute_features(omega_grid)
        f_neg_sym = model_sym.compute_features(-omega_grid)

        # f(omega) should equal f(-omega) when enforce_symmetry=True
        assert torch.allclose(f_sym, f_neg_sym, atol=1e-5)

        # Without symmetry, they might differ (unless network is exactly symmetric)
        f_no_sym = model_no_sym.compute_features(omega_grid)
        f_neg_no_sym = model_no_sym.compute_features(-omega_grid)

        # They should differ (unless by chance the network is symmetric)
        # We just check that the code runs without error
        assert f_no_sym.shape == f_neg_no_sym.shape

    def test_gradient_flow(self, model, sample_data):
        """Test that gradients flow through low-rank training."""
        X, y = sample_data
        omega_grid = torch.linspace(0, 4.0, 10).unsqueeze(-1)

        # Enable gradient tracking
        for param in model.parameters():
            param.requires_grad = True

        # Forward pass
        L = model.compute_lowrank_features(X, omega_grid)
        sigma2 = torch.tensor(0.01, requires_grad=False)
        loss = model.log_marginal_likelihood(L, y, sigma2)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
