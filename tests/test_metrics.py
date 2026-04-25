import math

import gpytorch
import pytest
import torch

from nsgp.metrics import (
    kl_posterior,
    log_pred_density_true_function,
    marginal_log_likelihood,
    negative_log_predictive_density,
    noise_variance,
    oracle_posterior,
)
from nsgp.models.dkl_gp import DKLGP
from nsgp.models.neural_gsm_gp import NeuralGSMGP
from nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork
from nsgp.models.standard_gp import StandardGP


def rbf(x1: torch.Tensor, x2: torch.Tensor, lengthscale: float = 1.0) -> torch.Tensor:
    """Plain RBF kernel for use as ground truth in tests."""
    sq = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2).sum(-1)
    return torch.exp(-0.5 * sq / lengthscale ** 2)


@pytest.fixture
def data():
    torch.manual_seed(0)
    X_train = torch.linspace(-3, 3, 20).unsqueeze(-1)
    y_train = torch.sin(X_train).squeeze() + 0.1 * torch.randn(20)
    X_test = torch.linspace(-2, 2, 10).unsqueeze(-1)
    y_test = torch.sin(X_test).squeeze()
    return X_train, y_train, X_test, y_test


class TestOraclePosterior:
    """Sanity checks on the Bayes-optimal GP posterior under a known kernel."""

    def test_returns_mvn_with_correct_shape(self, data):
        X_train, y_train, X_test, _ = data
        post = oracle_posterior(rbf, X_train, y_train, X_test, noise_var=0.01)
        assert isinstance(post, gpytorch.distributions.MultivariateNormal)
        assert post.mean.shape == (10,)
        assert post.covariance_matrix.shape == (10, 10)

    def test_kl_to_self_is_zero(self, data):
        X_train, y_train, X_test, _ = data
        p = oracle_posterior(rbf, X_train, y_train, X_test, noise_var=0.01)
        # Re-build to get a separate object; KL between identical MVNs is 0.
        q = oracle_posterior(rbf, X_train, y_train, X_test, noise_var=0.01)
        kl = kl_posterior(p, q)
        assert torch.isfinite(kl)
        assert abs(kl.item()) < 1e-4


class TestKLPosterior:
    def test_kl_nonnegative(self, data):
        X_train, y_train, X_test, _ = data
        oracle_short = oracle_posterior(rbf, X_train, y_train, X_test, noise_var=0.01)
        # Different lengthscale -> different posterior.
        oracle_long = oracle_posterior(
            lambda a, b: rbf(a, b, lengthscale=2.0),
            X_train, y_train, X_test, noise_var=0.01,
        )
        kl = kl_posterior(oracle_short, oracle_long)
        assert torch.isfinite(kl)
        # KL is non-negative up to numerical noise.
        assert kl.item() > -1e-4


class TestLogPredDensityTrueFunction:
    def test_returns_finite_scalar(self, data):
        X_train, y_train, X_test, y_test = data
        oracle = oracle_posterior(rbf, X_train, y_train, X_test, noise_var=0.01)
        lpd = log_pred_density_true_function(oracle, y_test)
        assert lpd.shape == torch.Size([])
        assert torch.isfinite(lpd)


class TestNLPDFromGPyTorch:
    def test_finite_on_fitted_model(self, data):
        X_train, y_train, X_test, y_test = data
        m = StandardGP()
        m.fit(X_train, y_train, epochs=5, verbose=False)
        pred = m._full_pred_dist(X_test)
        nlpd = negative_log_predictive_density(pred, y_test)
        assert torch.isfinite(nlpd)


class TestPerModelMLLAndNoise:
    """Verify the per-model adapters work for all four model classes."""

    @pytest.fixture
    def fitted_models(self, data):
        X_train, y_train, _, _ = data
        std = StandardGP()
        std.fit(X_train, y_train, epochs=5, verbose=False)
        ngsm = NeuralGSMGP(input_dim=1, n_components=1, hidden_dims=[8])
        ngsm.fit(X_train, y_train, epochs=5, verbose=False)
        dkl = DKLGP(input_dim=1, output_dim=2, hidden_dims=[8, 8])
        dkl.fit(X_train, y_train, epochs=5, verbose=False)
        fsdn = FactorizedSpectralDensityNetwork(
            input_dim=1, hidden_dims=[16], rank=4, n_features=20, omega_max=5.0
        )
        fsdn.fit(X_train, y_train, epochs=5, verbose=False)
        return {"std": std, "ngsm": ngsm, "dkl": dkl, "fsdn": fsdn}

    def test_mll_finite_for_all_models(self, fitted_models, data):
        X_train, y_train, _, _ = data
        for name, m in fitted_models.items():
            mll = marginal_log_likelihood(m, X_train, y_train)
            assert math.isfinite(mll), f"MLL not finite for {name}"

    def test_noise_var_positive_for_all_models(self, fitted_models):
        for name, m in fitted_models.items():
            nv = noise_variance(m)
            assert nv > 0, f"noise variance not positive for {name}: {nv}"
            assert math.isfinite(nv)

    def test_unsupported_model_type_raises(self):
        with pytest.raises(TypeError):
            marginal_log_likelihood(object(), torch.zeros(2, 1), torch.zeros(2))
        with pytest.raises(TypeError):
            noise_variance(object())
