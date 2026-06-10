import math

import pytest
import torch

from nsgp.metrics import (
    kl_posterior,
    marginal_log_likelihood,
    noise_variance,
    oracle_posterior,
)
from nsgp.models.dkl_gp import DKLGP
from nsgp.models.neural_gsm_gp import NeuralGSMGP
from nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork
from nsgp.models.standard_gp import StandardGP


def rbf(x1: torch.Tensor, x2: torch.Tensor, lengthscale: float = 1.0) -> torch.Tensor:
    sq = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2).sum(-1)
    return torch.exp(-0.5 * sq / lengthscale**2)


@pytest.fixture
def data():
    torch.manual_seed(0)
    X_train = torch.linspace(-3, 3, 20).unsqueeze(-1)
    y_train = torch.sin(X_train).squeeze() + 0.1 * torch.randn(20)
    X_test = torch.linspace(-2, 2, 10).unsqueeze(-1)
    return X_train, y_train, X_test


def test_kl_nonnegative(data):
    X_train, y_train, X_test = data
    p = oracle_posterior(rbf, X_train, y_train, X_test, noise_var=0.01)
    q = oracle_posterior(
        lambda a, b: rbf(a, b, lengthscale=2.0),
        X_train,
        y_train,
        X_test,
        noise_var=0.01,
    )
    kl = kl_posterior(p, q)
    assert torch.isfinite(kl)
    assert kl.item() > -1e-4


class TestPerModelMLLAndNoise:
    """Verify the per-model adapters work for all four model classes."""

    @pytest.fixture
    def fitted_models(self, data):
        X_train, y_train, _ = data
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

    def test_mll_finite_for_all_models(self, fitted_models):
        for name, m in fitted_models.items():
            assert math.isfinite(
                marginal_log_likelihood(m)
            ), f"MLL not finite for {name}"

    def test_noise_var_positive_for_all_models(self, fitted_models):
        for name, m in fitted_models.items():
            nv = noise_variance(m)
            assert nv > 0 and math.isfinite(
                nv
            ), f"noise variance invalid for {name}: {nv}"
