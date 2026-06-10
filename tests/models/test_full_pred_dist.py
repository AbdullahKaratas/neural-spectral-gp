import pytest
import torch
import gpytorch

from nsgp.models.standard_gp import StandardGP
from nsgp.models.neural_gsm_gp import NeuralGSMGP
from nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork
from nsgp.models.dkl_gp import DKLGP


@pytest.fixture
def data():
    torch.manual_seed(0)
    X_train = torch.linspace(-3, 3, 20).unsqueeze(-1)
    y_train = torch.sin(X_train).squeeze() + 0.1 * torch.randn(20)
    X_test = torch.linspace(-2, 2, 10).unsqueeze(-1)
    y_test = torch.sin(X_test).squeeze()
    return X_train, y_train, X_test, y_test


@pytest.fixture
def fitted_standard_gp(data):
    X_train, y_train, _, _ = data
    model = StandardGP()
    model.fit(X_train, y_train, epochs=5, verbose=False)
    return model


@pytest.fixture
def fitted_neural_gsm(data):
    X_train, y_train, _, _ = data
    model = NeuralGSMGP(input_dim=1, n_components=1, hidden_dims=[16])
    model.fit(X_train, y_train, epochs=5, verbose=False)
    return model


@pytest.fixture
def fitted_fsdn(data):
    X_train, y_train, _, _ = data
    model = FactorizedSpectralDensityNetwork(
        input_dim=1, hidden_dims=[16], rank=4, n_features=20, omega_max=5.0
    )
    model.fit(X_train, y_train, epochs=5, verbose=False)
    return model


@pytest.fixture
def fitted_dkl(data):
    X_train, y_train, _, _ = data
    model = DKLGP(input_dim=1, output_dim=2, hidden_dims=[8, 8])
    model.fit(X_train, y_train, epochs=5, verbose=False)
    return model


def assert_valid_nlpd(model, X_test, y_test):
    dist = model._full_pred_dist(X_test)
    nlpd = gpytorch.metrics.negative_log_predictive_density(dist, y_test)
    print(nlpd)
    assert nlpd.shape == torch.Size([])
    assert torch.isfinite(nlpd)


class TestNLPD:
    def test_standard_gp_nlpd(self, fitted_standard_gp, data):
        _, _, X_test, y_test = data
        assert_valid_nlpd(fitted_standard_gp, X_test, y_test)

    def test_neural_gsm_nlpd(self, fitted_neural_gsm, data):
        _, _, X_test, y_test = data
        assert_valid_nlpd(fitted_neural_gsm, X_test, y_test)

    def test_fsdn_nlpd(self, fitted_fsdn, data):
        _, _, X_test, y_test = data
        assert_valid_nlpd(fitted_fsdn, X_test, y_test)

    def test_dkl_nlpd(self, fitted_dkl, data):
        _, _, X_test, y_test = data
        assert_valid_nlpd(fitted_dkl, X_test, y_test)
