import pytest
import torch

from nsgp.models.dkl_gp import DKLGP


@pytest.fixture
def data():
    torch.manual_seed(0)
    X_train = torch.linspace(-3, 3, 20).unsqueeze(-1)
    y_train = torch.sin(X_train).squeeze() + 0.1 * torch.randn(20)
    X_test = torch.linspace(-2, 2, 10).unsqueeze(-1)
    return X_train, y_train, X_test


@pytest.fixture
def fitted_dkl(data):
    X_train, y_train, _ = data
    model = DKLGP(input_dim=1, output_dim=2, hidden_dims=[8, 8])
    model.fit(X_train, y_train, epochs=5, verbose=False)
    return model


class TestDKLGP:
    def test_predict_shapes(self, fitted_dkl, data):
        _, _, X_test = data
        mean, std = fitted_dkl.predict(X_test)
        assert mean.shape == (10,)
        assert std.shape == (10,)
