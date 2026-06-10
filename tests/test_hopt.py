import math

import pytest
import torch

from nsgp.models import (
    DKLGP,
    FactorizedSpectralDensityNetwork,
    NeuralGSMGP,
)
from nsgp.utils import build_model, optimize_hyperparameters


@pytest.fixture
def data():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 10).unsqueeze(-1)
    y = torch.sin(X).squeeze() + 0.1 * torch.randn(10)
    Xv = torch.linspace(-2, 2, 5).unsqueeze(-1)
    yv = torch.sin(Xv).squeeze()
    yield X, y, Xv, yv
    torch.set_default_dtype(prev)


def _factory(cls):
    def make(trial):
        h = trial.suggest_categorical("h", [16, 32])
        if cls is FactorizedSpectralDensityNetwork:
            return cls(input_dim=1, hidden_dims=[h, h], rank=2)
        return cls(input_dim=1, hidden_dims=[h, h])

    return make


@pytest.mark.parametrize(
    "cls, metric",
    [
        (DKLGP, "mll"),
        (NeuralGSMGP, "nlpd"),
        (FactorizedSpectralDensityNetwork, "mll"),
    ],
)
def test_optimize_across_models(data, cls, metric):
    X, y, Xv, yv = data
    val = () if metric == "mll" else (Xv, yv)
    study = optimize_hyperparameters(
        _factory(cls),
        X,
        y,
        *val,
        n_trials=2,
        fit_kwargs=dict(epochs=3),
        metric=metric,
        seed=0,
        show_progress_bar=False,
    )
    assert math.isfinite(study.best_value)
    assert isinstance(build_model(_factory(cls), study.best_params), cls)
