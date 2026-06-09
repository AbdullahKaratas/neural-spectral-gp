import math
import warnings
from typing import Callable, Optional

import torch


def sq_exp(x1, x2, dist=True):
    x1_eq_x2 = torch.equal(x1, x2)

    if dist:
        adjustment = x1.mean(-2, keepdim=True)
    else:
        adjustment = 0.0
    x1 = x1 - adjustment

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        x2, x2_norm, x2_pad = x1, x1_norm, x1_pad
    else:
        x2 = (
            x2 - adjustment
        )  # x1 and x2 should be identical in all dims except -2 at this point
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
    if dist:
        x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    else:
        x1_ = torch.cat([2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad and dist:
        res.diagonal(dim1=-2, dim2=-1).fill_(0)

    # Zero out negative values
    return res.clamp_min_(0)


def _objective_value(model, metric, X_val = None, y_val = None) -> float:

    _aliases = {
        "nlpd": "negative_log_predictive_density",
        "mse": "mean_squared_error",
        "mae": "mean_absolute_error",
        "msll": "mean_standardized_log_loss",
        "smse": "standardized_mean_squared_error",
        "qce": "quantile_coverage_error",
        "mll": "marginal_log_likelihood",
    }
    name = _aliases.get(metric.lower(), metric)

    # minimize -mll
    if name == "marginal_log_likelihood":
        from nsgp.metrics import marginal_log_likelihood
        return -marginal_log_likelihood(model)

    import gpytorch.metrics as gpytorch_metrics
    fn = getattr(gpytorch_metrics, name, None)
    if fn is None:
        raise ValueError(f"Unknown metric '{metric}'.")
    if X_val is None or y_val is None:
        raise ValueError(f"metric='{metric}' requires X_val and y_val.")
    with torch.no_grad():
        mvn = model._full_pred_dist(X_val, predictive_dist=True)
        return fn(mvn, y_val.reshape(-1)).item()


def optimize_hyperparameters(
    model_fn: Callable,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    n_trials: int = 40,
    metric: str = "mll",
    fit_kwargs: Optional[dict] = None,
    show_progress_bar: bool = True,
    **kwargs,
):
    """
    Optuna search over a hyperparameters.

    model_fn(trial) builds a model from trial-suggested hyperparameters. Each
    trial fits it on (X_train, y_train) and scores it by metric.

    Parameters
    ----------
    model_fn : Callable
        Builds a model from an Optuna trial's suggested hyperparameters.
    X_train, y_train : torch.Tensor
        Training data.
    X_val, y_val : torch.Tensor, optional
        Validation data; required for every metric except 'mll'.
    seed : int, optional
        Seeds the sampler and torch (seed + trial.number).
    n_trials : int
        Number of Optuna trials.
    metric : str
        Objective to minimize. Any gpytorch.metrics name or alias ('nlpd',
        'mse', 'mae', 'msll', 'smse', 'qce'), scored on (X_val, y_val); 'mll'
        maximizes the training marginal likelihood and needs no validation set.
    fit_kwargs : dict, optional
        Kwargs forwarded to every model.fit call (e.g. 'epochs', 'lr').
    show_progress_bar : bool
    **kwargs
        Forwarded to optuna.create_study.

    Returns
    -------
    optuna.study.Study
    """
    import optuna

    # Seed the sampler for a reproducible search by default
    kwargs.setdefault("sampler", optuna.samplers.TPESampler(seed=seed))

    def objective(trial):
        if seed is not None:
            torch.manual_seed(seed + trial.number)

        model = model_fn(trial)
        fk = {"verbose": False, **(fit_kwargs or {})}

        try:
            model.fit(X_train, y_train, **fk)
            value = _objective_value(model, metric, X_val, y_val)
        except Exception as exc:  # skip configs that fail
            warnings.warn(f"Trial {trial.number} failed ({type(exc).__name__}: {exc}).")
            return float("inf")

        return value if math.isfinite(value) else float("inf")

    study = optuna.create_study(
        direction="minimize",
        **kwargs
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


def build_model(model_fn: Callable, params: dict):
    """
    Build a model from a parameter dict.
    """
    import optuna
    return model_fn(optuna.trial.FixedTrial(params))
