"""
Posterior-quality metrics for the kernel-learning baseline comparison.

Four metrics, addressing R1 + R3 of the UAI rebuttal:

- ``negative_log_predictive_density(pred_dist, y_test)``
    Standard NLPD via GPyTorch. Average negative log predictive density per
    test point under the fitted posterior, evaluated on noisy observations.

- ``kl_posterior(mvn_fit, mvn_oracle)``
    KL(q || p) between the fitted posterior and the oracle Bayes posterior
    obtained by running standard GP regression with the *true* kernel and
    known noise variance. Variant (a): "how close is my Bayes inference to
    ideal Bayes inference."

- ``log_pred_density_true_function(mvn_fit, f_true_test)``
    Log predictive density of the noiseless true function values under the
    fitted posterior. R1 explicitly asks for this. Variant (b).

- ``oracle_posterior(kernel_fn, X_train, y_train, X_test, noise_var)``
    The "true" posterior used for the KL above: GP regression with the
    ground-truth kernel and noise variance.

Per-model adapters for marginal likelihood and noise variance live alongside
because the API differs between the GPyTorch-backed models and F-SDN.
"""
from typing import Callable

import torch
import gpytorch

from .models.standard_gp import StandardGP
from .models.neural_gsm_gp import NeuralGSMGP
from .models.dkl_gp import DKLGP
from .models.sdn_factorized import FactorizedSpectralDensityNetwork


def negative_log_predictive_density(
    pred_dist: gpytorch.distributions.MultivariateNormal,
    y_test: torch.Tensor,
) -> torch.Tensor:
    """Average NLPD per test point under the predictive distribution."""
    return gpytorch.metrics.negative_log_predictive_density(pred_dist, y_test)


def kl_posterior(
    mvn_fit: gpytorch.distributions.MultivariateNormal,
    mvn_oracle: gpytorch.distributions.MultivariateNormal,
) -> torch.Tensor:
    """
    KL( fitted || oracle ) between two MultivariateNormal posteriors at the
    same test locations.

    The oracle posterior is obtained from GP regression under the
    ground-truth kernel — see ``oracle_posterior`` below.
    """
    return torch.distributions.kl.kl_divergence(mvn_fit, mvn_oracle)


def log_pred_density_true_function(
    mvn_fit: gpytorch.distributions.MultivariateNormal,
    f_true_test: torch.Tensor,
) -> torch.Tensor:
    """
    Log predictive density of the noiseless ground-truth function values
    under the fitted posterior. Returns total log density (not per-point).

    R1 explicitly asks for this.
    """
    return mvn_fit.log_prob(f_true_test)


def oracle_posterior(
    kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    noise_var: float,
) -> gpytorch.distributions.MultivariateNormal:
    """
    Bayes-optimal GP posterior under the ground-truth kernel.

    Returns a MultivariateNormal over the noiseless function values at
    X_test, i.e. the predictive distribution **without** observation noise
    — this is the natural target for KL against a fitted posterior on f.
    """
    K_train = kernel_fn(X_train, X_train)
    K_cross = kernel_fn(X_test, X_train)
    K_test = kernel_fn(X_test, X_test)

    n = X_train.shape[0]
    A = K_train + noise_var * torch.eye(n, dtype=K_train.dtype, device=K_train.device)
    L = torch.linalg.cholesky(A)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze(-1)

    mean = K_cross @ alpha
    V = torch.cholesky_solve(K_cross.transpose(-1, -2), L)
    cov = K_test - K_cross @ V

    # Symmetrise + jitter for numerical stability before MVN construction.
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    jitter = 1e-6 * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return gpytorch.distributions.MultivariateNormal(mean, cov + jitter)


def _set_eval_mode(model) -> None:
    """Put both the GP and the likelihood into eval mode for inference."""
    model.model.eval()
    model.likelihood.eval()


def marginal_log_likelihood(model, X_train: torch.Tensor, y_train: torch.Tensor) -> float:
    """
    Marginal log-likelihood of the training data under the fitted model.

    Per-model implementation:
    - StandardGP / NeuralGSMGP / DKLGP use GPyTorch's ExactMarginalLogLikelihood.
    - FactorizedSpectralDensityNetwork has its own low-rank MLL routine,
      which returns a *negative* MLL (training loss); we negate it here.
    """
    if isinstance(model, (StandardGP, NeuralGSMGP, DKLGP)):
        if model.model is None:
            raise RuntimeError("Model not fitted yet.")
        _set_eval_mode(model)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model.model)
        with torch.no_grad():
            output = model.model(X_train)
            return float(mll(output, y_train).item())

    if isinstance(model, FactorizedSpectralDensityNetwork):
        with torch.no_grad():
            L = model.compute_lowrank_features(X_train)
            sigma2 = torch.exp(model.log_noise_var)
            nll = model.log_marginal_likelihood(L, y_train, sigma2)
            return float(-nll.item())

    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def noise_variance(model) -> float:
    """Fitted observation noise variance, in original (non-log) scale."""
    if isinstance(model, (StandardGP, NeuralGSMGP, DKLGP)):
        return float(model.likelihood.noise.item())
    if isinstance(model, FactorizedSpectralDensityNetwork):
        return float(torch.exp(model.log_noise_var).item())
    raise TypeError(f"Unsupported model type: {type(model).__name__}")
