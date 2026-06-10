"""
Custom metrics / values for the kernel-learning baseline
comparison. Some metrics not covered by `gpytorch.metrics`.
"""
from typing import Callable

import torch
import gpytorch

from .models.standard_gp import StandardGP
from .models.neural_gsm_gp import NeuralGSMGP
from .models.dkl_gp import DKLGP
from .models.sdn_factorized import FactorizedSpectralDensityNetwork


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


def oracle_posterior(
    kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    noise_var: float,
) -> gpytorch.distributions.MultivariateNormal:
    """
    True GP predictive distribution under the ground-truth kernel.
    Returns a MultivariateNormal at X_test.
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

    # Add observation noise for predictive distribution
    n_test = X_test.shape[0]
    cov = cov + noise_var * torch.eye(n_test, dtype=cov.dtype, device=cov.device)
    return gpytorch.distributions.MultivariateNormal(mean, cov)


def marginal_log_likelihood(model) -> float:
    """Marginal log-likelihood of the training data under the fitted model.

    All models recompute directly from the restored best weights:
    - StandardGP / DKLGP: exact MLL via ExactMarginalLogLikelihood.
    - NeuralGSMGP: MAP objective minus the Gaussian weight prior
      (GPyTorch includes the AddedLossTerm automatically).
    - F-SDN: low-rank MLL via log_marginal_likelihood.
    """
    if isinstance(model, (StandardGP, DKLGP, NeuralGSMGP)):
        if model.model is None:
            raise RuntimeError("Model not fitted yet.")
        mll_obj = gpytorch.mlls.ExactMarginalLogLikelihood(
            model.likelihood, model.model
        )
        X_train = model.model.train_inputs[0]
        y_train = model.model.train_targets
        model.model.train()
        model.likelihood.train()
        try:
            with torch.no_grad():
                map_val = mll_obj(model.model(X_train), y_train)
                if isinstance(model, NeuralGSMGP):
                    # Remove the weight prior (AddedLossTerm is included by GPyTorch)
                    map_val = (
                        map_val
                        - model.kernel._added_loss_terms["nn_weight_prior"].loss()
                    )
        finally:
            model.model.eval()
            model.likelihood.eval()
        return float(map_val.item())
    if isinstance(model, FactorizedSpectralDensityNetwork):
        if model.X_train is None:
            raise RuntimeError("Model not fitted yet.")
        with torch.no_grad():
            L = model.compute_lowrank_features(model.X_train)
            noise_var = torch.exp(model.log_noise_var)
            nll = model.log_marginal_likelihood(L, model.y_train, noise_var)
        n = model.X_train.shape[0]
        return float(-nll.item() / n)
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def noise_variance(model) -> float:
    """Fitted observation noise variance, in original (non-log) scale."""
    if isinstance(model, (StandardGP, NeuralGSMGP, DKLGP)):
        return float(model.likelihood.noise.item())
    if isinstance(model, FactorizedSpectralDensityNetwork):
        return float(torch.exp(model.log_noise_var).item())
    raise TypeError(f"Unsupported model type: {type(model).__name__}")
