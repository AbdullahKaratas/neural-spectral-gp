import math

import torch
import torch.nn as nn
import numpy as np
from gpytorch.kernels import Kernel
from gpytorch.mlls import AddedLossTerm
from typing import List


class GaussianPriorLoss(AddedLossTerm):
    """Log Gaussian prior on NN weights: log p(W) = -0.5/sigma^2 * ||W||^2.

    GPyTorch ADDS this to the log likelihood, so we return the log prior
    (negative value) to penalize large weights.
    """

    def __init__(self, parameters, prior_variance: float = 1.0):
        super().__init__()
        self.parameters = parameters
        self.prior_variance = prior_variance

    def loss(self):
        weights = torch.cat([p.view(-1) for name, p in self.parameters() if "weight" in name])
        return -0.5 * weights.norm().pow(2) / self.prior_variance


class GSMParameterNet(nn.Module):
    """
    Shared-backbone MLP that outputs Q positive values.

    Neural-GSM kernel shares hidden layers across Q components,
    separate final layer per component, softplus output.
    (Remes et al. 2018)
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int,
        hidden_dims: List[int] = [32, 32],
    ):
        super().__init__()
        self.n_components = n_components

        # Shared backbone with SELU
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SELU())
            prev = h
        self.backbone = nn.Sequential(*layers)

        # Separate scalar final layers per component (paper: shared except final layer)
        self.heads = nn.ModuleList([
            nn.Linear(prev, 1)
            for _ in range(n_components)
        ])

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform + zero bias everywhere (matches original GPflow repo)
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                limit = np.sqrt(6.0 / (m.in_features + m.out_features))
                nn.init.uniform_(m.weight, -limit, limit)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for head in self.heads:
            limit = np.sqrt(6.0 / (head.in_features + head.out_features))
            nn.init.uniform_(head.weight, -limit, limit)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (... x N x D)

        Returns
        -------
        out : (... x N x Q)
        """
        h = self.backbone(x)
        # Each head outputs (... x N x 1), cat along last dim -> (... x N x Q)
        return torch.cat([nn.functional.softplus(head(h)) for head in self.heads], dim=-1)


class NeuralGSMKernel(Kernel):
    """
    Neural-GSM kernel (Remes et al. 2018).
    https://github.com/sremes/nssm-gp

    Parameters
    ----------
    input_dim : int
        Dimensionality of input space.
    n_components : int
        Number of spectral mixture components Q.
    hidden_dims : list of int
        Hidden layer sizes for parameter networks.
    prior_variance : float
        Variance of Gaussian prior on NN weights. W sim N(0, prior_variance * I).
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(
        self,
        input_dim: int,
        n_components: int = 1,
        hidden_dims: List[int] = [32, 32],
        prior_variance: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_components = n_components
        self.prior_variance = prior_variance

        # Three parameter networks (paper Section 5)
        # w(x) -> Q scalars (variance/amplitude)
        self.var_net = GSMParameterNet(input_dim, n_components, hidden_dims=hidden_dims)
        # ell(x) -> Q scalars (lengthscales)
        self.len_net = GSMParameterNet(input_dim, n_components, hidden_dims=hidden_dims)
        # mu(x) -> Q scalars (frequencies)
        self.freq_net = GSMParameterNet(input_dim, n_components, hidden_dims=hidden_dims)

        # Gaussian prior on NN weights (matches GPflow's Gaussian(0, 1) prior)
        self.register_added_loss_term("nn_weight_prior")
        self.update_added_loss_term(
            "nn_weight_prior",
            GaussianPriorLoss(self.named_parameters, prior_variance),
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        """
        Compute the Neural-GSM kernel matrix.

        Parameters
        ----------
        x1 : (... x N x D)
        x2 : (... x M x D)
        diag : bool
            If True, return only diagonal elements.

        Returns
        -------
        K : (... x N x M) or (... x N,) if diag
        """
        x1_eq_x2 = torch.equal(x1, x2)

        w1_all = self.var_net(x1)
        l1_all = self.len_net(x1)
        mu1_all = self.freq_net(x1)

        if x1_eq_x2:
            w2_all, l2_all, mu2_all = w1_all, l1_all, mu1_all
        else:
            w2_all = self.var_net(x2)
            l2_all = self.len_net(x2)
            mu2_all = self.freq_net(x2)

        dist_sq = self.covar_dist(x1, x2, square_dist=True, diag=diag, **params)

        # Accumulate kernel over Q components
        K = None

        for q in range(self.n_components):
            w1 = w1_all[..., q].unsqueeze(-1)
            w2 = w2_all[..., q].unsqueeze(-1)
            l1 = l1_all[..., q].unsqueeze(-1)
            l2 = l2_all[..., q].unsqueeze(-1)
            mu1 = mu1_all[..., q].unsqueeze(-1)
            mu2 = mu2_all[..., q].unsqueeze(-1)

            if diag:
                WW = (w1 * w2).squeeze(-1)
                if x1_eq_x2:
                    gibbs = torch.ones_like(WW)
                    cos_term = torch.ones_like(WW)
                else:
                    S = (l1.pow(2) + l2.pow(2)).squeeze(-1)
                    prod = (l1 * l2).squeeze(-1)
                    prefactor = (2.0 * prod / S).sqrt()
                    gibbs = prefactor * (-dist_sq / S).exp()

                    phase1 = (mu1 * x1).sum(dim=-1)
                    phase2 = (mu2 * x2).sum(dim=-1)
                    cos_term = torch.cos(2.0 * math.pi * (phase1 - phase2))

            else:

                S = l1.pow(2) + l2.pow(2).transpose(-2, -1)
                prod = l1 * l2.transpose(-2, -1)
                WW = w1 @ w2.transpose(-2, -1)

                prefactor = (2.0 * prod / S).sqrt()
                gibbs = prefactor * (-dist_sq / S).exp()

                phase1 = (mu1 * x1).sum(dim=-1, keepdim=True)  # (... x N x 1)
                phase2 = (mu2 * x2).sum(dim=-1, keepdim=True)  # (... x M x 1)
                cos_term = torch.cos(2.0 * math.pi * self.covar_dist(
                    phase1, phase2, square_dist=False, diag=False, **params
                ))

            Kq = WW * gibbs * cos_term

            K = Kq if K is None else K + Kq

        return K
