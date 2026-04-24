from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel


class FeatureExtractor(nn.Module):
    """
    MLP feature map phi: R^D -> R^F used as the neural part of the DKL kernel.

    Architecture: Linear -> SELU -> ... -> Linear (no activation on output).
    Xavier uniform init + zero bias to match the style of `neural_gsm.py`.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input space D.
    output_dim : int
        Dimensionality of the feature space F (output of phi).
    hidden_dims : list of int
        Sizes of hidden layers. Empty list gives a single linear map.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [32, 32],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)

        layers: List[nn.Module] = []
        prev = input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SELU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                limit = np.sqrt(6.0 / (m.in_features + m.out_features))
                nn.init.uniform_(m.weight, -limit, limit)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (... x N x D)

        Returns
        -------
        phi : (... x N x F)
        """
        return self.net(x)


class DeepKernel(Kernel):
    """
    Deep Kernel Learning (Wilson et al. 2016): k(x, x') = k_base(phi(x), phi(x')).

    Stationary in feature space, non-stationary in input space through phi.
    phi is a `FeatureExtractor` MLP; the base kernel defaults to a scaled RBF.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input space D.
    feature_dim : int
        Dimensionality of the feature space F.
    hidden_dims : list of int
        Hidden layer sizes for the feature extractor.
    base_kernel : Kernel, optional
        Stationary base kernel on the feature space. Defaults to
        `ScaleKernel(RBFKernel(ard_num_dims=feature_dim))`.

    Notes
    -----
    `has_lengthscale = False` on purpose: the lengthscale lives inside
    the base kernel, not on this wrapper.
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 2,
        hidden_dims: List[int] = [32, 32],
        base_kernel: Optional[Kernel] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            output_dim=feature_dim,
            hidden_dims=hidden_dims,
        )

        if base_kernel is None:
            base_kernel = ScaleKernel(RBFKernel(ard_num_dims=feature_dim))
        self.base_kernel = base_kernel

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        """
        Compute the DKL kernel matrix.

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
        phi1 = self.feature_extractor(x1)
        if torch.equal(x1, x2):
            phi2 = phi1
        else:
            phi2 = self.feature_extractor(x2)

        return self.base_kernel.forward(phi1, phi2, diag=diag, **params)
