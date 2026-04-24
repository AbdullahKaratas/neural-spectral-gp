from typing import List

import numpy as np
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    MLP feature map phi: R^D -> R^F used as the neural part of a DKL GP.

    Architecture: Linear -> SELU -> ... -> Linear (no activation on output).
    Xavier uniform init + zero bias to match the style of `neural_gsm.py`.

    Intended usage follows the GPyTorch reference implementation of Deep
    Kernel Learning (Wilson et al. 2016): apply `phi` inside a GP model's
    `forward` and feed the result into a stationary base kernel such as
    `ScaleKernel(RBFKernel(ard_num_dims=F))`. No dedicated kernel wrapper
    is needed.

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
