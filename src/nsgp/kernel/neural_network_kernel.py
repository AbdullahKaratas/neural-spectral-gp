import math

import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


class NeuralNetworkKernel(Kernel):
    r"""
    Neural network kernel.

    Covariance of a single hidden-layer neural network with erf
    activations in the infinite-width limit. This is eq. (11) of
    Williams (1996), "Computing with Infinite Networks".

    Parameters
    ----------
    aug_dim : int
        Equals d + 1, when d is the input dimension.
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, aug_dim, variance_prior=None, variance_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.aug_dim = aug_dim

        self.register_parameter(
            name="raw_variance",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, aug_dim)),
        )

        if variance_constraint is None:
            variance_constraint = Positive()

        if variance_prior is not None:
            self.register_prior(
                "variance_prior",
                variance_prior,
                lambda m: m.variance,
                lambda m, v: m._set_variance(v),
            )

        self.register_constraint("raw_variance", variance_constraint)

    @property
    def variance(self) -> torch.Tensor:
        """Diagonal of Sigma"""
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        self._set_variance(value)

    def _set_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        self.initialize(
            raw_variance=self.raw_variance_constraint.inverse_transform(value)
        )

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend a constant 1 to the feature."""
        ones = torch.ones(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
        return torch.cat([ones, x], dim=-1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x1 : (... x N x D)
        x2 : (... x M x D)
        diag : bool
            If True, return only the diagonal matrix.

        Returns
        -------
        K : (... x N x M) or (... x N,) if diag.
        """

        if self.aug_dim != x1.shape[-1] + 1:
            raise ValueError(
                f"Expected inputs with {self.aug_dim - 1} dimension, "
                f"got {x1.shape[-1]}."
            )
        x1_ = self._augment(x1)
        x2_ = self._augment(x2)

        # (... x 1 x D+1)
        var = self.variance.unsqueeze(-2)
        sx2 = x2_ * var

        # x^T Sigma x
        denom1 = 1.0 + 2.0 * (x1_ * x1_ * var).sum(-1)
        denom2 = 1.0 + 2.0 * (x2_ * sx2).sum(-1)

        if diag:
            num = 2.0 * (x1_ * sx2).sum(-1)  # (... x N)
            ratio = num / torch.sqrt(denom1 * denom2)
        else:
            num = 2.0 * (x1_ @ sx2.transpose(-2, -1))  # (... x N x M)
            denom = torch.sqrt(denom1.unsqueeze(-1) * denom2.unsqueeze(-2))
            ratio = num / denom

        ratio = ratio.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.asin(ratio).mul(2.0).div(math.pi)
