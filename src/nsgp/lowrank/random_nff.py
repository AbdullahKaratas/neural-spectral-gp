import math
from typing import Optional, Callable

import torch


class RandomNonstationaryFeatures:
    """
    Monte Carlo nonstationary Fourier features (Ton et al., 2018).

    Parameters
    ----------
    spectral_sampler : Callable
        Function that returns (omega1, omega2) each of shape (m, D)
        representing m frequency pairs sampled from the spectral density.
    n_feat : int
        Number of frequency pairs m.
    """

    def __init__(
        self,
        spectral_sampler: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        n_feat: int = 100,
    ):
        self.spectral_sampler = spectral_sampler
        self.n_feat = n_feat
        self.omega1 = None
        self.omega2 = None

    def sample_frequencies(self, seed: Optional[int] = None):
        """
        Draw frequency pairs from the spectral density.

        Seeding uses a local ``torch.Generator`` and is forwarded to
        ``spectral_sampler`` as a ``generator=`` kwarg if supported, so the
        global RNG is not mutated. Samplers that don't accept ``generator``
        fall back to a ``torch.manual_seed`` context that is restored after
        sampling.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        """
        if seed is None:
            self.omega1, self.omega2 = self.spectral_sampler(self.n_feat)
            return

        gen = torch.Generator().manual_seed(seed)
        try:
            self.omega1, self.omega2 = self.spectral_sampler(self.n_feat, generator=gen)
        except TypeError:
            prev_state = torch.random.get_rng_state()
            try:
                torch.manual_seed(seed)
                self.omega1, self.omega2 = self.spectral_sampler(self.n_feat)
            finally:
                torch.random.set_rng_state(prev_state)

    def compute_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the nonstationary Fourier feature map.

        Phi = [cos(X Omega1^T) + cos(X Omega2^T) | sin(X Omega1^T) + sin(X Omega2^T)]

        Parameters
        ----------
        X : torch.Tensor, shape (n, D)
            Input locations.

        Returns
        -------
        Phi : torch.Tensor, shape (n, 2m)
        """
        if self.omega1 is None:
            raise RuntimeError(
                "Frequencies not sampled yet. Call sample_frequencies() first."
            )

        if X.dim() == 1:
            X = X.unsqueeze(-1)

        # (n, D) @ (D, m) -> (n, m)
        phase1 = X @ self.omega1.T
        phase2 = X @ self.omega2.T

        cos_part = torch.cos(phase1) + torch.cos(phase2)
        sin_part = torch.sin(phase1) + torch.sin(phase2)

        return torch.cat([cos_part, sin_part], dim=1)  # (n, 2m)

    def lowrank(self, X: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        Compute low-rank feature matrix L such that K ~ L @ L^T.

        L = (1 / sqrt(4m)) * Phi, so that L @ L^T = (1/4m) * Phi @ Phi^T = K_hat.

        Parameters
        ----------
        X : torch.Tensor, shape (n, D)
        seed : int, optional
            If provided, resamples frequencies before computing features.

        Returns
        -------
        L : torch.Tensor, shape (n, 2m)
        """
        if seed is not None or self.omega1 is None:
            self.sample_frequencies(seed=seed)

        Phi = self.compute_features(X)
        return Phi.mul(0.5).div(math.sqrt(self.n_feat))

    def kernel_estimate(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute K_hat = (1/4m) * Phi(X1) @ Phi(X2)^T.

        The ``X2 is X1`` check is an identity check (not ``torch.equal``)
        because it only exists to skip a redundant feature computation for
        the symmetric case. Two independently constructed tensors with the
        same values fall through to the cross-kernel path, which gives the
        same numerical result.

        Parameters
        ----------
        X1 : torch.Tensor, shape (n1, D)
        X2 : torch.Tensor, shape (n2, D), optional

        Returns
        -------
        K : torch.Tensor, shape (n1, n2)
        """
        # lowrank(X1) may resample frequencies if omega1 is None;
        # lowrank(X2) reuses the same frequencies (desired).
        Phi1 = self.lowrank(X1)

        if X2 is None or X2 is X1:
            return Phi1 @ Phi1.T

        Phi2 = self.lowrank(X2)
        return Phi1 @ Phi2.T

    def simulation(
        self,
        X: torch.Tensor,
        n_samples: int = 1,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample from the GP prior using the MC low-rank approximation.

        Parameters
        ----------
        X : torch.Tensor, shape (n, D)
        n_samples : int
        seed : int, optional

        Returns
        -------
        samples : torch.Tensor, shape (n_samples, n)
        """
        L = self.lowrank(X, seed=seed)
        rvs = torch.randn(L.shape[1], n_samples, dtype=L.dtype)
        return (L @ rvs).T
