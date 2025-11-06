"""
Regular Nonstationary Fourier Features (NFFs)

Implementation of the method from Jawaid (2024) for efficient simulation
of harmonizable stochastic processes.
"""

import torch
import numpy as np
from typing import Callable, Optional


class NFFs:
    """
    Regular Nonstationary Fourier Features for GP simulation.

    This class implements Algorithm 6.1 from Jawaid (2024), enabling
    efficient O(M·n) simulation of nonstationary Gaussian processes
    with harmonizable covariance structure.

    Parameters
    ----------
    spectral_density : Callable
        Function s(ω, ω') returning spectral density at frequency pairs
    n_features : int
        Number of Fourier features M
    omega_max : float
        Cutoff frequency for integration domain
    input_dim : int
        Spatial dimension
    """

    def __init__(
        self,
        spectral_density: Callable,
        n_features: int = 100,
        omega_max: float = 10.0,
        input_dim: int = 1
    ):
        self.spectral_density = spectral_density
        self.n_features = n_features
        self.omega_max = omega_max
        self.input_dim = input_dim

        # Pre-compute frequency grid
        self._setup_frequency_grid()

    def _setup_frequency_grid(self):
        """
        Create regular grid of frequencies in [-ω_max, ω_max]^d.
        """
        # Create 1D grid for each dimension
        if self.input_dim == 1:
            self.omegas = torch.linspace(
                -self.omega_max,
                self.omega_max,
                self.n_features
            )
        else:
            # Multi-dimensional grid
            n_per_dim = int(np.ceil(self.n_features ** (1/self.input_dim)))
            grids = [
                torch.linspace(-self.omega_max, self.omega_max, n_per_dim)
                for _ in range(self.input_dim)
            ]
            # Create meshgrid
            meshgrids = torch.meshgrid(*grids, indexing='ij')
            self.omegas = torch.stack(
                [grid.flatten() for grid in meshgrids],
                dim=-1
            )[:self.n_features]

    def _compute_spectral_matrix(self) -> torch.Tensor:
        """
        Compute spectral density matrix S with entries s(ω_i, ω_j).

        Returns
        -------
        torch.Tensor, shape (M, M)
            Spectral density matrix
        """
        M = self.omegas.shape[0]
        S = torch.zeros(M, M)

        for i in range(M):
            for j in range(M):
                omega_i = self.omegas[i:i+1]
                omega_j = self.omegas[j:j+1]
                S[i, j] = self.spectral_density(omega_i, omega_j)

        return S

    def _whiten(self, S: torch.Tensor) -> torch.Tensor:
        """
        Compute whitening transformation via Cholesky decomposition.

        Parameters
        ----------
        S : torch.Tensor, shape (M, M)
            Spectral density matrix

        Returns
        -------
        torch.Tensor, shape (M, M)
            Lower triangular Cholesky factor L where S = LL^T
        """
        # Add regularization for numerical stability and positive definiteness
        # Adaptive jitter based on matrix norm
        jitter = max(1e-6, 0.01 * torch.mean(torch.diag(S)).item())
        S_reg = S + jitter * torch.eye(S.shape[0])

        # Try Cholesky with increasing jitter if needed
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                L = torch.linalg.cholesky(S_reg)
                return L
            except RuntimeError:
                if attempt < max_attempts - 1:
                    jitter *= 10
                    S_reg = S + jitter * torch.eye(S.shape[0])
                else:
                    raise

    def simulate(
        self,
        X: torch.Tensor,
        n_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Simulate GP realizations at locations X.

        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Locations for simulation
        n_samples : int, optional
            Number of independent sample paths, default=1
        seed : int, optional
            Random seed for reproducibility, default=None

        Returns
        -------
        torch.Tensor, shape (n_samples, n)
            Simulated GP values
        """
        if seed is not None:
            torch.manual_seed(seed)

        n = X.shape[0]
        M = self.omegas.shape[0]

        # Step 1: Compute spectral matrix S
        S = self._compute_spectral_matrix()

        # Step 2: Whitening transformation L
        L = self._whiten(S)

        # Step 3: Sample white noise
        xi = torch.randn(n_samples, M, 2)  # Real and imaginary parts

        # Step 4: Apply whitening
        xi_transformed = torch.einsum('sm,nmc->nsc', L, xi)

        # Step 5: Compute Fourier basis at locations X
        # Φ(x) = [cos(ω₁ᵀx), ..., cos(ωₘᵀx), sin(ω₁ᵀx), ..., sin(ωₘᵀx)]
        if self.input_dim == 1:
            omega_x = torch.outer(X.squeeze(), self.omegas)  # (n, M)
        else:
            omega_x = X @ self.omegas.T  # (n, M)

        cos_basis = torch.cos(omega_x)  # (n, M)
        sin_basis = torch.sin(omega_x)  # (n, M)

        # Step 6: Combine to get GP samples
        # Z(x) = Re(∑ exp(iωx)·ξ) = ∑[cos(ωx)·ξ^R - sin(ωx)·ξ^I]
        Z = torch.zeros(n_samples, n)
        for i in range(n_samples):
            # Real part contribution
            Z[i] = torch.sum(xi_transformed[i, :, 0:1].T * cos_basis, dim=1)
            # Imaginary part contribution (note the MINUS sign!)
            Z[i] -= torch.sum(xi_transformed[i, :, 1:2].T * sin_basis, dim=1)

        # Normalize by volume element and Fourier transform convention
        # The 1/(2π)^d factor comes from the inverse Fourier transform
        volume = (2 * self.omega_max) ** self.input_dim / M
        fourier_norm = (2 * np.pi) ** self.input_dim
        Z = Z * np.sqrt(volume / fourier_norm)

        return Z
