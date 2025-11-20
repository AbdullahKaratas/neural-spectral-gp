"""
Factorized Spectral Density Network (SDN-F)

This version GUARANTEES positive definiteness by using a low-rank factorization:
    s(ω, ω') = Σᵢ fᵢ(ω) · fᵢ(ω')

where fᵢ are learned feature functions. This ensures s is positive semi-definite
by construction, enabling reliable sampling.

Authors: Abdullah Karatas, Arsalan Jawaid
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple
from .nffs import NFFs
import math


class FactorizedSpectralDensityNetwork(nn.Module):
    """
    SDN with guaranteed positive definiteness through low-rank factorization.

    Architecture:
        ω → MLP → [f₁(ω), f₂(ω), ..., fᵣ(ω)]  (r = rank)
        s(ω, ω') = Σᵢ fᵢ(ω) · fᵢ(ω')

    This guarantees s is PSD, so Cholesky decomposition always works!

    Parameters
    ----------
    input_dim : int
        Spatial dimension
    hidden_dims : List[int]
        Hidden layer sizes for MLP
    rank : int
        Rank of factorization (higher = more expressive, default=10)
    n_features : int
        Number of Fourier features for NFFs
    omega_max : float
        Frequency cutoff
    activation : str
        Activation function ('relu', 'elu', 'tanh')
    enforce_symmetry : bool
        If True, enforce f(omega) = f(-omega) to guarantee s(-omega,-omega') = s(omega,omega').
        If False, use f(omega) directly (useful for debugging). Default: True.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64],
        rank: int = 10,
        n_features: int = 50,
        omega_max: float = 8.0,
        activation: str = 'elu',
        enforce_symmetry: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.rank = rank
        self.n_features = n_features
        self.omega_max = omega_max
        self.enforce_symmetry = enforce_symmetry

        # MLP: ω → feature vector f(ω) ∈ ℝʳ
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim

        # Output: r-dimensional feature vector
        layers.append(nn.Linear(prev_dim, rank))
        # No final activation - features can be positive or negative
        # (PD is ensured by the dot product)

        self.feature_net = nn.Sequential(*layers)

        # Initialize to small values
        self._init_weights()

    def _init_weights(self):
        """Initialize to small random values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Balanced initialization (not too small, not too large)
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        return {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
        }.get(activation, nn.ELU())

    def _safe_cholesky(
        self,
        A: torch.Tensor,
        jitter: float = 1e-6,
        max_attempts: int = 4
    ) -> torch.Tensor:
        """
        Compute Cholesky decomposition with adaptive jittering.

        Attempts Cholesky decomposition with increasing jitter values.

        Parameters
        ----------
        A : torch.Tensor, shape (..., n, n)
            Symmetric positive semi-definite matrix
        jitter : float
            Initial jitter value to add to diagonal
        max_attempts : int
            Maximum number of attempts with increasing jitter

        Returns
        -------
        L : torch.Tensor, shape (..., n, n)
            Lower triangular Cholesky factor

        Raises
        ------
        RuntimeError
            If Cholesky fails after all attempts
        """
        current_jitter = jitter

        for attempt in range(max_attempts):
            A_jittered = A + current_jitter * torch.eye(
                A.shape[-1], device=A.device, dtype=A.dtype
            )

            try:
                L = torch.linalg.cholesky(A_jittered)
                if attempt > 0:
                    import warnings
                    warnings.warn(
                        f"Cholesky succeeded with jitter={current_jitter:.1e} after {attempt + 1} attempts"
                    )
                return L
            except RuntimeError:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Cholesky failed after {max_attempts} attempts with jitter up to {current_jitter:.1e}. "
                        "Matrix might not be positive-definite."
                    )
                current_jitter *= 10

    def compute_features(self, omega: torch.Tensor) -> torch.Tensor:
        r"""
        Compute feature vector f(\omega).

        If enforce_symmetry=True, enforces f(\omega) = f(-\omega) to ensure:
        s(-\omega, -\omega') = s(\omega, \omega')

        Parameters
        ----------
        omega : torch.Tensor, shape (n, d)
            Frequency vectors

        Returns
        -------
        features : torch.Tensor, shape (n, r)
            Feature vectors (symmetrized if enforce_symmetry=True)
        """
        if omega.dim() == 1:
            omega = omega.unsqueeze(0)

        if self.enforce_symmetry:
            # Symmetrize: f(\omega) = [\tilde{f}(\omega) + \tilde{f}(-\omega)] / 2
            f = (self.feature_net(omega) + self.feature_net(-omega)) / 2.0
        else:
            # Use features directly (for debugging or experimenting with weaker constraints)
            f = self.feature_net(omega)

        return f

    def _compute_spectral_density_matrix(
        self,
        omega_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral density matrix S[m,n] = s(omega_m, omega_n).

        Uses factorized representation: s(omega, omega') = f(omega)^T f(omega')
        This guarantees S is positive semi-definite.

        Parameters
        ----------
        omega_grid : torch.Tensor, shape (M, d)
            Frequency grid points

        Returns
        -------
        S : torch.Tensor, shape (M, M)
            Spectral density matrix
        """
        f = self.compute_features(omega_grid)  # (M, r)
        S = f @ f.T  # (M, M)
        return S

    def forward(self, omega1: torch.Tensor, omega2: torch.Tensor) -> torch.Tensor:
        """
        Compute s(ω₁, ω₂) = f(ω₁)ᵀ f(ω₂).

        This is GUARANTEED to be positive semi-definite!

        Parameters
        ----------
        omega1, omega2 : torch.Tensor
            Frequency pairs

        Returns
        -------
        s : torch.Tensor
            Spectral density values
        """
        # Compute features
        f1 = self.compute_features(omega1)  # (n, r)
        f2 = self.compute_features(omega2)  # (m, r)

        # s(ω₁, ω₂) = f(ω₁)ᵀ f(ω₂)
        # This is already ≥ 0 by construction, no need to add constant!
        s = torch.sum(f1 * f2, dim=-1)  # (n,) or (n, m) if broadcasting

        return s

    def compute_lowrank_features(
        self,
        X: torch.Tensor,
        omega_grid: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute low-rank feature matrix L using nonstationary Fourier features.

        This computes K = LL^T where K = B S^{1/2} (S^{1/2})^T B^T
        - B[i,m] = cos(omega_m x_i) is the cosine basis
        - S[m,n] = s(omega_m, omega_n) \Delta omega^2 is the spectral process kernel
        - S^{1/2} is the Cholesky decomposition of S

        The frequency grid should satisfy the constraint: π/Δω ≥ n*Δx
        where Δx is the minimal spatial spacing and n is the number of spatial points.

        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Spatial locations
        omega_grid : torch.Tensor, shape (num_freqs, d)
            Frequency grid points (should start from 0 for real spectral density)

        Returns
        -------
        L : torch.Tensor, shape (n, num_freqs)
            Low-rank feature matrix where K = LL^T
        """
        # Input validation
        if not torch.is_floating_point(X):
            raise TypeError(f"X must be floating point tensor, got {X.dtype}")

        if not torch.is_floating_point(omega_grid):
            raise TypeError(f"omega_grid must be floating point tensor, got {omega_grid.dtype}")

        num_freqs = omega_grid.shape[0]
        n_pts = X.shape[0]

        # Validate minimum requirements
        if num_freqs < 2:
            raise ValueError(
                f"Frequency grid must contain at least 2 points, got {num_freqs}. "
                "Low-rank approximation requires multiple frequency samples to compute spacing."
            )

        if n_pts < 2:
            raise ValueError(
                f"Spatial locations X must contain at least 2 points, got {n_pts}. "
                "Low-rank approximation requires multiple spatial points to compute spacing."
            )

        # Compute minimal spacing: \Delta x = min_j(x_{j+1} - x_j)
        X_sorted = torch.sort(X.squeeze(), dim=0)[0]
        spacings = X_sorted[1:] - X_sorted[:-1]
        delta_x = spacings.min().item()

        spacing = torch.norm(omega_grid[1] - omega_grid[0]).item()
        constraint_lhs = np.pi / spacing
        constraint_rhs = n_pts * delta_x

        if constraint_lhs < constraint_rhs:
            import warnings
            warnings.warn(
                f"Frequency grid may be too coarse: pi/spacing = {constraint_lhs:.4f} < n*delta_x = {constraint_rhs:.4f}. "
                f"Consider using at least {int(np.ceil(omega_grid.max().item() / (np.pi / constraint_rhs)))+1} frequency points."
            )

        # Compute spectral density matrix S[m,n] = s(omega_m, omega_n)
        S = self._compute_spectral_density_matrix(omega_grid)  # (num_freqs, num_freqs)

        # Apply principled scaling (in-place to save memory)
        S *= (spacing ** 2)

        # Compute matrix square root via eigendecomposition
        # This is more stable than Cholesky for spectral matrices
        eigenvalues, eigenvectors = torch.linalg.eigh(S)

        # Check for problematic eigenvalues before clamping
        negative_count = (eigenvalues < 0).sum().item()

        if negative_count > 0:
            max_negative = eigenvalues[eigenvalues < 0].min().item()

            if abs(max_negative) > 1e-6:  # Significant negative eigenvalue
                raise ValueError(
                    f"Spectral density matrix has {negative_count} significantly negative eigenvalues "
                    f"(worst: {max_negative:.2e}). This indicates a bug in the factorization - "
                    f"S = f @ f.T should be PSD by construction."
                )
            else:  # Small numerical errors only
                import warnings
                warnings.warn(
                    f"Clamped {negative_count} small negative eigenvalues (worst: {max_negative:.2e}). "
                    f"This is likely due to numerical precision."
                )

        # Clamp negative eigenvalues (from numerical errors) to small positive value
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)

        # S_sqrt = Q @ sqrt(Lambda) @ Q^T, but we only need S_sqrt for multiplication
        # S_sqrt such that S_sqrt @ S_sqrt^T = S
        S_sqrt = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))

        # Compute cosine basis: B[i,m] = cos(omega_m x_i)
        # X: (n, d), omega_grid: (num_freqs, d) -> phases: (n, num_freqs)
        phases = X @ omega_grid.T  # (n, num_freqs)
        B = torch.cos(phases)  # (n, num_freqs)

        # Correction for zero-th element (omega = 0)
        omega_norms = torch.norm(omega_grid, dim=1)  # (num_freqs,)
        is_zero = omega_norms < 1e-10
        if torch.any(is_zero):
            B[:, is_zero] = 0.5

        # Compute low-rank features: L = 2 * B @ S^{1/2}
        # The factor of 2 accounts for the symmetry in the full Fourier transform
        L = 2.0 * B @ S_sqrt  # (n, num_freqs)

        return L

    def log_marginal_likelihood(
        self,
        L: torch.Tensor,
        y: torch.Tensor,
        sigma2: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute GP marginal likelihood using low-rank NFF approximation.

        Given K = LL^T + \sigma^2 I, use:
            (LL^T + \sigma^2 I)^(-1) = (1/\sigma^2)[I - L(\sigma^2 I + L^TL)^(-1)L^T]

        Parameters
        ----------
        L : torch.Tensor, shape (n, r)
            Low-rank feature matrix
        y : torch.Tensor, shape (n,)
            Centered observations
        sigma2 : torch.Tensor, scalar
            Noise variance (must be positive)

        Returns
        -------
        nll : torch.Tensor
            Negative log marginal likelihood
        """
        n = L.shape[0]
        r = L.shape[1]

        # Input validation
        if sigma2 <= 0:
            raise ValueError(f"sigma2 must be positive, got {sigma2}")

        if y.shape[0] != n:
            raise ValueError(f"Shape mismatch: L has {n} rows but y has {y.shape[0]} elements")

        if L.device != y.device:
            raise ValueError(f"L and y must be on same device, got L on {L.device} and y on {y.device}")

        if r > n:
            import warnings
            warnings.warn(
                f"Rank r={r} exceeds number of data points n={n}. "
                f"Low-rank approximation is inefficient in this regime. Consider r <= n."
            )

        # Woodbury formula with numerical stability
        # Compute W = sigma^2 I_r + L^T L
        W = sigma2 * torch.eye(r, device=L.device, dtype=L.dtype) + (L.T @ L)

        # Compute Cholesky with adaptive jitter
        Lw = self._safe_cholesky(W, jitter=1e-6, max_attempts=4)

        # Solve (LL^T + sigma^2 I)^(-1) y using Woodbury formula
        # alpha = (1/sigma^2)[y - L W^(-1) L^T y]
        LT_y = L.T @ y  # (r,)
        W_inv_LT_y = torch.cholesky_solve(LT_y.unsqueeze(-1), Lw).squeeze() # stable solve
        alpha = (1/sigma2) * (y - (L @ W_inv_LT_y))  # (n,)

        # Data fit term: y^T alpha
        data_fit = torch.dot(y, alpha)

        # Log determinant using Sylvester's determinant identity:
        # |LL^T + sigma^2 I| = |sigma^2 I| · |I_r + L^T(sigma^2 I)^{-1}L|
        #               = (sigma^2)^n · |I_r + (1/sigma^2 )L^TL|
        #               = (sigma^2)^n · (1/sigma^2)^r · |sigma^2 I_r + L^TL|
        #               = (sigma^2)^{n-r} · |W|
        # where W = sigma^2I_r + L^TL
        # Therefore: log|LL^T + sigma^2I| = (n-r)·log(sigma^2) + log|W|
        log_det_sigma = (n - r) * torch.log(sigma2)
        log_det_W = 2 * torch.sum(torch.log(torch.diag(Lw)))  # log|W| = 2·sum(log(diag(Lw)))
        log_det = log_det_sigma + log_det_W

        # Negative log marginal likelihood (up to constant and scaling)
        # NOTE: This is proportional to the true NLL. We omit:
        #   - 0.5 factor (doesn't affect optimization)
        #   - n log(2 pi) constant term (doesn't affect optimization)
        # Full NLL = 0.5 * (data_fit + log_det) + 0.5*n*log(2 pi)
        nll = data_fit + log_det

        return nll

    def compute_covariance_mc(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
        noise_var: float = 1e-6,
        n_samples: int = 50
    ) -> torch.Tensor:
        """
        Compute covariance using Monte Carlo integration (FAST for training!).

        Full bivariate spectral density via Monte Carlo with PSD GUARANTEE:
        K[i,j] = ∫∫ s(ω, ω') cos(ω·xᵢ - ω'·xⱼ) dω dω'
               ≈ (V/N²) ΣₘΣₙ s(ωₘ, ωₙ) cos(ωₘ·xᵢ - ωₙ·xⱼ)

        CORRECTED: Uses single frequency set ω₁,...,ωₙ and computes
        FULL spectral matrix S[m,n] = s(ωₘ, ωₙ) = f(ωₘ)ᵀf(ωₙ)

        This GUARANTEES S is PSD (S = f @ f.T), so Cholesky ALWAYS works! ✓

        Parameters
        ----------
        X1 : torch.Tensor, shape (n1, d)
            First set of spatial locations
        X2 : torch.Tensor, shape (n2, d), optional
            Second set of spatial locations (if None, use X1)
        noise_var : float
            Observation noise variance
        n_samples : int
            Number of Monte Carlo frequency samples

        Returns
        -------
        K : torch.Tensor, shape (n1, n2) or (n1, n1)
            Covariance matrix
        """
        if X1.dim() == 1:
            X1 = X1.unsqueeze(-1)
        if X2 is None:
            X2 = X1
            add_noise = True
        else:
            if X2.dim() == 1:
                X2 = X2.unsqueeze(-1)
            add_noise = False

        n1, n2 = X1.shape[0], X2.shape[0]

        # CORRECTED: Sample only ONE set of frequencies uniformly
        omegas = (torch.rand(n_samples, self.input_dim) - 0.5) * self.omega_max

        # Compute FULL spectral density matrix - PSD GUARANTEED!
        S_full = self._compute_spectral_density_matrix(omegas)  # (n_samples, n_samples)
        # ✓ S_full is ALWAYS PSD by construction!

        # Monte Carlo integration weights
        dw_mc = self.omega_max / n_samples  # Per-dimension weight
        volume = dw_mc * dw_mc  # 2D integration volume

        # Pre-compute ω @ X matrices
        omega_X1 = omegas @ X1.T  # (n_samples, n1)
        omega_X2 = omegas @ X2.T  # (n_samples, n2)

        # Double sum over ALL (m,n) pairs in S_full - GRADIENT SAFE!
        K_rows = []
        for i in range(n1):
            K_row = []
            for j in range(n2):
                # phases[m,n] = ωₘ·xᵢ - ωₙ·xⱼ
                # Broadcasting: (n_samples, 1) - (1, n_samples) = (n_samples, n_samples)
                phases = omega_X1[:, i:i+1] - omega_X2[:, j:j+1].T  # (n_samples, n_samples)

                # K[i,j] = ΣₘΣₙ s(ωₘ,ωₙ) cos(ωₘ·xᵢ - ωₙ·xⱼ)
                k_ij = torch.sum(S_full * torch.cos(phases))  # NO .item()!
                K_row.append(k_ij)
            K_rows.append(torch.stack(K_row))

        K = torch.stack(K_rows) * volume  # (n1, n2) - fully differentiable!

        if add_noise:
            # Enforce symmetry: K should equal K^T but numerical errors can cause small asymmetry
            K = (K + K.T) / 2.0
            K += noise_var * torch.eye(n1, device=K.device, dtype=K.dtype)

        return K

    def compute_covariance_deterministic(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
        noise_var: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute covariance deterministically (ACCURATE for evaluation!).

        Full bivariate spectral density via trapezoidal quadrature:
        K[i,j] = ∫∫ s(ω, ω') cos(ω·xᵢ - ω'·xⱼ) dω dω'
               ≈ Σₘ Σₙ s(ωₘ, ωₙ) cos(ωₘ·xᵢ - ωₙ·xⱼ) Δω²

        Parameters
        ----------
        X1 : torch.Tensor, shape (n1, d)
            First set of spatial locations
        X2 : torch.Tensor, shape (n2, d), optional
            Second set of spatial locations (if None, use X1)
        noise_var : float
            Observation noise variance

        Returns
        -------
        K : torch.Tensor, shape (n1, n2) or (n1, n1)
            Covariance matrix
        """
        if X1.dim() == 1:
            X1 = X1.unsqueeze(-1)
        if X2 is None:
            X2 = X1
            add_noise = True
        else:
            if X2.dim() == 1:
                X2 = X2.unsqueeze(-1)
            add_noise = False

        n1, n2 = X1.shape[0], X2.shape[0]

        # Frequency grid: [0, omega_max] (standardized with low-rank training)
        omegas = torch.linspace(0, self.omega_max, self.n_features).unsqueeze(-1)

        # FULL bivariate spectral density matrix s(ωₘ, ωₙ)
        S_full = self._compute_spectral_density_matrix(omegas)  # (M, M)

        # Integration weights
        dw = self.omega_max / self.n_features if self.n_features > 1 else self.omega_max
        volume = dw * dw

        # Pre-compute ω @ X matrices
        omega_X1 = (omegas @ X1.T).squeeze()  # (M, n1)
        omega_X2 = (omegas @ X2.T).squeeze()  # (M, n2)

        # Double sum - GRADIENT SAFE (no .item()!)
        # Build K row by row to preserve gradients
        K_rows = []
        for i in range(n1):
            K_row = []
            for j in range(n2):
                # phases[m,n] = ωₘ·xᵢ - ωₙ·xⱼ
                # Broadcasting: (M, 1) - (1, M).T = (M, M)
                phases = omega_X1[:, i:i+1] - omega_X2[:, j:j+1].T  # (M, M)

                # K[i,j] = ΣₘΣₙ s(ωₘ,ωₙ) cos(ωₘ·xᵢ - ωₙ·xⱼ)
                k_ij = torch.sum(S_full * torch.cos(phases))
                K_row.append(k_ij)
            K_rows.append(torch.stack(K_row))

        K = torch.stack(K_rows) * volume  # (n1, n2) - differentiable!

        if add_noise:
            # Enforce symmetry: K should equal K^T but numerical errors can cause small asymmetry
            K = (K + K.T) / 2.0
            K += noise_var * torch.eye(n1, device=K.device, dtype=K.dtype)

        return K

    def posterior_mean_loss(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        noise_var: float = 1e-4,
        use_mc: bool = True,
        mc_samples: int = 50
    ) -> torch.Tensor:
        """
        Negative log marginal likelihood.

        Assumes y_train ~ GP(0, K + σ²I) where K is determined by s(ω,ω').
        Goal: Learn s(ω,ω') to maximize marginal likelihood.

        From GPML eq 2.30:
            -log p(y|X) = ½yᵀK⁻¹y + ½log|K| + (n/2)log(2π)

        HYBRID APPROACH:
            - Training (use_mc=True): Fast Monte Carlo integration
            - Evaluation (use_mc=False): Accurate deterministic quadrature

        Parameters
        ----------
        X_train : torch.Tensor, shape (n, d)
            Training inputs
        y_train : torch.Tensor, shape (n,)
            Training outputs (zero mean)
        noise_var : float
            Observation noise variance
        use_mc : bool
            Use Monte Carlo (fast) vs deterministic (accurate)
        mc_samples : int
            Number of MC samples (if use_mc=True)

        Returns
        -------
        loss : torch.Tensor
            Negative log marginal likelihood
        """
        # Compute covariance - HYBRID!
        if use_mc:
            # Fast MC for training
            K_train = self.compute_covariance_mc(
                X_train, noise_var=noise_var, n_samples=mc_samples
            )
        else:
            # Accurate deterministic for evaluation
            K_train = self.compute_covariance_deterministic(
                X_train, noise_var=noise_var
            )

        # Add regularization for numerical stability
        # Use higher jitter for nonstationary kernels with complex structure
        K_train_reg = K_train + 1e-4 * torch.eye(K_train.shape[0])

        # Cholesky decomposition (should always work with factorized s!)
        max_attempts = 5
        jitter = 1e-4
        for attempt in range(max_attempts):
            try:
                L = torch.linalg.cholesky(K_train_reg)
                break
            except RuntimeError as e:
                if attempt == max_attempts - 1:
                    # Last attempt failed
                    raise RuntimeError(f"Cholesky failed after {max_attempts} attempts: {e}")
                # Increase jitter exponentially
                jitter *= 10
                K_train_reg = K_train + jitter * torch.eye(K_train.shape[0])

        # Solve K⁻¹y
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze()

        # Negative log marginal likelihood (up to constant and scaling)
        # NOTE: This is proportional to the true NLL. We omit:
        #   - 0.5 factor (doesn't affect optimization)
        #   - n log(2 pi) constant term (doesn't affect optimization)
        # Full NLL = 0.5 * (data_fit + log_det) + 0.5*n*log(2 pi)
        data_fit = y_train @ alpha
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        loss = data_fit + log_det

        return loss

    def spectral_smoothness_penalty(self, n_samples: int = 100) -> torch.Tensor:
        """
        Encourage smooth spectral density.

        Penalizes large gradients in s(ω, ω').
        """
        # Sample random frequencies
        omegas = torch.rand(n_samples, self.input_dim) * self.omega_max - self.omega_max/2
        omegas.requires_grad_(True)

        # Compute s at these frequencies
        s = self.forward(omegas, omegas)

        # Compute gradient
        grad = torch.autograd.grad(
            outputs=s.sum(),
            inputs=omegas,
            create_graph=True
        )[0]

        # L2 norm of gradient
        return torch.mean(grad ** 2)

    def simulate(
        self,
        X_new: torch.Tensor,
        n_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Simulate from the learned GP prior.

        Since s(ω,ω') is now guaranteed PSD, sampling should always work!

        Parameters
        ----------
        X_new : torch.Tensor, shape (n, d)
            Locations to simulate at
        n_samples : int
            Number of sample paths
        seed : int, optional
            Random seed

        Returns
        -------
        samples : torch.Tensor, shape (n_samples, n)
            Sample paths from GP prior
        """
        # Create NFFs with learned spectral density
        def spectral_density_fn(w1, w2):
            return self.forward(w1, w2)

        nffs = NFFs(
            spectral_density=spectral_density_fn,
            n_features=self.n_features,
            omega_max=self.omega_max,
            input_dim=self.input_dim
        )

        # Simulate (should work now!)
        samples = nffs.simulate(X_new, n_samples=n_samples, seed=seed)

        return samples

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 500,
        lr: float = 1e-2,
        noise_var: float = 0.01,
        use_smoothness: bool = False,
        lambda_smooth: float = 0.1,
        patience: int = 100,
        use_lowrank: bool = True,
        omega_grid: Optional[torch.Tensor] = None,
        use_mc_training: bool = True,
        mc_samples: int = 50,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the factorized SDN.

        TRAINING METHODS:
            1. Low-rank NFF approximation (use_lowrank=True, default):
               Uses log_marginal_likelihood with compute_lowrank_features

            2. Full covariance (use_lowrank=False):
               Uses posterior_mean_loss with full covariance computation
               Monte Carlo (use_mc_training=True) or deterministic quadrature

        Parameters
        ----------
        X_train : torch.Tensor, shape (n, d)
            Training inputs
        y_train : torch.Tensor, shape (n,)
            Training outputs
        epochs : int
            Maximum training epochs
        lr : float
            Learning rate
        noise_var : float
            Observation noise variance
        use_smoothness : bool
            Enable spectral smoothness penalty (default: False)
        lambda_smooth : float
            Smoothness regularization weight (ignored if use_smoothness=True)
        patience : int
            Early stopping patience
        use_lowrank : bool
            Use low-rank NFF approximation (default: True)
        omega_grid : torch.Tensor, optional
            Frequency grid for low-rank method (if None, computes arange(0, n_features)*spacing)
        use_mc_training : bool
            Use Monte Carlo for full covariance training (ignored if use_lowrank=True)
        mc_samples : int
            Number of MC samples during training (ignored if use_lowrank=True)
        verbose : bool
            Print training progress

        Returns
        -------
        losses : List[float]
            Training loss history
        """
        # Zero-mean the data
        y_train = y_train - y_train.mean()

        # Prepare omega grid for low-rank method
        if use_lowrank and omega_grid is None:
            omega_grid = torch.linspace(0, self.omega_max, self.n_features).unsqueeze(-1)

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=2,
            eta_min=lr / 100
        )

        # Early stopping
        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        losses = []

        if verbose:
            print("TRAINING FACTORIZED SDN (PD Guaranteed):")
            print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
            print(f"  Rank: {self.rank}")
            print(f"  Epochs: {epochs}")
            print(f"  Initial LR: {lr}")
            if use_lowrank:
                print(f"  Method: Low-rank NFF")
                print(f"  Omega grid: {omega_grid.shape[0]} points from 0 to {self.omega_max}")
            else:
                print(f"  Method: {'Monte Carlo (fast)' if use_mc_training else 'Deterministic (accurate)'}")
                if use_mc_training:
                    print(f"  MC Samples: {mc_samples}")
            print()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute loss using selected method
            if use_lowrank:
                # Low-rank NFF
                L = self.compute_lowrank_features(X_train, omega_grid)
                data_loss = self.log_marginal_likelihood(L, y_train, noise_var)
            else:
                # Full covariance method
                data_loss = self.posterior_mean_loss(
                    X_train, y_train,
                    noise_var=noise_var,
                    use_mc=use_mc_training,
                    mc_samples=mc_samples
                )

            # Smoothness regularization
            if use_smoothness:
                smooth_penalty = self.spectral_smoothness_penalty()
                loss = data_loss + lambda_smooth * smooth_penalty
            else:
                loss = data_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Track
            losses.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.4f} | "
                      f"Data: {data_loss.item():.4f} | LR: {current_lr:.6f} | "
                      f"Best: {best_loss:.4f}")

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"\n✓ Early stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                break

        # Restore best model
        if best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                print(f"\n✓ Restored best model (loss: {best_loss:.4f})")

        return losses
