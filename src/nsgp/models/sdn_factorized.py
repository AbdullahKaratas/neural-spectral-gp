"""
Factorized Spectral Density Network (SDN-F)

This version guarantees positive semi-definiteness by using a low-rank factorization:
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
        enforce_symmetry: bool = True,
        omega_grid: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.rank = rank
        self.n_features = n_features
        self.omega_max = omega_max
        self.enforce_symmetry = enforce_symmetry

        # Frequency grid for low-rank NFF (optional, created if None)
        if omega_grid is None:
            self.omega_grid = torch.linspace(0, omega_max, n_features).unsqueeze(-1)
        else:
            self.omega_grid = omega_grid

        # Learnable global scale (log variance)
        # Initialize to -2.0 for moderate initial scale (exp(-2) ≈ 0.135)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))

        # MLP: ω → feature vector f(ω) ∈ ℝʳ
        # This is the core of the low-rank factorization
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim

        # Output: r-dimensional feature vector
        layers.append(nn.Linear(prev_dim, rank))

        # Add final activation to bound features and prevent explosion
        # Tanh bounds to [-1, 1], helping with stable training
        layers.append(nn.Tanh())

        self.feature_net = nn.Sequential(*layers)

        # Initialize with Xavier (better than std=0.01)
        self._init_weights()

    def _init_weights(self):
        """Initialize with Xavier uniform for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization - good default for tanh/sigmoid activations
                # gain=1.0 for tanh (default)
                nn.init.xavier_uniform_(m.weight, gain=1.0)
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
            
        if torch.isnan(f).any():
             print("compute_features produced NaNs!")
             print(f"omega stats: {omega.min()}/{omega.max()}")

        return f

    def _compute_spectral_density_matrix(self) -> torch.Tensor:
        """
        Compute spectral density matrix S[m,n] = s(omega_m, omega_n).

        Uses factorized representation: s(omega, omega') = f(omega)^T f(omega')
        This guarantees S is positive semi-definite.

        Returns
        -------
        S : torch.Tensor, shape (M, M)
            Spectral density matrix
        """
        f = self.compute_features(self.omega_grid)  # (M, r)
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
    ) -> torch.Tensor:
        r"""
        Compute low-rank feature matrix L using nonstationary Fourier features.

        This computes K = LL^T where K = B S^{1/2} (S^{1/2})^T B^T
        - B[i,m] = cos(omega_m x_i) is the cosine basis
        - S[m,n] = s(omega_m, omega_n) \Delta omega^2 is the spectral process kernel
        - S^{1/2} is the matrix square root of S

        Mathematical Background
        -----------------------
        Computes K = LL^T via bivariate spectral representation:
            k(x,x') = ∫∫ s(ω,ω') cos(ωx - ω'x') dω dω'

        With low-rank factorization s(ω,ω') = f(ω)^T f(ω'), this becomes:
            L = B @ S^{1/2}  where B[i,m] = cos(ω_m x_i), S = F F^T

        See paper Section 3 for full mathematical derivation.

        Frequency Grid Constraint
        -------------------------
        Grid spacing Δω must satisfy: π/Δω ≥ n·Δx (conservative bound)
        to avoid periodicity artifacts. See paper for details.

        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Spatial locations

        Returns
        -------
        L : torch.Tensor, shape (n, num_freqs)
            Low-rank feature matrix where K = LL^T
        """
        # Input validation
        if not torch.is_floating_point(X):
            raise TypeError(f"X must be floating point tensor, got {X.dtype}")

        if not torch.is_floating_point(self.omega_grid):
            raise TypeError(f"omega_grid must be floating point tensor, got {self.omega_grid.dtype}")

        num_freqs = self.omega_grid.shape[0]
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

        spacing = torch.norm(self.omega_grid[1] - self.omega_grid[0]).item()
        constraint_lhs = np.pi / spacing
        constraint_rhs = n_pts * delta_x

        if constraint_lhs < constraint_rhs:
            import warnings
            warnings.warn(
                f"Frequency grid may be too coarse: pi/spacing = {constraint_lhs:.4f} < n*delta_x = {constraint_rhs:.4f}. "
                f"Consider using at least {int(np.ceil(self.omega_grid.max().item() / (np.pi / constraint_rhs)))+1} frequency points."
            )

        # Compute low rank features * spacing
        S_sqrt = self.compute_features(self.omega_grid) * spacing

        # Compute cosine basis
        # X: (n, d), self.omega_grid: (num_freqs, d) -> phases: (n, num_freqs)
        phases = X @ self.omega_grid.T  # (n, num_freqs)
        B_cos = torch.cos(phases)  # (n, num_freqs)

        # Correction for zero frequency (Trapezoidal rule boundary)
        # At ω=0, the weight should be 0.5 * dω (trapezoidal rule for boundary points).
        # Since K ~ L*L^T, multiplying B by 0.5 results in 0.25 weight for the (0,0) corner term in 2D integration.
        # This helps the network learn a smooth f(ω) without needing to learn a discontinuity at 0.
        # Note: We don't apply 0.5 at ω_max because spectral density → 0 there (negligible contribution).
        omega_norms = torch.norm(self.omega_grid, dim=1)  # (num_freqs,)
        is_zero = omega_norms < 1e-10
        if torch.any(is_zero):
            B_cos[:, is_zero] = 0.5

        # Compute low-rank features
        # S already includes (Δω)² scaling
        #
        # LOW-RANK KERNEL APPROXIMATION:
        # k(x,x') = ∫∫ s(ω,ω') cos(ωx) cos(ω'x') dω dω'
        #         ≈ Σ_m Σ_n s(ω_m, ω_n) cos(ω_m x) cos(ω_n x') Δω²
        #
        # With s(ω,ω') = f(ω)^T f(ω') and S = FF^T:
        # k(x,x') ≈ [B @ F @ Δω] @ [B @ F @ Δω]^T
        #         = [B @ S^{1/2} @ Δω] @ [B @ S^{1/2} @ Δω]^T
        #         = L @ L^T
        #
        # Since S already contains Δω² scaling (line 418), we have:
        L = B_cos @ S_sqrt  # (n, num_freqs)

        # Apply learnable scale: L_scaled = sqrt(theta) * L
        L *= torch.exp(0.5 * self.log_scale)  # Inplace operation

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
        MIN_SIGMA2 = 1e-8  # Numerical stability threshold

        if sigma2 <= 0:
            raise ValueError(f"sigma2 must be positive, got {sigma2}")

        if sigma2 < MIN_SIGMA2:
            raise ValueError(
                f"sigma2={sigma2:.2e} is too small for numerical stability. "
                f"Minimum allowed: {MIN_SIGMA2:.2e}. "
                f"Division by sigma2 would cause overflow (1/sigma2={1/sigma2:.2e})."
            )

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
        log_det_sigma = (n - r) * torch.log(torch.as_tensor(sigma2))
        log_det_W = 2 * torch.sum(torch.log(torch.diag(Lw)))  # log|W| = 2·sum(log(diag(Lw)))
        log_det = log_det_sigma + log_det_W

        # Negative log marginal likelihood (up to constant and scaling)
        # NOTE: This is proportional to the true NLL. We omit:
        #   - 0.5 factor (doesn't affect optimization)
        #   - n log(2 pi) constant term (doesn't affect optimization)
        # Full NLL = 0.5 * (data_fit + log_det) + 0.5*n*log(2 pi)
        nll = data_fit + log_det

        return nll

    def compute_covariance(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if X1.dim() == 1:
            X1 = X1.unsqueeze(-1)

        L1 = self.compute_lowrank_features(X1)

        if X2 is None:
            return L1 @ L1.T
        else:
            if X2.dim() == 1:
                X2 = X2.unsqueeze(-1)
            L2 = self.compute_lowrank_features(X2)

        return L1 @ L2.T

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

    def spectral_diversity_penalty(self) -> torch.Tensor:
        """
        Encourage diverse spectral structure (prevent rank collapse).

        Uses eigenvalue entropy to ensure S has multiple significant eigenvalues
        instead of collapsing to rank-1 (spectral collapse).

        High entropy = diverse eigenvalues = good ✓
        Low entropy = rank collapse = bad ✗

        Returns
        -------
        penalty : torch.Tensor
            Negative entropy (minimize to maximize diversity)
        """
        S = self._compute_spectral_density_matrix()

        # Eigenvalue decomposition
        eigenvalues = torch.linalg.eigvalsh(S)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)  # Numerical stability

        # Normalize to probability distribution
        probs = eigenvalues / eigenvalues.sum()

        # Shannon entropy: H = -Σ pᵢ log(pᵢ)
        # Higher entropy = more diverse eigenvalues
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        # Normalize by max possible entropy (uniform distribution)
        max_entropy = torch.log(torch.tensor(len(eigenvalues), dtype=torch.float32))
        normalized_entropy = entropy / max_entropy

        # Return negative (we minimize loss, but want to maximize entropy)
        # Also subtract from 1 so penalty is positive when diversity is low
        return 1.0 - normalized_entropy

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
        use_diversity: bool = True,
        lambda_diversity: float = 0.1,
        patience: int = 100,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the factorized SDN using low-rank NFF approximation.

        Uses log_marginal_likelihood with compute_lowrank_features.

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
            Smoothness regularization weight (ignored if use_smoothness=False)
        use_diversity : bool
            Enable spectral diversity penalty to prevent rank collapse (default: True)
        lambda_diversity : float
            Diversity regularization weight (default: 0.1)
        patience : int
            Early stopping patience
        verbose : bool
            Print training progress

        Returns
        -------
        losses : List[float]
            Training loss history
        """
        # Zero-mean the data
        y_train = y_train - y_train.mean()

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
            print(f"  Method: Low-rank NFF")
            print(f"  Omega grid: {self.omega_grid.shape[0]} points from 0 to {self.omega_max}")
            print()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute loss using low-rank NFF
            L = self.compute_lowrank_features(X_train)
            data_loss = self.log_marginal_likelihood(L, y_train, noise_var)

            # Regularization
            loss = data_loss

            if use_smoothness:
                smooth_penalty = self.spectral_smoothness_penalty()
                loss = loss + lambda_smooth * smooth_penalty

            if use_diversity:
                diversity_penalty = self.spectral_diversity_penalty()
                loss = loss + lambda_diversity * diversity_penalty

            if torch.isnan(loss):
                if verbose:
                    print(f"Warning: Loss is NaN at epoch {epoch}. Skipping update.")
                optimizer.zero_grad()
                continue

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
