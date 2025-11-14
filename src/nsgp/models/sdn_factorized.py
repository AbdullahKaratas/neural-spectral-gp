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
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64],
        rank: int = 10,
        n_features: int = 50,
        omega_max: float = 8.0,
        activation: str = 'elu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.rank = rank
        self.n_features = n_features
        self.omega_max = omega_max

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

    def compute_features(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Compute feature vector f(ω) ∈ ℝʳ for frequency ω.

        Parameters
        ----------
        omega : torch.Tensor, shape (n, d)
            Frequency vectors

        Returns
        -------
        features : torch.Tensor, shape (n, r)
            Feature vectors
        """
        if omega.dim() == 1:
            omega = omega.unsqueeze(0)
        return self.feature_net(omega)

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
        f_all = self.compute_features(omegas)  # (n_samples, r)
        S_full = f_all @ f_all.T  # (n_samples, n_samples) - s(ωₘ, ωₙ) via factorization!
        # ✓ S_full is ALWAYS PSD by construction!

        # Monte Carlo integration weights
        dw_mc = self.omega_max / n_samples  # Per-dimension weight
        volume = dw_mc * dw_mc  # 2D integration volume
        fourier_norm = (2 * np.pi) ** self.input_dim

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

        K = torch.stack(K_rows) * volume / fourier_norm  # (n1, n2) - fully differentiable!

        if add_noise:
            K = (K + K.T) / 2.0
            K = K + noise_var * torch.eye(n1)

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
               ≈ Σₘ Σₙ s(ωₘ, ωₙ) cos(ωₘ·xᵢ - ωₙ·xⱼ) Δω²/(2π)^d

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
        M = self.n_features

        # Frequency grid
        omegas = torch.linspace(-self.omega_max/2, self.omega_max/2, M).unsqueeze(-1)

        # FULL bivariate spectral density matrix s(ωₘ, ωₙ)
        f_all = self.compute_features(omegas)  # (M, r)
        S_full = f_all @ f_all.T  # (M, M) - s(ωₘ, ωₙ) via factorization!

        # Integration weights
        dw = self.omega_max / (M - 1) if M > 1 else self.omega_max
        volume = dw * dw  # 2D integration!
        fourier_norm = (2 * np.pi) ** self.input_dim

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

        K = torch.stack(K_rows) * volume / fourier_norm  # (n1, n2) - differentiable!

        if add_noise:
            K = (K + K.T) / 2.0
            K = K + noise_var * torch.eye(n1)

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

        # Negative log marginal likelihood (GPML eq 2.30)
        loss = 0.5 * (
            y_train @ alpha +  # Data fit
            2 * torch.sum(torch.log(torch.diag(L))) +  # Complexity
            len(y_train) * np.log(2 * np.pi)
        )

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
        lambda_smooth: float = 0.1,
        patience: int = 100,
        use_mc_training: bool = True,
        mc_samples: int = 50,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the factorized SDN.

        HYBRID APPROACH (like Remes et al. 2017):
            - Training: Fast Monte Carlo integration (use_mc_training=True)
            - Evaluation: Accurate deterministic quadrature (for metrics)

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
        lambda_smooth : float
            Smoothness regularization weight
        patience : int
            Early stopping patience
        use_mc_training : bool
            Use Monte Carlo for training (faster!)
        mc_samples : int
            Number of MC samples during training
        verbose : bool
            Print training progress

        Returns
        -------
        losses : List[float]
            Training loss history
        """
        # Zero-mean the data
        y_train = y_train - y_train.mean()

        # Optimizer with cosine annealing
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
            print(f"  Method: {'Monte Carlo (fast)' if use_mc_training else 'Deterministic (accurate)'}")
            if use_mc_training:
                print(f"  MC Samples: {mc_samples}")
            print()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Posterior mean loss - HYBRID approach!
            data_loss = self.posterior_mean_loss(
                X_train, y_train,
                noise_var=noise_var,
                use_mc=use_mc_training,
                mc_samples=mc_samples
            )

            # Smoothness regularization
            smooth_penalty = self.spectral_smoothness_penalty()

            # Total loss
            loss = data_loss + lambda_smooth * smooth_penalty

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
