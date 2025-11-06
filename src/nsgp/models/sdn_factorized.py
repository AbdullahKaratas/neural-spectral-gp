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
        """Initialize to small random values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
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
        s = torch.sum(f1 * f2, dim=-1)  # (n,) or (n, m) if broadcasting

        # Add small positive constant for numerical stability
        s = s + 1e-6

        return s

    def compute_covariance_deterministic(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
        noise_var: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute covariance K(X1, X2) deterministically using inverse Fourier transform.

        K[i,j] = ∫∫ s(ω, ω') exp(iω·xᵢ - iω'·xⱼ) dω dω'
               ≈ Σₘ s(ωₘ, ωₘ) cos(ωₘ·(xᵢ-xⱼ)) Δω/(2π)ᵈ

        Parameters
        ----------
        X1 : torch.Tensor, shape (n1, d)
            First set of spatial locations
        X2 : torch.Tensor, shape (n2, d), optional
            Second set of spatial locations (if None, use X1)
        noise_var : float
            Observation noise variance (added to diagonal if X2 is None)

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

        # Generate frequency grid
        M = self.n_features
        omegas = torch.linspace(-self.omega_max/2, self.omega_max/2, M).unsqueeze(-1)

        # Compute spectral density at diagonal s(ωₘ, ωₘ)
        s_diag = self.forward(omegas, omegas)  # (M,)

        # Volume element and Fourier normalization
        volume = (self.omega_max) ** self.input_dim / M
        fourier_norm = (2 * np.pi) ** self.input_dim

        # Initialize covariance
        K = torch.zeros(n1, n2)

        # Compute K[i,j] = Σₘ s(ωₘ, ωₘ) cos(ωₘ·(xᵢ-xⱼ)) Δω/(2π)ᵈ
        for i in range(n1):
            for j in range(n2):
                omega_x = omegas @ (X1[i] - X2[j])
                K[i, j] = torch.sum(s_diag * torch.cos(omega_x)) * volume / fourier_norm

        # Symmetrize and add noise if needed
        if add_noise:
            K = (K + K.T) / 2.0
            K = K + noise_var * torch.eye(n1)

        return K

    def posterior_mean_loss(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        noise_var: float = 1e-4
    ) -> torch.Tensor:
        """
        Negative log marginal likelihood (deterministic posterior loss).

        Assumes y_train ~ GP(0, K + σ²I) where K is determined by s(ω,ω').
        Goal: Learn s(ω,ω') to maximize marginal likelihood.

        From GPML eq 2.30:
            -log p(y|X) = ½yᵀK⁻¹y + ½log|K| + (n/2)log(2π)

        Parameters
        ----------
        X_train : torch.Tensor, shape (n, d)
            Training inputs
        y_train : torch.Tensor, shape (n,)
            Training outputs (zero mean)
        noise_var : float
            Observation noise variance

        Returns
        -------
        loss : torch.Tensor
            Negative log marginal likelihood
        """
        # Compute covariance
        K_train = self.compute_covariance_deterministic(X_train, noise_var=noise_var)

        # Add regularization for numerical stability
        K_train_reg = K_train + 1e-5 * torch.eye(K_train.shape[0])

        # Cholesky decomposition (should always work now!)
        try:
            L = torch.linalg.cholesky(K_train_reg)
        except RuntimeError as e:
            # This should never happen with factorized s, but just in case...
            print(f"  ⚠️ Cholesky failed (this shouldn't happen!): {e}")
            # Add more jitter
            K_train_reg = K_train_reg + 1e-3 * torch.eye(K_train_reg.shape[0])
            L = torch.linalg.cholesky(K_train_reg)

        # Solve K⁻¹y
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze()

        # Negative log marginal likelihood
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
        verbose: bool = True
    ) -> List[float]:
        """
        Train the factorized SDN.

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
            print()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Posterior mean loss (deterministic)
            data_loss = self.posterior_mean_loss(X_train, y_train, noise_var=noise_var)

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
