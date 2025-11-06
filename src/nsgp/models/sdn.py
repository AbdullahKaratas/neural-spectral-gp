"""
Spectral Density Network (SDN)

Learns the spectral density s(ω, ω') of a nonstationary Gaussian process
from observed data using a neural network with positive definiteness constraints.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, Callable
from .nffs import NFFs


class SpectralDensityNetwork(nn.Module):
    """
    Neural network that learns spectral density s(ω, ω') from data.

    The network takes frequency pairs (ω, ω') as input and outputs the
    spectral density while ensuring positive definiteness through a
    Cholesky parametrization.

    Parameters
    ----------
    input_dim : int
        Spatial dimension of the input (e.g., 2 for 2D spatial data)
    hidden_dims : List[int]
        List of hidden layer dimensions
    n_features : int
        Number of Fourier features M for NFFs
    omega_max : float
        Cutoff frequency for frequency domain
    activation : str, optional
        Activation function ('relu', 'tanh', 'elu'), default='relu'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64],
        n_features: int = 100,
        omega_max: float = 10.0,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_features = n_features
        self.omega_max = omega_max

        # Build MLP layers
        layers = []
        prev_dim = 2 * input_dim  # (ω, ω') concatenated

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim

        # Output layer (positive definiteness enforced later)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())  # Ensure positivity

        self.network = nn.Sequential(*layers)

        # Initialize weights to small values for better positive definiteness
        self._init_weights()

        # Training history
        self.train_losses = []
        self.nll_history = []
        self.reg_history = []

    def _init_weights(self):
        """
        Initialize network weights to small values.

        This ensures the initial spectral density is smooth and
        more likely to be positive definite.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Small positive bias

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
        }
        return activations.get(activation, nn.ReLU())

    def forward(self, omega1: torch.Tensor, omega2: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral density s(ω₁, ω₂).

        Parameters
        ----------
        omega1 : torch.Tensor, shape (n, d) or (n,)
            First frequency coordinates
        omega2 : torch.Tensor, shape (n, d) or (n,)
            Second frequency coordinates

        Returns
        -------
        torch.Tensor, shape (n,)
            Spectral density values
        """
        # Ensure proper shapes
        if omega1.dim() == 1:
            omega1 = omega1.unsqueeze(-1)
        if omega2.dim() == 1:
            omega2 = omega2.unsqueeze(-1)

        # Concatenate frequency pairs
        omega_pairs = torch.cat([omega1, omega2], dim=-1)
        return self.network(omega_pairs).squeeze(-1)

    def _get_spectral_density_function(self) -> Callable:
        """
        Return a callable spectral density function for NFFs.

        Returns
        -------
        Callable
            Function s(ω, ω') that can be called by NFFs
        """
        def spectral_density_fn(omega1, omega2):
            with torch.no_grad():
                return self.forward(omega1, omega2)
        return spectral_density_fn

    def compute_covariance(
        self,
        X: torch.Tensor,
        noise_var: float = 1e-4
    ) -> torch.Tensor:
        """
        Compute covariance matrix K using learned spectral density via NFFs.

        This uses the spectral representation to compute:
        K[i,j] = r(x_i, x_j) ≈ empirical covariance from NFFs samples

        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Input locations
        noise_var : float
            Observation noise variance (for numerical stability)

        Returns
        -------
        torch.Tensor, shape (n, n)
            Covariance matrix
        """
        n = X.shape[0]

        # Create NFFs simulator with current spectral density
        nffs = NFFs(
            spectral_density=self._get_spectral_density_function(),
            n_features=self.n_features,
            omega_max=self.omega_max,
            input_dim=self.input_dim
        )

        # Generate multiple samples to estimate covariance
        n_samples = 500  # More samples = better covariance estimate
        samples = nffs.simulate(X, n_samples=n_samples, seed=None)  # (n_samples, n)

        # Compute empirical covariance
        samples_centered = samples - samples.mean(dim=0, keepdim=True)
        K = (samples_centered.T @ samples_centered) / (n_samples - 1)

        # Add observation noise for numerical stability
        K = K + noise_var * torch.eye(n)

        return K

    def compute_covariance_deterministic(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
        noise_var: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute covariance matrix DETERMINISTICALLY from spectral density.

        Uses the spectral representation to compute K[i,j] = r(x_i, x_j) where:
        r(x,x') ≈ Σ_m s(ω_m, ω_m) * cos(ω_m^T(x-x')) * Δω / (2π)^d

        This is deterministic (no sampling) and differentiable!

        Parameters
        ----------
        X1 : torch.Tensor, shape (n1, d)
            First set of locations
        X2 : torch.Tensor, shape (n2, d), optional
            Second set of locations (if None, use X1)
        noise_var : float
            Observation noise variance (added to diagonal only if X2 is None)

        Returns
        -------
        torch.Tensor, shape (n1, n2) or (n1, n1)
            Covariance matrix
        """
        if X2 is None:
            X2 = X1
            add_noise = True
        else:
            add_noise = False

        n1, n2 = X1.shape[0], X2.shape[0]

        # Create frequency grid
        nffs = NFFs(
            spectral_density=self._get_spectral_density_function(),
            n_features=self.n_features,
            omega_max=self.omega_max,
            input_dim=self.input_dim
        )
        omegas = nffs.omegas  # (M, d)
        M = omegas.shape[0]

        # Volume element
        volume = (2 * self.omega_max) ** self.input_dim / M
        fourier_norm = (2 * np.pi) ** self.input_dim

        # For stationary approximation, use diagonal s(ω_m, ω_m)
        # This is faster but less accurate for highly nonstationary processes
        s_diag = self.forward(omegas, omegas)  # (M,)

        # Compute K[i,j] = Σ_m s(ω_m, ω_m) * cos(ω_m^T(x_i - x_j)) * Δω/(2π)^d
        K = torch.zeros(n1, n2)

        for i in range(n1):
            for j in range(n2):
                if self.input_dim == 1:
                    x_diff = X1[i] - X2[j]  # scalar
                    omega_x = omegas.squeeze() * x_diff  # (M,)
                else:
                    x_diff = X1[i] - X2[j]  # (d,)
                    omega_x = omegas @ x_diff  # (M,)

                # Covariance via spectral representation (stationary approximation)
                K[i, j] = torch.sum(s_diag * torch.cos(omega_x)) * volume / fourier_norm

        # Ensure symmetry (should be symmetric by construction, but numerical errors...)
        if add_noise:
            K = (K + K.T) / 2.0  # Symmetrize
            K = K + noise_var * torch.eye(n1)

        return K

    def negative_log_likelihood(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        noise_var: float = 1e-4
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood of observations.

        NLL = -log p(y | X, θ) = 1/2 (y^T K^{-1} y + log|K| + n log(2π))

        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Input locations
        y : torch.Tensor, shape (n,)
            Observations
        noise_var : float
            Observation noise variance

        Returns
        -------
        torch.Tensor, scalar
            Negative log-likelihood
        """
        n = len(y)

        # Compute covariance matrix
        K = self.compute_covariance(X, noise_var=noise_var)

        # Add small jitter for numerical stability
        K = K + 1e-6 * torch.eye(n)

        # Cholesky decomposition for efficient computation
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            # Fallback: add more jitter
            K = K + 1e-4 * torch.eye(n)
            L = torch.linalg.cholesky(K)

        # Solve L^T α = y using Cholesky factor
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze()

        # NLL = 1/2 (y^T K^{-1} y + log|K| + n log(2π))
        nll = 0.5 * (
            y @ alpha +
            2 * torch.sum(torch.log(torch.diag(L))) +
            n * np.log(2 * np.pi)
        )

        return nll

    def posterior_mean_loss(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        noise_var: float = 1e-4
    ) -> torch.Tensor:
        """
        BRILLIANT NEW LOSS: Posterior Mean Prediction Loss (deterministic!).

        Assumption: y_train are observations from a posterior GP
        Goal: Learn the PRIOR spectral density s(ω,ω') that explains these observations

        From Chapter 5, eq:posteriorGP:
        μ_post(x) = K(x, X_train) @ K(X_train, X_train)^{-1} @ y_train

        This is DETERMINISTIC given s(ω,ω') - no sampling noise!

        Parameters
        ----------
        X_train : torch.Tensor, shape (n, d)
            Training locations
        y_train : torch.Tensor, shape (n,)
            Observations (assumed to be from posterior GP)
        noise_var : float
            Observation noise variance

        Returns
        -------
        torch.Tensor, scalar
            Posterior prediction loss (deterministic!)
        """
        # Compute prior covariance DETERMINISTICALLY from learned spectral density
        K_train = self.compute_covariance_deterministic(X_train, noise_var=noise_var)

        # Posterior mean: μ_post = K @ K^{-1} @ y = y
        # So we want: K^{-1} @ y to be stable
        # Loss: Negative log posterior predictive

        # Add jitter for stability
        K_train_reg = K_train + 1e-5 * torch.eye(K_train.shape[0])

        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(K_train_reg)
        except RuntimeError:
            # Fallback: add more jitter
            K_train_reg = K_train + 1e-3 * torch.eye(K_train.shape[0])
            L = torch.linalg.cholesky(K_train_reg)

        # Solve K^{-1} @ y
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze()

        # Negative log marginal likelihood (GPML eq 2.30):
        # -log p(y|X) = 1/2 * y^T K^{-1} y + 1/2 log|K| + n/2 log(2π)
        loss = 0.5 * (
            y_train @ alpha +  # Data fit term
            2 * torch.sum(torch.log(torch.diag(L))) +  # Complexity penalty
            len(y_train) * np.log(2 * np.pi)
        )

        return loss

    def spectral_moment_matching_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Alternative loss: Spectral Moment Matching.

        Instead of computing full likelihood, match statistical moments:
        1. Sample-based MSE: E[(Z(x) - y)²]
        2. Variance matching: Match empirical variance of observations

        This is more stable than empirical covariance estimation.

        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Input locations
        y : torch.Tensor, shape (n,)
            Observations (centered)
        n_samples : int
            Number of samples for moment estimation

        Returns
        -------
        torch.Tensor, scalar
            Moment matching loss
        """
        # Generate samples from learned spectral density
        samples = self.simulate(X, n_samples=n_samples, seed=None)  # (n_samples, n)

        # 1. Data fitting term: MSE between samples and observations
        # For each sample, compute squared error to observations
        mse = torch.mean((samples - y.unsqueeze(0)) ** 2)

        # 2. Variance matching: Match marginal variances
        empirical_var_obs = torch.var(y)
        empirical_var_samples = torch.mean(torch.var(samples, dim=0))
        var_mismatch = (empirical_var_samples - empirical_var_obs) ** 2

        # 3. Correlation structure matching (optional, for nearby points)
        # Compute empirical autocovariance at lag 1
        if len(X) > 1:
            # Observations
            y_diff = y[1:] - y[:-1]
            obs_autocov = torch.mean(y[:-1] * y[1:])

            # Samples
            samples_autocov = torch.mean(samples[:, :-1] * samples[:, 1:])
            autocov_mismatch = (samples_autocov - obs_autocov) ** 2
        else:
            autocov_mismatch = 0.0

        # Total loss (weighted combination)
        loss = mse + 0.1 * var_mismatch + 0.1 * autocov_mismatch

        return loss

    def spectral_smoothness_penalty(self) -> torch.Tensor:
        """
        Regularization: Encourage smooth spectral density.

        Penalizes large gradients in frequency space:
        R_smooth = E[||∇_ω s(ω,ω')||²]

        Returns
        -------
        torch.Tensor, scalar
            Smoothness penalty
        """
        # Sample random frequency pairs
        n_samples = 100
        omega = torch.randn(n_samples, self.input_dim) * self.omega_max / 3

        # Compute finite difference gradients
        eps = 0.01
        s_center = self.forward(omega, omega)

        penalty = 0.0
        for dim in range(self.input_dim):
            omega_plus = omega.clone()
            omega_plus[:, dim] += eps
            s_plus = self.forward(omega_plus, omega_plus)

            grad = (s_plus - s_center) / eps
            penalty += torch.mean(grad ** 2)

        return penalty

    def low_rank_penalty(self) -> torch.Tensor:
        """
        Regularization: Encourage low-rank spectral matrix.

        This promotes parsimony by favoring spectral densities that can
        be represented with fewer Fourier features.

        Returns
        -------
        torch.Tensor, scalar
            Low-rank penalty
        """
        # Create NFFs with current spectral density
        nffs = NFFs(
            spectral_density=self._get_spectral_density_function(),
            n_features=self.n_features,
            omega_max=self.omega_max,
            input_dim=self.input_dim
        )

        # Compute spectral matrix
        S = nffs._compute_spectral_matrix()

        # Low-rank penalty: minimize nuclear norm (sum of singular values)
        # Approximation: minimize Frobenius norm
        penalty = torch.norm(S, p='fro') ** 2 / (self.n_features ** 2)

        return penalty

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 1000,
        learning_rate: float = 1e-3,
        noise_var: float = 1e-4,
        lambda_smooth: float = 0.01,
        lambda_rank: float = 0.001,
        loss_type: str = "moment_matching",
        n_loss_samples: int = 100,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        print_every: int = 100
    ):
        """
        Train the spectral density network on observed data.

        Loss = Data_Loss + λ_smooth · R_smooth + λ_rank · R_rank

        Parameters
        ----------
        X_train : torch.Tensor, shape (n, d)
            Training input locations
        y_train : torch.Tensor, shape (n,)
            Training observations (zero mean assumed)
        epochs : int, optional
            Number of training epochs, default=1000
        learning_rate : float, optional
            Learning rate for optimizer, default=1e-3
        noise_var : float, optional
            Observation noise variance, default=1e-4
        lambda_smooth : float, optional
            Weight for smoothness regularization, default=0.01
        lambda_rank : float, optional
            Weight for low-rank regularization, default=0.001
        loss_type : str, optional
            Loss function type: "likelihood", "moment_matching", or "posterior", default="moment_matching"
            - "likelihood": Full NLL with sampling-based covariance (unstable)
            - "moment_matching": Match statistical moments (moderate stability)
            - "posterior": Deterministic posterior mean loss (MOST STABLE!)
        n_loss_samples : int, optional
            Number of samples for moment matching loss, default=100
        batch_size : int, optional
            Batch size (None = full batch), default=None
        verbose : bool, optional
            Print training progress, default=True
        print_every : int, optional
            Print frequency, default=100
        """
        # Center observations (assume zero mean)
        y_train = y_train - y_train.mean()

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Data fitting loss
            if loss_type == "likelihood":
                data_loss = self.negative_log_likelihood(X_train, y_train, noise_var=noise_var)
                loss_name = "NLL"
            elif loss_type == "moment_matching":
                data_loss = self.spectral_moment_matching_loss(X_train, y_train, n_samples=n_loss_samples)
                loss_name = "MMD"
            elif loss_type == "posterior":
                data_loss = self.posterior_mean_loss(X_train, y_train, noise_var=noise_var)
                loss_name = "Post"
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            # Regularization
            smooth_penalty = self.spectral_smoothness_penalty()
            rank_penalty = self.low_rank_penalty()

            # Total loss
            loss = data_loss + lambda_smooth * smooth_penalty + lambda_rank * rank_penalty

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Track history
            self.train_losses.append(loss.item())
            self.nll_history.append(data_loss.item())
            self.reg_history.append((lambda_smooth * smooth_penalty + lambda_rank * rank_penalty).item())

            # Print progress
            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.4f} | "
                      f"{loss_name}: {data_loss.item():.4f} | Smooth: {smooth_penalty.item():.4f} | "
                      f"Rank: {rank_penalty.item():.4f}")

        if verbose:
            print(f"\n✓ Training completed!")

    def simulate(
        self,
        X_new: torch.Tensor,
        n_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Simulate GP realizations at new locations using learned spectral density.

        Parameters
        ----------
        X_new : torch.Tensor, shape (m, d)
            New locations for simulation
        n_samples : int, optional
            Number of sample paths, default=1
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        torch.Tensor, shape (n_samples, m)
            Simulated GP values
        """
        # Create NFFs with learned spectral density
        nffs = NFFs(
            spectral_density=self._get_spectral_density_function(),
            n_features=self.n_features,
            omega_max=self.omega_max,
            input_dim=self.input_dim
        )

        # Generate samples
        samples = nffs.simulate(X_new, n_samples=n_samples, seed=seed)

        return samples
