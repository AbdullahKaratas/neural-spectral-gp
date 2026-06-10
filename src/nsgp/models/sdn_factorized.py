import math
import warnings

import torch
import torch.nn as nn
import gpytorch
import numpy as np
from typing import Optional, List

from linear_operator.operators import (
    DenseLinearOperator,
    DiagLinearOperator,
    LowRankRootLinearOperator,
)


class FactorizedSpectralDensityNetwork(nn.Module):
    """
    Factorized Spectral Density Network (F-SDN)

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
        Activation function ('relu', 'elu', 'tanh'). Default: 'relu'
    enforce_symmetry : bool
        If True, enforce f(omega) = f(-omega). Default: False.
    spectral_real : bool
        If True, f(omega) is real-valued. If False, f(omega) is complex-valued.
        Incompatible with enforce_symmetry=True when False. (default=True)
    embedding_dim : int
        Number of Fourier embeddings (Tancik et al., 2020). 0 disables the
        embedding. Default: 0.
    embedding_scale : float
        Std of B sim N(0, embedding_scale^2) (Tancik et al., 2020).
        Default: 1.0.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64],
        rank: int = 10,
        n_features: int = 50,
        omega_max: float = 8.0,
        activation: str = "relu",
        enforce_symmetry: bool = False,
        spectral_real: bool = True,
        learn_log_scale: bool = True,
        prior_variance: Optional[float] = None,
        embedding_dim: int = 0,
        embedding_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.prior_variance = prior_variance
        self.hidden_dims = hidden_dims
        self.rank = rank
        self._n_features_raw = n_features
        self.omega_max = omega_max
        self.enforce_symmetry = enforce_symmetry
        self.learn_log_scale = learn_log_scale
        self.spectral_real = spectral_real
        self.activation = activation

        if not spectral_real and enforce_symmetry:
            raise ValueError(
                "spectral_real=False is incompatible with enforce_symmetry=True."
            )

        # Frequency grid for low-rank NFF
        if enforce_symmetry:
            self.n_features = self._n_features_raw
            spacing = omega_max / self.n_features
            self.register_buffer(
                "omega_grid",
                torch.arange(0, self.n_features)
                .reshape(-1, 1)
                .to(torch.get_default_dtype())
                * spacing,
            )
        else:
            k = (self._n_features_raw + 1) // 2
            spacing = omega_max / float(k)
            self.register_buffer(
                "omega_grid",
                torch.arange(-k + 1, k).reshape(-1, 1).to(torch.get_default_dtype())
                * spacing,
            )
            self.n_features = 2 * k - 1

        # Learnable global scale (log variance)
        # Initialize to 0.0 for unit signal variance (exp(0) = 1.0)
        # when targets are standardized
        if learn_log_scale:
            self.log_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("log_scale", torch.tensor(0.0))

        # Learnable noise variance (log scale for numerical stability)
        # Initialize to log(0.5^2) for noise_std = 0.5
        self.log_noise_var = nn.Parameter(torch.tensor(math.log(0.25)))

        # Fourier embedding for high frequency learning (Tancik et al, 2020)
        self.embedding_dim = embedding_dim
        if embedding_dim > 0:
            self.register_buffer(
                "B", torch.randn(input_dim, embedding_dim) * embedding_scale
            )
            mlp_input_dim = 2 * embedding_dim
        else:
            mlp_input_dim = input_dim

        output_dim = rank if spectral_real else 2 * rank
        self.feature_net = self._build_mlp(
            mlp_input_dim, output_dim, hidden_dims, activation
        )
        self.best_loss = None
        self.X_train = None
        self.y_train = None

        # Initialize with Xavier (better than std=0.01)
        self._init_weights()

    def _build_mlp(
        self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: str
    ) -> nn.Sequential:
        """Build an MLP network."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _init_weights(self):
        act = self.activation

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if act in ("relu", "elu"):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                elif act == "tanh":
                    nn.init.xavier_uniform_(
                        m.weight, gain=nn.init.calculate_gain("tanh")
                    )
                else:
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last = [m for m in self.feature_net.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            last.weight.mul_(0.05)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        return {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
        }.get(activation, nn.ELU())

    def log_prior(self) -> torch.Tensor:
        """Gaussian prior on NN weights: log p(W) = -0.5/prior_variance * ||W||^2."""
        weights = torch.cat(
            [
                p.view(-1)
                for name, p in self.feature_net.named_parameters()
                if "weight" in name
            ]
        )
        return -0.5 * weights.norm().pow(2) / self.prior_variance

    def _embed(self, omega: torch.Tensor) -> torch.Tensor:
        """Fourier embedding."""

        if self.embedding_dim > 0:
            proj = (omega / self.omega_max) @ self.B
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return omega / self.omega_max

    def compute_features(self, omega: torch.Tensor) -> torch.Tensor:
        r"""
        Compute feature vector f(omega).

        If enforce_symmetry=True:
            Enforces f(omega) = f(-omega)

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

        if self.enforce_symmetry:
            # Symmetrize: f(omega) = [tilde{f}(omega) + tilde{f}(-omega)] / 2
            f = (
                self.feature_net(self._embed(omega))
                + self.feature_net(self._embed(-omega))
            ) / 2.0
        else:
            f = self.feature_net(self._embed(omega))

        if not self.spectral_real:
            f_re, f_im = f.chunk(2, dim=-1)
            f = torch.complex(f_re, f_im)

        if torch.isnan(f).any():
            warnings.warn("compute_features produced NaNs!")

        return f

    def compute_lowrank_features(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute low-rank feature matrix L using nonstationary Fourier features.

        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Spatial locations

        Returns
        -------
        L : torch.Tensor, shape (n, r) if enforce_symmetry=True, else (n, 4r)
            Low-rank feature matrix where K = LL^T
        """
        # Input validation
        if X.shape[1] != 1:
            raise NotImplementedError(
                "compute_lowrank_features only supports 1D inputs (X.shape[1] == 1)."
            )

        if not torch.is_floating_point(X):
            raise TypeError(f"X must be floating point tensor, got {X.dtype}")

        if not torch.is_floating_point(self.omega_grid):
            raise TypeError(
                f"omega_grid must be floating point tensor, got {self.omega_grid.dtype}"
            )

        num_freqs = self.omega_grid.shape[0]
        n_pts = X.shape[0]

        # Validate minimum requirements
        if num_freqs < 2:
            raise ValueError(
                f"Frequency grid must contain at least 2 points, "
                f"got {num_freqs}. Low-rank approximation requires "
                f"multiple frequency samples to compute spacing."
            )

        if n_pts < 2:
            raise ValueError(
                f"Spatial locations X must contain at least 2 points, "
                f"got {n_pts}. Low-rank approximation requires multiple "
                f"spatial points to compute spacing."
            )

        # Compute minimal spacing: \Delta x = min_j(x_{j+1} - x_j)
        X_sorted = torch.sort(X.squeeze(), dim=0)[0]
        spacings = X_sorted[1:] - X_sorted[:-1]
        delta_x = spacings.min().item()

        # Grid spacing must satisfy aliasing
        spacing = torch.norm(self.omega_grid[1] - self.omega_grid[0]).item()
        constraint_lhs = np.pi / spacing
        constraint_rhs = n_pts * delta_x

        if constraint_lhs < constraint_rhs:
            rec = (
                int(np.ceil(self.omega_grid.max().item() / (np.pi / constraint_rhs)))
                + 1
            )
            warnings.warn(
                f"Frequency grid may be too coarse: pi/spacing = "
                f"{constraint_lhs:.4f} < n*delta_x = {constraint_rhs:.4f}. "
                f"Consider using at least {rec} frequency points."
            )

        # Compute low rank features
        # Paper: [F]_{kj} = conj(f_j(omega_k)); no-op when spectral_real=True
        F_pos = self.compute_features(self.omega_grid).conj()  # (num_freqs, r)
        if not self.enforce_symmetry:
            F_neg = self.compute_features(-self.omega_grid)  # (num_freqs, r)

        # Compute cosine basis
        # X: (n, d), self.omega_grid: (num_freqs, d) -> phases: (n, num_freqs)
        phases = X @ self.omega_grid.T  # (n, num_freqs)
        B_cos = torch.cos(phases)  # (n, num_freqs)
        if not self.enforce_symmetry:
            B_sin = torch.sin(phases)  # (n, num_freqs)

        # Correction for zero frequency
        omega_norms = torch.norm(self.omega_grid, dim=1)  # (num_freqs,)
        is_zero = omega_norms < 1e-10
        if torch.any(is_zero):
            B_cos[:, is_zero] = 0.5
            if not self.enforce_symmetry:
                B_sin[:, is_zero] = 0.0

        if self.spectral_real:
            psi_real = B_cos @ F_pos * spacing
            if self.enforce_symmetry:
                L = math.sqrt(2.0) * psi_real
            else:
                psi_imag = B_sin @ F_pos * spacing
                psi_neg_real = B_cos @ F_neg * spacing
                psi_neg_imag = B_sin @ F_neg * spacing
                L = math.sqrt(2.0) * torch.cat(
                    [psi_real, psi_imag, psi_neg_real, psi_neg_imag], dim=-1
                )
        else:
            psi_real = torch.cat(
                [
                    B_cos @ F_pos.real - B_sin @ F_pos.imag,
                    B_cos @ F_neg.real - B_sin @ F_neg.imag,
                ],
                dim=-1,
            )
            psi_imag = torch.cat(
                [
                    B_cos @ F_pos.imag + B_sin @ F_pos.real,
                    B_cos @ F_neg.imag + B_sin @ F_neg.real,
                ],
                dim=-1,
            )

            L = math.sqrt(2.0) * torch.cat([psi_real, psi_imag], dim=-1) * spacing

        # Apply learnable scale: L_scaled = sqrt(theta) * L
        L *= torch.exp(0.5 * self.log_scale)

        return L

    def log_marginal_likelihood(
        self, L: torch.Tensor, y: torch.Tensor, sigma2: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute negative marginal likelihood using low-rank NFF approximation.

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
            raise ValueError(
                f"Shape mismatch: L has {n} rows but y has {y.shape[0]} elements"
            )

        if L.device != y.device:
            raise ValueError(
                f"L and y must be on same device, got L on {L.device} "
                f"and y on {y.device}"
            )

        if r > n:
            warnings.warn(
                f"Rank r={r} exceeds number of data points n={n}. "
                f"Low-rank approximation is inefficient in this regime. "
                f"Consider r <= n."
            )

        # Woodbury and log determinant formula with linear_operator
        covar = LowRankRootLinearOperator(L) + DiagLinearOperator(
            sigma2 * torch.ones(n, device=L.device, dtype=L.dtype)
        )
        data_fit, log_det = covar.inv_quad_logdet(
            inv_quad_rhs=y.unsqueeze(-1), logdet=True
        )

        # Negative log marginal likelihood
        nll = 0.5 * (data_fit + log_det + n * math.log(2.0 * math.pi))

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

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 500,
        lr: float = 1e-2,
        patience: Optional[int] = None,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the factorized SDN using low-rank NFF approximation.

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
        patience : int, optional
            Early stopping patience. If None, no early stopping (default: None)
        verbose : bool
            Print training progress

        Returns
        -------
        losses : List[float]
            Training loss history
        """
        self.X_train = X_train
        self.y_train = y_train

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, amsgrad=True)

        # Early stopping and best state
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        if patience is None:
            patience = epochs

        losses = []

        if verbose:
            print("TRAINING:")
            print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
            print(f"  Rank: {self.rank}")
            print(f"  Epochs: {epochs}")
            print(f"  Initial LR: {lr}")
            print(f"  Features: {self.omega_grid.shape[0]}")
            print()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute loss using low-rank NFF
            L = self.compute_lowrank_features(X_train)
            noise_var = torch.exp(self.log_noise_var)
            data_loss = self.log_marginal_likelihood(L, y_train, noise_var)

            # MAP (with prior) or MLE (without)
            loss = data_loss
            if self.prior_variance is not None:
                loss = loss - self.log_prior()

            if torch.isnan(loss):
                if verbose:
                    print(f"Warning: Loss is NaN at epoch {epoch}. Skipping update.")
                optimizer.zero_grad()
                continue

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            # Track
            losses.append(loss.item())
            optimizer.param_groups[0]["lr"]

            # Early stopping and best state tracking
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.4f} | "
                    f"Best: {best_loss:.4f}"
                )

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)"
                    )
                break

        # Restore best model and store best loss
        if best_state is not None:
            self.load_state_dict(best_state)
            self.best_loss = best_loss
            if verbose:
                print(f"Restored best model (loss: {best_loss:.4f})")
        else:
            self.best_loss = loss.item()

        return losses

    def _full_pred_dist(
        self,
        X_test: torch.Tensor,
        predictive_dist: bool = True,
        diag: bool = False,
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Return the posterior distribution.

        Parameters
        ----------
        X_test : torch.Tensor, shape (t, d)
            Test locations.
        predictive_dist : bool
            If True, include observation noise.
        diag : bool
            If True, computes only the diagonal covariance.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
        """
        if self.X_train is None:
            raise RuntimeError("Model not fitted yet.")

        with torch.no_grad():
            noise = torch.exp(self.log_noise_var)

            L = self.compute_lowrank_features(self.X_train)
            _, rank_4r = L.shape

            M = DenseLinearOperator(L.mT @ L) + DiagLinearOperator(
                noise * torch.ones(rank_4r, dtype=L.dtype, device=L.device)
            )
            L_star = self.compute_lowrank_features(X_test)

            # Posterior mean
            beta = M.solve(L.mT @ self.y_train.unsqueeze(-1))
            mean = (L_star @ beta).squeeze(-1)

            # Posterior covar
            if diag:
                var = noise * (L_star * M.solve(L_star.mT).mT).sum(dim=1)
                if predictive_dist:
                    var = var + noise
                return gpytorch.distributions.MultivariateNormal(
                    mean, DiagLinearOperator(var)
                )
            else:
                covar = noise * (L_star @ M.solve(L_star.mT))
                if predictive_dist:
                    covar = covar + noise * torch.eye(
                        L_star.shape[0], dtype=L_star.dtype, device=L_star.device
                    )
                return gpytorch.distributions.MultivariateNormal(mean, covar)

    def predict(self, X_test, predictive_dist=True):
        """
        Posterior using Low-rank approximation.
        """
        pred = self._full_pred_dist(X_test, predictive_dist=predictive_dist, diag=True)
        var = torch.clamp(pred.variance, min=1e-6)
        return pred.mean, torch.sqrt(var)
