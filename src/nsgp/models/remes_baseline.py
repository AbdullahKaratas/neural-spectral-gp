"""
Remes et al. 2017 Baseline: Non-Stationary Spectral Kernels

This is a PyTorch port of the "Neural Non-Stationary Spectral Kernel" from:
    Remes, Heinonen, Kaski (2017). "Non-Stationary Spectral Kernels". NIPS 2017.

Key Difference from F-SDN:
    - Remes: Direct kernel computation K = WW * E * COS
    - F-SDN: Factorized spectral density s(ω,ω') = f(ω)^T f(ω')

PD Guarantee:
    - Remes: NO - must check eigenvalues post-hoc (can fail!)
    - F-SDN: YES - always PSD by construction

This implementation is designed to DEMONSTRATE when Remes fails PD,
which is our key contribution!

Authors: Abdullah Karatas, Arsalan Jawaid
Reference: https://github.com/sremes/nssm-gp (GPflow implementation)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple


class RemesNeuralSpectralKernel(nn.Module):
    """
    Remes et al. 2017 Neural Spectral Kernel baseline.

    Architecture:
        Three neural networks predict input-dependent parameters:
        - freq(x): Frequencies (d-dimensional)
        - len(x): Lengthscales (d-dimensional)
        - var(x): Variance (scalar)

    Kernel computation:
        K[i,j] = w(xi)*w(xj) * exp_term(xi,xj) * cos_term(xi,xj)

    PD Issue:
        K is NOT guaranteed to be PSD! Must check eigenvalues and add jitter.
        This can FAIL during training, which is our key message!

    Parameters
    ----------
    input_dim : int
        Spatial dimension
    hidden_dims : List[int]
        Hidden layer sizes for MLPs
    n_components : int
        Number of spectral mixture components (Q in Remes paper)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [32, 32],
        n_components: int = 1,
        activation: str = 'selu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_components = n_components

        # Create 3 neural networks for each component
        self.freq_nets = nn.ModuleList()
        self.len_nets = nn.ModuleList()
        self.var_nets = nn.ModuleList()

        for q in range(n_components):
            self.freq_nets.append(self._create_mlp(input_dim, input_dim, hidden_dims, activation))
            self.len_nets.append(self._create_mlp(input_dim, input_dim, hidden_dims, activation))
            self.var_nets.append(self._create_mlp(input_dim, 1, hidden_dims, activation))

        # Track PD failures for reporting
        self.pd_failures = 0
        self.total_checks = 0

    def _create_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str
    ) -> nn.Module:
        """Create MLP following Remes architecture."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'selu':
                layers.append(nn.SELU())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.ELU())
            prev_dim = hidden_dim

        # Final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        # Softplus to ensure positivity (freq, len, var all > 0)
        layers.append(nn.Softplus())

        net = nn.Sequential(*layers)

        # Xavier initialization (similar to GPflow)
        for m in net.modules():
            if isinstance(m, nn.Linear):
                limit = np.sqrt(6.0 / (m.in_features + m.out_features))
                nn.init.uniform_(m.weight, -limit, limit)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        return net

    def forward(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute kernel matrix K[i,j] following Remes et al. 2017.

        WARNING: This is NOT guaranteed to be PSD!

        Parameters
        ----------
        X1 : torch.Tensor, shape (n1, d)
            First set of locations
        X2 : torch.Tensor, shape (n2, d), optional
            Second set of locations (if None, use X1)

        Returns
        -------
        K : torch.Tensor, shape (n1, n2)
            Kernel matrix (possibly NOT PSD!)
        """
        if X1.dim() == 1:
            X1 = X1.unsqueeze(-1)
        if X2 is None:
            X2 = X1
        elif X2.dim() == 1:
            X2 = X2.unsqueeze(-1)

        n1, n2 = X1.shape[0], X2.shape[0]

        # Initialize kernel matrix
        K = torch.zeros(n1, n2)

        # Sum over mixture components
        for q in range(self.n_components):
            # Compute input-dependent parameters
            freq1 = self.freq_nets[q](X1)  # (n1, d)
            freq2 = self.freq_nets[q](X2)  # (n2, d)
            len1 = self.len_nets[q](X1)    # (n1, d)
            len2 = self.len_nets[q](X2)    # (n2, d)
            var1 = self.var_nets[q](X1)    # (n1, 1)
            var2 = self.var_nets[q](X2)    # (n2, 1)

            # Variance term: WW = w(x1) * w(x2)^T
            WW = var1 @ var2.T  # (n1, n2)

            # Lengthscale/exponential term (Remes eq. 6)
            # E = sqrt(2*l1*l2/(l1²+l2²)) * exp(-||x1-x2||²/(l1²+l2²))
            X1_exp = X1.unsqueeze(1)  # (n1, 1, d)
            X2_exp = X2.unsqueeze(0)  # (1, n2, d)
            l1_exp = len1.unsqueeze(1)  # (n1, 1, d)
            l2_exp = len2.unsqueeze(0)  # (1, n2, d)

            L = l1_exp**2 + l2_exp**2  # (n1, n2, d)
            D = torch.sum((X1_exp - X2_exp)**2 / L, dim=2)  # (n1, n2)
            det = torch.prod(torch.sqrt(2 * l1_exp * l2_exp / L), dim=2)  # (n1, n2)
            E = det * torch.exp(-D)  # (n1, n2)

            # Cosine term (Remes eq. 6)
            # COS = cos(2π * freq^T (x1 - x2))
            muX = torch.sum(freq1.unsqueeze(1) * X1_exp, dim=2)  # (n1, n2)
            muX2 = torch.sum(freq2.unsqueeze(0) * X2_exp, dim=2)  # (n1, n2)
            COS = torch.cos(2 * np.pi * (muX - muX2))  # (n1, n2)

            # Component kernel
            K += WW * E * COS

        return K

    def make_psd(
        self,
        K: torch.Tensor,
        max_jitter: float = 1e-1,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, bool]:
        """
        Attempt to make kernel PSD by adding jitter (Remes approach).

        This is the KEY difference from F-SDN:
            - Remes: Check eigenvalues AFTER computing K, add jitter if needed
            - F-SDN: ALWAYS PSD by factorization construction

        Returns
        -------
        K_psd : torch.Tensor
            (Hopefully) PSD kernel matrix
        success : bool
            Whether PSD correction succeeded
        """
        self.total_checks += 1

        # Check eigenvalues
        try:
            eigvals = torch.linalg.eigvalsh(K)
            min_eig = torch.min(eigvals)

            if min_eig >= -1e-6:
                # Already PSD (within numerical tolerance)
                return K, True

            # NOT PSD - need to fix!
            self.pd_failures += 1

            if verbose:
                print(f"⚠️  Remes PD failure #{self.pd_failures}/{self.total_checks}: "
                      f"min_eig = {min_eig:.6f}")

            # Add jitter to make PSD (Remes strategy)
            jitter = torch.abs(min_eig) + 1e-4

            if jitter > max_jitter:
                # Jitter too large - FAIL!
                if verbose:
                    print(f"❌ Remes FAILED: jitter {jitter:.6f} > max {max_jitter}")
                return K, False

            K_psd = K + jitter * torch.eye(K.shape[0])

            if verbose:
                print(f"✓ Remes fixed with jitter = {jitter:.6f}")

            return K_psd, True

        except RuntimeError as e:
            # Eigenvalue decomposition failed
            self.pd_failures += 1
            if verbose:
                print(f"❌ Remes eigenvalue decomp FAILED: {e}")
            return K, False

    def compute_covariance(
        self,
        X1: torch.Tensor,
        X2: Optional[torch.Tensor] = None,
        noise_var: float = 1e-6,
        max_jitter: float = 1e-1,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, bool]:
        """
        Compute covariance matrix with PSD correction.

        Returns
        -------
        K : torch.Tensor
            Covariance matrix
        psd_success : bool
            Whether K is guaranteed PSD
        """
        # Compute kernel (possibly NOT PSD!)
        K = self.forward(X1, X2)

        if X2 is None:
            # Symmetrize
            K = (K + K.T) / 2.0
            # Add noise
            K = K + noise_var * torch.eye(K.shape[0])
            # Try to make PSD
            K, success = self.make_psd(K, max_jitter=max_jitter, verbose=verbose)
            return K, success
        else:
            # Cross-covariance - don't add noise or check PSD
            return K, True

    def posterior_mean_loss(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        noise_var: float = 1e-4,
        max_jitter: float = 1e-1,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, bool]:
        """
        Negative log marginal likelihood (GPML eq 2.30).

        Returns
        -------
        loss : torch.Tensor
            NLL (or inf if Cholesky fails)
        success : bool
            Whether computation succeeded
        """
        # Compute covariance (possibly NOT PSD!)
        K_train, psd_success = self.compute_covariance(
            X_train, noise_var=noise_var, max_jitter=max_jitter, verbose=verbose
        )

        if not psd_success:
            # PSD correction failed - return large loss
            return torch.tensor(1e6), False

        # Try Cholesky (can still fail even after PSD correction!)
        try:
            L = torch.linalg.cholesky(K_train)
        except RuntimeError:
            # Cholesky failed - Remes problem!
            if verbose:
                print("❌ Remes Cholesky FAILED even after PSD correction!")
            return torch.tensor(1e6), False

        # Solve K^{-1} y
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze()

        # Negative log marginal likelihood
        loss = 0.5 * (
            y_train @ alpha +
            2 * torch.sum(torch.log(torch.diag(L))) +
            len(y_train) * np.log(2 * np.pi)
        )

        return loss, True

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 500,
        lr: float = 1e-2,
        noise_var: float = 0.01,
        patience: int = 100,
        verbose: bool = True
    ) -> Tuple[List[float], int]:
        """
        Train the Remes kernel.

        Returns
        -------
        losses : List[float]
            Training loss history
        n_failures : int
            Number of PD failures during training
        """
        # Zero-mean the data
        y_train = y_train - y_train.mean()

        # Reset PD failure counter
        self.pd_failures = 0
        self.total_checks = 0

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2, eta_min=lr/100
        )

        # Early stopping
        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        losses = []

        if verbose:
            print("TRAINING REMES BASELINE (NO PD Guarantee!):")
            print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
            print(f"  Components: {self.n_components}")
            print(f"  Epochs: {epochs}")
            print()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute loss (can fail!)
            loss, success = self.posterior_mean_loss(
                X_train, y_train, noise_var=noise_var, verbose=False
            )

            if not success:
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch:4d}/{epochs} | PD FAILURE - skipping update")
                # Skip this update
                scheduler.step()
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
                fail_rate = 100 * self.pd_failures / max(self.total_checks, 1)
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.6f} | PD failures: {self.pd_failures}/{self.total_checks} "
                      f"({fail_rate:.1f}%)")

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"\n✓ Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                print(f"\n✓ Restored best model (loss: {best_loss:.4f})")
                print(f"✓ Total PD failures: {self.pd_failures}/{self.total_checks} "
                      f"({100*self.pd_failures/max(self.total_checks,1):.1f}%)")

        return losses, self.pd_failures
