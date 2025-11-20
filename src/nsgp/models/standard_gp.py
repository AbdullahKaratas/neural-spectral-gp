"""
Standard GP Baseline

Implements a standard stationary Gaussian Process using exact inference.
This serves as a strong baseline to compare against F-SDN.

Kernels supported:
- RBF (Squared Exponential)
- Matérn (1/2, 3/2, 5/2)
- Spectral Mixture (Wilson & Adams, 2013)

Authors: Abdullah Karatas, Arsalan Jawaid
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, Union

class StandardGP(nn.Module):
    """
    Standard Stationary Gaussian Process.
    
    Parameters
    ----------
    kernel_type : str
        'rbf', 'matern12', 'matern32', 'matern52', 'sm' (Spectral Mixture)
    input_dim : int
        Input dimension
    variance : float
        Initial signal variance (output scale)
    lengthscale : float
        Initial lengthscale
    noise_var : float
        Initial noise variance
    """
    
    def __init__(
        self, 
        kernel_type: str = 'rbf',
        input_dim: int = 1,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        noise_var: float = 0.01,
        n_mixtures: int = 5  # For Spectral Mixture kernel only
    ):
        super().__init__()
        self.kernel_type = kernel_type.lower()
        self.input_dim = input_dim
        
        # Hyperparameters (in log space for positivity)
        self.log_variance = nn.Parameter(torch.tensor(np.log(variance)))
        self.log_lengthscale = nn.Parameter(torch.tensor(np.log(lengthscale)))
        self.log_noise = nn.Parameter(torch.tensor(np.log(noise_var)))
        
        # Spectral Mixture specific parameters
        if self.kernel_type == 'sm':
            # Weights, means, variances for Q components
            # Initialize with random values
            self.sm_log_weights = nn.Parameter(torch.randn(n_mixtures))
            self.sm_log_means = nn.Parameter(torch.randn(n_mixtures, input_dim))
            self.sm_log_variances = nn.Parameter(torch.randn(n_mixtures, input_dim))
            self.n_mixtures = n_mixtures

    @property
    def variance(self):
        return torch.exp(self.log_variance)
        
    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)
        
    @property
    def noise(self):
        return torch.exp(self.log_noise)

    def forward(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute kernel matrix K(X1, X2).
        """
        if X2 is None:
            X2 = X1
            
        # Ensure correct dimensions
        if X1.dim() == 1: X1 = X1.unsqueeze(-1)
        if X2.dim() == 1: X2 = X2.unsqueeze(-1)
        
        # Compute pairwise distances
        # dist_sq[i,j] = ||x1_i - x2_j||^2
        # Using (a-b)^2 = a^2 + b^2 - 2ab
        x1_sq = torch.sum(X1**2, dim=1, keepdim=True)
        x2_sq = torch.sum(X2**2, dim=1, keepdim=True)
        dist_sq = x1_sq + x2_sq.t() - 2 * X1 @ X2.t()
        dist_sq = torch.clamp(dist_sq, min=1e-8) # Numerical stability
        dist = torch.sqrt(dist_sq)
        
        if self.kernel_type == 'rbf':
            # K = σ² exp(-r² / 2ℓ²)
            K = self.variance * torch.exp(-dist_sq / (2 * self.lengthscale**2))
            
        elif self.kernel_type == 'matern12':
            # K = σ² exp(-r / ℓ)
            K = self.variance * torch.exp(-dist / self.lengthscale)
            
        elif self.kernel_type == 'matern32':
            # K = σ² (1 + √3r/ℓ) exp(-√3r/ℓ)
            sqrt3_r = np.sqrt(3) * dist / self.lengthscale
            K = self.variance * (1 + sqrt3_r) * torch.exp(-sqrt3_r)
            
        elif self.kernel_type == 'matern52':
            # K = σ² (1 + √5r/ℓ + 5r²/3ℓ²) exp(-√5r/ℓ)
            sqrt5_r = np.sqrt(5) * dist / self.lengthscale
            K = self.variance * (1 + sqrt5_r + (5 * dist_sq) / (3 * self.lengthscale**2)) * torch.exp(-sqrt5_r)
            
        elif self.kernel_type == 'sm':
            # Spectral Mixture Kernel
            # k(tau) = sum_q w_q * cos(2pi * mu_q^T * tau) * prod_d exp(-2pi^2 * tau_d^2 * v_q_d)
            
            tau = X1.unsqueeze(1) - X2.unsqueeze(0) # (N1, N2, D)
            
            weights = torch.exp(self.sm_log_weights)
            weights = weights / weights.sum() * self.variance # Normalize and scale
            
            means = torch.exp(self.sm_log_means)
            variances = torch.exp(self.sm_log_variances)
            
            K = torch.zeros(X1.shape[0], X2.shape[0], device=X1.device)
            
            for q in range(self.n_mixtures):
                # Cosine term: cos(2pi * mu_q^T * tau)
                # tau: (N1, N2, D), means[q]: (D)
                cos_term = torch.cos(2 * np.pi * torch.sum(tau * means[q], dim=-1))
                
                # Exp term: exp(-2pi^2 * tau^T * V_q * tau)
                exp_term = torch.exp(-2 * np.pi**2 * torch.sum(tau**2 * variances[q], dim=-1))
                
                K += weights[q] * cos_term * exp_term
                
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
        return K

    def compute_covariance(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None, noise_var: float = 0.0) -> torch.Tensor:
        """
        Compute covariance matrix with optional noise.
        """
        K = self.forward(X1, X2)
        
        if X2 is None or X2 is X1:
            # Add noise to diagonal
            K = K + noise_var * torch.eye(K.shape[0], device=K.device)
            
        return K

    def neg_log_marginal_likelihood(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Negative Log Marginal Likelihood.
        """
        K = self.compute_covariance(X, noise_var=self.noise)
        
        # Cholesky
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            # Add jitter if failed
            K = K + 1e-4 * torch.eye(K.shape[0], device=K.device)
            L = torch.linalg.cholesky(K)
            
        # Solve alpha = K^-1 y
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze()
        
        # Terms
        data_fit = 0.5 * y @ alpha
        complexity = torch.sum(torch.log(torch.diag(L)))
        constant = 0.5 * len(y) * np.log(2 * np.pi)
        
        return data_fit + complexity + constant

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100, lr: float = 0.1, verbose: bool = True):
        """
        Optimize hyperparameters.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        losses = []
        for i in range(epochs):
            optimizer.zero_grad()
            loss = self.neg_log_marginal_likelihood(X_train, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if verbose and (i % 10 == 0 or i == epochs - 1):
                print(f"Epoch {i}: Loss = {loss.item():.4f}, Lengthscale = {self.lengthscale.item():.4f}, Variance = {self.variance.item():.4f}")
                
        if verbose:
            print(f"StandardGP Optimization finished.")
            print(f"  Final Loss: {loss.item():.4f}")
            print(f"  Lengthscale: {self.lengthscale.item():.4f}")
            print(f"  Variance: {self.variance.item():.4f}")
            print(f"  Noise: {self.noise.item():.4f}")
            
        return losses
