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
import gpytorch

from typing import Optional, Tuple


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class StandardGP:
    """
    Standard Stationary Gaussian Process.
    """
    
    def __init__(self):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def compute_covariance(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute covariance matrix with optional noise.
        """
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            return self.model.covar_module(X1, X2).to_dense()

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100, lr: float = 0.1, verbose: bool = True):
        """
        Optimize hyperparameters.
        """

        self.model = ExactGPModel(X_train, y_train, self.likelihood)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        losses = []
        for i in range(epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            if verbose and (i % 100 == 0 or i == epochs - 1):
                print(f"Epoch {i}: Loss = {loss.item():.4f}")
            optimizer.step()
            losses.append(loss.item())

        if verbose:
            print(f"Standard GP Optimization finished.")
            print(f"  Final Loss: {loss.item():.4f}")

        return losses

    def predict(self, X_test: torch.Tensor, predictive_dist=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Posterior prediction using exact GP inference.
        """

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            if predictive_dist:
                pred = self.likelihood(self.model(X_test))
            else:
                pred = self.model(X_test)
        
        mean = pred.mean 
        var = pred.variance
        return mean, torch.sqrt(var)
