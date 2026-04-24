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
        self.model = None

    def compute_covariance(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute covariance matrix with optional noise.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            return self.model.covar_module(X1, X2).to_dense()

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100, lr: float = 0.1, patience: int = None, verbose: bool = True):
        """
        Optimize hyperparameters. Assumes zero-mean GP.
        """

        self.model = ExactGPModel(X_train, y_train, self.likelihood)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Early stopping and best state
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        if patience is None:
            patience = epochs

        losses = []
        for i in range(epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Early stopping and best state tracking
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if verbose and (i % 100 == 0 or i == epochs - 1):
                print(f"Epoch {i}: Loss = {loss.item():.4f} | Best: {best_loss:.4f}")

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {i} (no improvement for {patience} epochs)")
                break

        # Restore best model and store best loss
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_loss = best_loss
            if verbose:
                print(f"Standard GP Optimization finished.")
                print(f"  Best Loss: {best_loss:.4f}")
        else:
            self.best_loss = loss.item()
            if verbose:
                print(f"Standard GP Optimization finished.")
                print(f"  Final Loss: {loss.item():.4f}")

        return losses

    def _full_pred_dist(
        self, X_test: torch.Tensor, predictive_dist: bool = True
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Return the full joint predictive distribution.

        Parameters
        ----------
        X_test : torch.Tensor, shape (t, d)
            Test locations.
        predictive_dist : bool
            If True, include observation noise.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            if predictive_dist:
                return self.likelihood(self.model(X_test))
            else:
                return self.model(X_test)

    def predict(self, X_test: torch.Tensor, predictive_dist=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Posterior prediction using exact GP inference.
        """
        pred = self._full_pred_dist(X_test, predictive_dist=predictive_dist)
        mean = pred.mean
        var = pred.variance
        return mean, torch.sqrt(var)
