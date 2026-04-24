import torch
import gpytorch
from typing import List, Optional, Tuple

from ..kernel.deep_kernel import FeatureExtractor


class ExactDKLGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLGP:
    """
    Exact GP with stationary neural kernel (Wilson et al. 2015).
    Fully connected network defaults to paper architecture

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input space D.
    output_dim : int
        Dimensionality of the feature space F (output of phi).
    hidden_dims : list of int
        Sizes of hidden layers. Empty list gives a single linear map.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 2,
        hidden_dims: List[int] = [1000, 500, 50],
        ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims
        )
        self.model = None
        self.best_loss = None

    def compute_covariance(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the DKL covariance matrix k(x, x') = k_RBF(s(phi(x)), s(phi(x'))),
        where phi is the feature extractor and s is `ScaleToBounds`.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            phi1 = self.model.scale_to_bounds(self.model.feature_extractor(X1))
            if X2 is None:
                phi2 = phi1
            else:
                phi2 = self.model.scale_to_bounds(self.model.feature_extractor(X2))
            return self.model.covar_module(phi1, phi2).to_dense()


    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 100, lr: float = 0.1, patience: int = None, verbose: bool = True):
        """
        Optimize hyperparameters. Assumes zero-mean GP.
        """
        self.model = ExactDKLGP(X_train, y_train, self.likelihood, self.feature_extractor)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        if patience is None:
            patience = epochs

        losses = []

        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"TRAINING DKL:")
            print(f"  Parameters: {n_params:,}")
            print(f"  Hidden dims: {self.hidden_dims}")
            print(f"  Epochs: {epochs}")
            print()

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)

            if torch.isnan(loss):
                if verbose:
                    print(f"Warning: NaN loss at epoch {epoch}, skipping.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.4f} | Best: {best_loss:.4f}")

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
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

    def predict(
        self,
        X_test: torch.Tensor,
        predictive_dist: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self._full_pred_dist(X_test, predictive_dist=predictive_dist)
        return pred.mean, torch.sqrt(pred.variance)
