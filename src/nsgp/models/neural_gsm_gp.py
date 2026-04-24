import torch
import gpytorch
from typing import List, Optional, Tuple

from ..kernel.neural_gsm import NeuralGSMKernel


class ExactNeuralGSMGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NeuralGSMGP:
    """
    Exact GP with the Neural-GSM kernel (Remes et al. 2018).

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    n_components : int
        Number of spectral mixture components Q.
    hidden_dims : list of int
        Hidden layer sizes for parameter networks.
    prior_variance : float
        Variance of Gaussian prior on NN weights. W sim N(0, prior_variance * I).
        Default 1.0, matching GPflow's Gaussian(0, 1).
    """

    def __init__(
        self,
        input_dim: int = 1,
        n_components: int = 1,
        hidden_dims: List[int] = [32, 32],
        prior_variance: float = 1.0,
    ):
        self.input_dim = input_dim
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.prior_variance = prior_variance
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.kernel = NeuralGSMKernel(
            input_dim=input_dim,
            n_components=n_components,
            hidden_dims=hidden_dims,
            prior_variance=prior_variance,
        )
        self.model = None
        self.best_loss = None

    def compute_covariance(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            return self.model.covar_module(X1, X2).to_dense()

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.1,
        patience: Optional[int] = None,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train via MAP.
        MAP equals exact marginal likelihood + L2 regularization on NN weights.

        Returns
        -------
        losses : list of float
            Training loss history.
        """
        self.model = ExactNeuralGSMGP(X_train, y_train, self.likelihood, self.kernel)
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
            print(f"TRAINING NEURAL-GSM (Q={self.n_components}):")
            print(f"  Parameters: {n_params:,}")
            print(f"  Hidden dims: {self.hidden_dims}")
            print(f"  Prior variance: {self.prior_variance}")
            print(f"  Epochs: {epochs}")
            print()

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            # MLL includes the Gaussian prior via AddedLossTerm automatically
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
