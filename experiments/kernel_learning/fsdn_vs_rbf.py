import torch
import numpy as np
from scipy import stats
import pandas as pd

from nsgp.models import FactorizedSpectralDensityNetwork
from nsgp.models import StandardGP
from nsgp.kernel import LocalStationaryKernel


def run_single_comparison(
    kernel_fn,
    seed: int,
    n_train: int = 50,
    n_test: int = 100,
    epochs: int = 4000,
    noise_var: float = 1e-4,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    X_train = torch.linspace(-5, 5, n_train).unsqueeze(-1)
    X_test = torch.linspace(-10, 10, n_test).unsqueeze(-1)

    # Ground truth
    K_true_train = kernel_fn(X_train, X_train)

    # Generate noisy observations
    L = torch.linalg.cholesky(K_true_train + noise_var * torch.eye(n_train))
    y_train = (L @ torch.randn(n_train)).squeeze()

    # Ground truth test kernel
    K_true_test = kernel_fn(X_test, X_test)

    # Standard GP Baseline
    gp = StandardGP()
    gp.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=False)
    K_gp = gp.compute_covariance(X_test, X_test)
    # Compute error
    error_gp = torch.norm(K_gp - K_true_test) / torch.norm(K_true_test)

    # F-SDN
    sdn = FactorizedSpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[128, 128],
        rank=8,
        n_features=256,
        omega_max=10.0,
        enforce_symmetry=False,
    )
    sdn.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=False)
    K_sdn = sdn.compute_covariance(X_test)
    error_sdn = torch.norm(K_sdn - K_true_test) / torch.norm(K_true_test)

    return {
        "seed": seed,
        "error_gp": error_gp.item(),
        "error_sdn": error_sdn.item(),
        "improvement": (error_gp.item() - error_sdn.item()) / error_gp.item(),
    }


def main(n_seeds=10):
    results = []
    lsk = LocalStationaryKernel(a=0.5)

    for seed in range(n_seeds):
        print(f"Seed {seed + 1}/{n_seeds} (seed={seed + 42})")
        result = run_single_comparison(
            lsk.kernel,
            seed=seed + 42,
        )
        results.append(result)
        print(f"RBF GP Error: {result['error_gp']*100:.2f} %")
        print(f"F-SDN Error: {result['error_sdn']*100:.2f} %")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Compute 95 % confidence intervals
    n = len(df)
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)

    # RBF GP statistics
    mean_gp = df["error_gp"].mean()
    std_gp = df["error_gp"].std(ddof=1)
    ci_gp = t_crit * std_gp / np.sqrt(n)

    print("RBF:")
    print(f"Mean Error: {mean_gp*100:.2f} %")
    print(f"+- {ci_gp*100:.2f} %")

    # F-SDN statistics
    mean_sdn = df["error_sdn"].mean()
    std_sdn = df["error_sdn"].std(ddof=1)
    ci_sdn = t_crit * std_sdn / np.sqrt(n)

    print("F-SDN (Ours):")
    print(f"Mean Error: {mean_sdn*100:.2f} %")
    print(f"+- {ci_sdn*100:.2f} %")

    return df


if __name__ == "__main__":
    df = main()
