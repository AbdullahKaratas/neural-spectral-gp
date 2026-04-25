import math
import torch
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nsgp.kernel import LocalStationaryKernel, HarmonizableMixtureKernel

from nsgp.models import FactorizedSpectralDensityNetwork
from nsgp.models import NeuralGSMGP
from nsgp.models import StandardGP
from nsgp.models import DKLGP


def make_hmk():
    # Set up parameters
    d = 1
    # Q conjugate frequency pairs
    Q = 1
    # HMK components
    P = 1

    # Conjugate frequency pairs
    eta = torch.tensor([[1.0]])
    frequencies = torch.cat([eta, -eta], dim=0)

    B = torch.tensor(
        [[2.0 + 0.0j, 0.0 + 0.5j], [0.0 - 0.5j, 2.0 + 0.0j]], dtype=torch.complex64
    )

    # HMK kernel with specific parameters to match Silverman's kernel
    sigma1 = torch.eye(d) * (1.0 / (math.pi**2))
    sigma2 = torch.eye(d) * (1.0 / (2.0 * math.pi) ** 2)

    centers = [torch.zeros(d)]
    scalings = [torch.ones(d)]
    frequencies_list = [frequencies]
    psd_matrices = [B]
    return HarmonizableMixtureKernel(
        sigma1=sigma1,
        sigma2=sigma2,
        centers=centers,
        scalings=scalings,
        frequencies=frequencies_list,
        psd_matrices=psd_matrices,
    )


def hmk_real_kernel(x1, x2):
    hmk = make_hmk()
    return hmk.k_hmk(x1, x2).real


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
    rbf = StandardGP()
    rbf.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=False)
    K_rbf = rbf.compute_covariance(X_test, X_test)
    # Compute error
    error_rbf = torch.norm(K_rbf - K_true_test) / torch.norm(K_true_test)

    # Neural-GSM (Remes et al. 2018)
    ngsm = NeuralGSMGP(
        input_dim=1,
        n_components=2,
        hidden_dims=[32, 32],
        prior_variance=1.0,
    )
    ngsm.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=False)
    K_ngsm = ngsm.compute_covariance(X_test)
    error_ngsm = torch.norm(K_ngsm - K_true_test) / torch.norm(K_true_test)

    # Deep Kernel Learning (Wilson et al. 2016) — paper-default architecture
    dkl = DKLGP(input_dim=1)
    dkl.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=False)
    K_dkl = dkl.compute_covariance(X_test)
    error_dkl = torch.norm(K_dkl - K_true_test) / torch.norm(K_true_test)

    # F-SDN
    sdnreal = FactorizedSpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[128, 128],
        rank=8,
        n_features=256,
        omega_max=10.0,
        enforce_symmetry=False,
        spectral_real=True,
    )
    sdncomplex = FactorizedSpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[128, 128],
        rank=8,
        n_features=256,
        omega_max=10.0,
        enforce_symmetry=False,
        spectral_real=False,
    )

    sdnreal.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=False)
    K_sdnreal = sdnreal.compute_covariance(X_test)
    error_sdnreal = torch.norm(K_sdnreal - K_true_test) / torch.norm(K_true_test)

    sdncomplex.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=False)
    K_sdncomplex = sdncomplex.compute_covariance(X_test)
    error_sdncomplex = torch.norm(K_sdncomplex - K_true_test) / torch.norm(K_true_test)

    return {
        "seed": seed,
        "error_rbf": error_rbf.item(),
        "error_ngsm": error_ngsm.item(),
        "error_dkl": error_dkl.item(),
        "error_sdnreal": error_sdnreal.item(),
        "error_sdncomplex": error_sdncomplex.item(),
    }


def run_benchmark(kernel_fn, kernel_name: str, n_seeds: int = 5):
    print(f"Benchmark: {kernel_name}")

    results = []
    for seed in range(n_seeds):
        print(f"Seed {seed+1}/{n_seeds} (seed={seed + 42})")
        result = run_single_comparison(
            kernel_fn,
            seed=seed + 42,
        )
        results.append(result)
        print(f"RBF: {result['error_rbf']*100:.2f} %")
        print(f"Neural-GSM: {result['error_ngsm']*100:.2f}%")
        print(f"DKL: {result['error_dkl']*100:.2f}%")
        print(f"F-SDN (real): {result['error_sdnreal']*100:.2f}%")
        print(f"F-SDN (complex): {result['error_sdncomplex']*100:.2f}%")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Compute 95 % confidence intervals
    n = len(df)
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha/2, n - 1)

    if n_seeds > 1:
        for method, col in [("RBF", "error_rbf"), ("Neural-GSM", "error_ngsm"), ("DKL", "error_dkl"), ("F-SDN (real)", "error_sdnreal"), ("F-SDN (complex)", "error_sdncomplex")]:
            mean = df[col].mean()
            std = df[col].std(ddof=1)
            ci = t_crit * std / np.sqrt(n)
            print(f"{method}:")
            print(f"K-error: {mean*100:.2f}% +/- {ci*100:.2f}%")
    return df


def main(n_seeds: int = 5):
    # 1. Silverman locally stationary kernel
    lsk = LocalStationaryKernel(a=0.5)
    df_lsk = run_benchmark(lsk.kernel, "Silverman Locally Stationary", n_seeds=n_seeds)

    # 2. Harmonizable Mixture Kernel
    df_hmk = run_benchmark(hmk_real_kernel, "Harmonizable Mixture Kernel", n_seeds=n_seeds)

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    methods = ["RBF", "Neural-GSM", "DKL", "F-SDN (real)", "F-SDN (complex)"]
    columns = ["error_rbf", "error_ngsm", "error_dkl", "error_sdnreal", "error_sdncomplex"]
    colors = sns.cubehelix_palette(n_colors=len(methods), reverse=True)

    for ax, df, name in zip(axes, [df_lsk, df_hmk], ["Silverman LS", "HMK"]):
        n = len(df)
        means = [df[col].mean() * 100 for col in columns]
        if n > 1:
            t_crit = stats.t.ppf(0.975, n - 1)
            cis = [t_crit * df[col].std(ddof=1) / np.sqrt(n) * 100 for col in columns]
        else:
            cis = [0.0] * len(columns)

        ax.bar(methods, means, yerr=cis, capsize=6, color=colors, edgecolor="black", width=0.6)
        ax.set_title(name)
        ax.set_ylabel("K-error in %")
        ax.tick_params(axis='x', rotation=25)

    fig.suptitle("Kernel Approximation Error", fontsize=13)
    fig.tight_layout()
    plt.show()

    return df_lsk, df_hmk


if __name__ == "__main__":
    main()
