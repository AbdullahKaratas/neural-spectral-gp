import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from nsgp.lowrank import RegularNonstationaryFeatures
from nsgp.lowrank import RandomNonstationaryFeatures
from nsgp.kernel import LocalStationaryKernel


DATA_DIR = Path(__file__).parent / "data" / "regular_vs_random"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def silverman_sampler(m, a=1.0):
    """Sample (omega1, omega2) from Silverman's spectral density.

    The spectral density simplifies as
    s(w1, w2) = N(w1; 0, 2a) * N(w2; 0, 2a).
    """
    std = math.sqrt(2.0 * a)
    omega1 = torch.randn(m, 1) * std
    omega2 = torch.randn(m, 1) * std
    return omega1, omega2


def relative_error(K_approx, K_true):
    """Relative Frobenius norm error."""
    return (torch.norm(K_approx - K_true) / torch.norm(K_true)).item()


if __name__ == "__main__":
    a = 1.0
    lsk = LocalStationaryKernel(a=a)

    # Spatial grid
    delta_x = 0.001
    n_pts = 2500
    x = torch.arange(n_pts).reshape(-1, 1) * delta_x

    # Ground truth kernel
    with torch.no_grad():
        K_true = lsk.kernel(x, x)

    # Symmetrized kernel (irreducible bias floor for random NFF)
    xx, yy = x, x.T
    K_sym = 0.5 * torch.exp(-a * (xx**2 + yy**2)) + 0.5 * torch.exp(
        -a * (xx - yy) ** 2
    )
    err_sym = relative_error(K_sym, K_true)

    # Regular NFF
    num_feat_nff = 20
    cutoff = 5.0
    spacing = cutoff / num_feat_nff

    nff = RegularNonstationaryFeatures(
        spectral=lsk.spectral, spectral_real=True, num_feat=num_feat_nff
    )
    with torch.no_grad():
        K_nff = nff.kernel_estimate(x, x, spacing=spacing)
    err_nff = relative_error(K_nff, K_true)

    print(f"Regular NFF (m={num_feat_nff}): relative error = {err_nff:.4f}")

    # Random NFF (MC, Ton et al. 2018)
    mc_features_list = [20, 50, 200, 500, 2000]
    n_seeds = 10

    results = []

    def sampler(n_feat, a=a):
        return silverman_sampler(n_feat, a=a)

    for m in mc_features_list:
        errors = []
        for seed in range(n_seeds):
            rnff = RandomNonstationaryFeatures(spectral_sampler=sampler, n_feat=m)
            rnff.sample_frequencies(seed=seed)

            with torch.no_grad():
                K_mc = rnff.kernel_estimate(x, x)

            err = relative_error(K_mc, K_true)
            errors.append(err)

        mean_err = np.mean(errors)
        std_err = np.std(errors, ddof=1)
        results.append(
            {
                "n_features": m,
                "mean_error": mean_err,
                "std_error": std_err,
                "errors": errors,
            }
        )
        ci = 2.0 * std_err / np.sqrt(n_seeds)
        print(f"Random NFF (m={m:4d}): {mean_err:.4f} +/- {ci:.4f}")

    # Plot: convergence of Random NFF vs Regular NFF baseline
    fig, ax = plt.subplots(figsize=(8, 5))

    mc_m = [r["n_features"] for r in results]
    mc_mean = [r["mean_error"] for r in results]
    mc_std = [2.0 * r["std_error"] / np.sqrt(n_seeds) for r in results]

    ax.errorbar(mc_m, mc_mean, yerr=mc_std, fmt="o-", label="Random NFF")
    ax.axhline(
        y=err_sym, color="gray", linestyle=":", label="Symmetrization bias floor"
    )
    ax.axhline(
        y=err_nff, color="r", linestyle="--", label=f"Regular NFF (m={num_feat_nff})"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Silverman Kernel: Regular NFF vs Random NFF")
    ax.legend()

    fig.savefig(DATA_DIR / "convergence.png", dpi=150, bbox_inches="tight")

    # Plot: kernel heatmaps (true, NFF, Random NFF)
    largest_m = results[-1]["n_features"]
    rnff_largest = RandomNonstationaryFeatures(
        spectral_sampler=sampler,
        n_feat=largest_m,
    )
    rnff_largest.sample_frequencies(seed=0)
    with torch.no_grad():
        K_mc_largest = rnff_largest.kernel_estimate(x, x)

    x_max = (n_pts - 1) * delta_x
    N = 256
    cmap = sns.cubehelix_palette(n_colors=N, as_cmap=True, reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, K, title in zip(
        axes,
        [K_true, K_nff, K_mc_largest],
        [
            "True Kernel",
            f"Regular NFF (m={num_feat_nff})",
            f"Random NFF (m={largest_m})",
        ],
    ):
        ax.imshow(
            K.numpy(),
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[0, x_max, 0, x_max],
            aspect="auto",
        )
        ax.set_title(title)

    plt.tight_layout()
    fig.savefig(DATA_DIR / "kernel_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.show()
