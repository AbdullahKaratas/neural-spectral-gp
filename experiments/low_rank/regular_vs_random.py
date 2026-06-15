import math
import torch
import numpy as np
from scipy import stats

from nsgp.lowrank import RegularNonstationaryFeatures
from nsgp.lowrank import RandomNonstationaryFeatures
from nsgp.kernel import LocalStationaryKernel


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
    err_sym = relative_error(K_sym, K_true) * 100.0

    # Regular NFF
    num_feat_nff = 20
    cutoff = 5.0
    spacing = cutoff / num_feat_nff

    nff = RegularNonstationaryFeatures(
        spectral=lsk.spectral, spectral_real=True, num_feat=num_feat_nff
    )
    with torch.no_grad():
        K_nff = nff.kernel_estimate(x, x, spacing=spacing)
    err_nff = relative_error(K_nff, K_true) * 100.0

    print(f"Regular NFF (m={num_feat_nff}): relative error = {err_nff:.4f}%")

    # Random NFF (MC, Ton et al. 2018)
    mc_features_list = [20, 50, 200, 500, 2000]
    n_seeds = 10

    results = []

    def sampler(n_feat, a=a):
        return silverman_sampler(n_feat, a=a)

    for m in mc_features_list:
        row = {"n_features": m}

        # Regular NFF
        nff_m = RegularNonstationaryFeatures(
            spectral=lsk.spectral, spectral_real=True, num_feat=m
        )
        with torch.no_grad():
            K_nff_m = nff_m.kernel_estimate(x, x, spacing=cutoff / m)
        row["regular_error"] = relative_error(K_nff_m, K_true) * 100.0

        # Random NFF (MC over seeds)
        errors = []
        for seed in range(n_seeds):
            rnff = RandomNonstationaryFeatures(spectral_sampler=sampler, n_feat=m)
            rnff.sample_frequencies(seed=seed)

            with torch.no_grad():
                K_mc = rnff.kernel_estimate(x, x)

            err = relative_error(K_mc, K_true)
            errors.append(err)

        vals = np.array(errors) * 100.0
        n_ok = len(vals)
        t_crit = stats.t.ppf(0.975, n_ok - 1)
        row["random_mean"] = float(vals.mean())
        row["random_ci95"] = float(t_crit * vals.std(ddof=1) / np.sqrt(n_ok))
        row["errors"] = vals.tolist()
        results.append(row)

        print(
            f"m={m:4d}: regular={row['regular_error']:.4f}  "
            f"random={row['random_mean']:.4f} +/- {row['random_ci95']:.4f}"
        )
