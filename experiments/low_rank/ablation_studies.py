import torch
import pandas as pd
from pathlib import Path

from nsgp.lowrank import RegularNonstationaryFeatures
from nsgp.kernel import LocalStationaryKernel

import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def relative_error(K_true, K_approx):
    return (torch.norm(K_true - K_approx) / torch.norm(K_true)).item() * 100


lsk = LocalStationaryKernel(a=1.0)

with torch.no_grad():
    delta_x = 0.001
    results = []

    n_xdata_grid = [1000, 2000, 4000, 8000]

    for n_xdata in n_xdata_grid:

        x = torch.arange(n_xdata).reshape(-1, 1) * delta_x
        K_true = lsk.kernel(x, x)

        # Ablation 1: Error vs number of features (fixed omega_max)
        omega_max = 5.0
        num_feat_grid = [20, 40, 80, 160, 320]

        for num_feat in num_feat_grid:
            delta_omega = omega_max / num_feat
            nff = RegularNonstationaryFeatures(
                spectral=lsk.spectral,
                spectral_real=True,
                num_feat=num_feat
            )
            K_approx = nff.kernel_estimate(x, x, spacing=delta_omega)
            error = relative_error(K_true, K_approx)

            results.append({
                "ablation": "num_features",
                "num_feat": num_feat,
                "omega_max": omega_max,
                "n_xdata": n_xdata,
                "error_percent": error,
            })

        # Ablation 2: Error vs omega_max (fixed num_feat)
        num_feat = 100
        omega_max_grid = [4, 6, 8, 10, 12]

        for omega_max in omega_max_grid:
            delta_omega = omega_max / num_feat
            nff = RegularNonstationaryFeatures(
                spectral=lsk.spectral,
                spectral_real=True,
                num_feat=num_feat
            )
            K_approx = nff.kernel_estimate(x, x, spacing=delta_omega)
            error = relative_error(K_true, K_approx)

            results.append({
                "ablation": "omega_max",
                "num_feat": num_feat,
                "omega_max": omega_max,
                "n_xdata": n_xdata,
                "error_percent": error,
            })

    # Save results
    df = pd.DataFrame(results)

    # Export separate CSV files for LaTeX plots
    df_feat = df[df['ablation'] == 'num_features']
    df_omega = df[df['ablation'] == 'omega_max']

    for n in n_xdata_grid:
        # Error vs features
        df_n = df_feat[df_feat['n_xdata'] == n][['num_feat', 'error_percent']]
        feat_path = DATA_DIR / f'ablation_features_n{n}.csv'
        df_n.to_csv(feat_path, index=False)

        # Error vs omega
        df_n = df_omega[df_omega['n_xdata'] == n][['omega_max', 'error_percent']]
        omega_path = DATA_DIR / f'ablation_omega_n{n}.csv'
        df_n.to_csv(omega_path, index=False)

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Error vs number of features (curves for different n_xdata)
    df_num_feat = df[df['ablation'] == 'num_features']
    for n_xdata in n_xdata_grid:
        df_n = df_num_feat[df_num_feat['n_xdata'] == n_xdata]
        ax1.plot(df_n['num_feat'], df_n['error_percent'], 'o-',
                 linewidth=2, markersize=6, label=f'n={n_xdata}')
    ax1.set_xlabel(r'Number of features $m$')
    ax1.set_ylabel(r'Relative error$~/~\%$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Error vs Number of Features')
    ax1.legend()

    # Plot 2: Error vs cutoff frequency (curves for different n_xdata)
    df_omega = df[df['ablation'] == 'omega_max']
    for n_xdata in n_xdata_grid:
        df_n = df_omega[df_omega['n_xdata'] == n_xdata]
        ax2.plot(df_n['omega_max'], df_n['error_percent'], 's-',
                 linewidth=2, markersize=6, label=f'n={n_xdata}')
    ax2.set_xlabel(r'Cutoff frequency $\omega_{\mathrm{max}}$')
    ax2.set_ylabel(r'Relative error$~/~\%$')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Error vs Cutoff Frequency')
    ax2.legend()

    plt.tight_layout()
    plt.show()
