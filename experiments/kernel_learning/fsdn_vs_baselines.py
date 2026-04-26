import math
from dataclasses import dataclass
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats

from nsgp.kernel import HarmonizableMixtureKernel, LocalStationaryKernel
from gpytorch.metrics import negative_log_predictive_density
from nsgp.metrics import (
    kl_posterior,
    marginal_log_likelihood,
    noise_variance,
    oracle_posterior,
)
from nsgp.models import (
    DKLGP,
    FactorizedSpectralDensityNetwork,
    NeuralGSMGP,
    StandardGP,
)


METRIC_KEYS = ["k_error", "nlpd", "kl", "mll", "noise_var"]


@dataclass
class MethodSpec:
    """A method to evaluate. ``factory`` returns a fresh model on each call."""
    label: str
    factory: Callable[[], object]


def make_methods(include_complex: bool = True) -> List[MethodSpec]:
    methods = [
        MethodSpec("RBF", lambda: StandardGP()),
        MethodSpec("Neural-GSM", lambda: NeuralGSMGP(
            input_dim=1, n_components=2, hidden_dims=[32, 32], prior_variance=1.0,
        )),
        MethodSpec("DKL", lambda: DKLGP(input_dim=1)),
        MethodSpec("F-SDN (real)", lambda: FactorizedSpectralDensityNetwork(
            input_dim=1, hidden_dims=[128, 128], rank=8, n_features=256,
            omega_max=10.0, enforce_symmetry=False, spectral_real=True,
        )),
    ]
    if include_complex:
        methods.append(MethodSpec("F-SDN (complex)", lambda: FactorizedSpectralDensityNetwork(
            input_dim=1, hidden_dims=[128, 128], rank=8, n_features=256,
            omega_max=10.0, enforce_symmetry=False, spectral_real=False,
        )))
    return methods


def make_hmk():
    d = 1
    eta = torch.tensor([[1.0]])
    frequencies = torch.cat([eta, -eta], dim=0)
    B = torch.tensor(
        [[2.0 + 0.0j, 0.0 + 0.5j], [0.0 - 0.5j, 2.0 + 0.0j]], dtype=torch.complex64
    )
    sigma1 = torch.eye(d) * (1.0 / (math.pi**2))
    sigma2 = torch.eye(d) * (1.0 / (2.0 * math.pi) ** 2)
    return HarmonizableMixtureKernel(
        sigma1=sigma1,
        sigma2=sigma2,
        centers=[torch.zeros(d)],
        scalings=[torch.ones(d)],
        frequencies=[frequencies],
        psd_matrices=[B],
    )


def hmk_real_kernel(x1, x2):
    return make_hmk().k_hmk(x1, x2).real


def _evaluate_one_method(
    spec: MethodSpec,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    K_true_test: torch.Tensor,
    oracle_post,
    epochs: int,
) -> dict:
    """Fit ``spec.factory()`` and compute the five metrics. NaN on any failure."""
    out = {"method": spec.label, **{k: math.nan for k in METRIC_KEYS}}
    try:
        model = spec.factory()
        model.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=False)

        # K-error: relative Frobenius norm of the kernel matrix at test points.
        K_pred = model.compute_covariance(X_test, X_test) if isinstance(model, StandardGP) \
            else model.compute_covariance(X_test)
        out["k_error"] = float(
            (torch.norm(K_pred - K_true_test) / torch.norm(K_true_test)).item()
        )

        pred_dist = model._full_pred_dist(X_test)
        out["nlpd"] = float(negative_log_predictive_density(pred_dist, y_test).item())
        out["kl"] = float(kl_posterior(pred_dist, oracle_post).item())
        out["mll"] = marginal_log_likelihood(model)
        out["noise_var"] = noise_variance(model)
    except Exception as exc:  # noqa: BLE001 — single seeds are allowed to fail
        out["error_msg"] = f"{type(exc).__name__}: {exc}"
    return out


def run_single_comparison(
    kernel_fn,
    seed: int,
    n_train: int = 50,
    n_test: int = 50,
    epochs: int = 4000,
    noise_var: float = 1e-4,
    x_lo: float = -5.0,
    x_hi: float = 5.0,
    include_complex: bool = True,
) -> List[dict]:
    """Return one dict per method for the given seed (long format)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = torch.linspace(x_lo, x_hi, n_train).unsqueeze(-1)
    X_test, _ = torch.sort(torch.rand(n_test, 1) * (x_hi - x_lo) + x_lo, dim=0)

    XX = torch.cat([X_train, X_test], dim=0)
    K_joint = kernel_fn(XX, XX)
    n = n_train + n_test
    L = torch.linalg.cholesky(K_joint + noise_var * torch.eye(n))
    y = (L @ torch.randn(n)).squeeze()
    y_train, y_test = y[:n_train], y[n_train:]

    K_true_test = K_joint[n_train:, n_train:]

    oracle_post = oracle_posterior(kernel_fn, X_train, y_train, X_test, noise_var=noise_var)

    rows = []
    for spec in make_methods(include_complex=include_complex):
        row = _evaluate_one_method(
            spec, X_train, y_train, X_test, y_test, K_true_test,
            oracle_post, epochs,
        )
        row["seed"] = seed
        rows.append(row)
    return rows


def run_benchmark(
    kernel_fn,
    kernel_name: str,
    seeds: tuple = (42, 43),
    x_lo: float = -5.0,
    x_hi: float = 5.0,
    noise_var: float = 1e-4,
    include_complex: bool = True,
) -> pd.DataFrame:
    print(f"Benchmark: {kernel_name}")
    all_rows = []
    for i, seed in enumerate(seeds):
        print(f"Seed {i+1}/{len(seeds)} (seed={seed})")
        rows = run_single_comparison(
            kernel_fn, seed=seed, x_lo=x_lo, x_hi=x_hi, noise_var=noise_var, include_complex=include_complex,
        )
        all_rows.extend(rows)
        for r in rows:
            kerr = "NaN" if math.isnan(r["k_error"]) else f"{r['k_error']*100:.2f}%"
            nlpd = "NaN" if math.isnan(r["nlpd"]) else f"{r['nlpd']:.3f}"
            print(f"  {r['method']:<16} K-err={kerr:>10}  NLPD={nlpd:>8}")
    return pd.DataFrame(all_rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method mean ± 95% CI over non-NaN seeds, in long format."""
    rows = []
    for method, sub in df.groupby("method", sort=False):
        n_total = len(sub)
        row = {"method": method, "n_total": n_total}
        for key in METRIC_KEYS:
            vals = sub[key].dropna()
            n_ok = len(vals)
            row[f"{key}_n_ok"] = n_ok
            if n_ok == 0:
                row[f"{key}_mean"] = np.nan
                row[f"{key}_ci95"] = np.nan
            elif n_ok == 1:
                row[f"{key}_mean"] = float(vals.iloc[0])
                row[f"{key}_ci95"] = np.nan
            else:
                t_crit = stats.t.ppf(0.975, n_ok - 1)
                row[f"{key}_mean"] = float(vals.mean())
                row[f"{key}_ci95"] = float(t_crit * vals.std(ddof=1) / np.sqrt(n_ok))
        rows.append(row)
    return pd.DataFrame(rows)


def render_table(summary: pd.DataFrame, dataset_name: str) -> str:
    """Markdown table: methods × metrics with mean ± CI and n_ok/n_total."""
    header_metrics = [
        ("k_error",  "K-error %",       lambda v: f"{v*100:.1f}",  lambda c: f"±{c*100:.1f}"),
        ("nlpd",     "NLPD",            lambda v: f"{v:.2f}",      lambda c: f"±{c:.2f}"),
        ("kl",       "KL(fit‖oracle)",  lambda v: f"{v:.2f}",      lambda c: f"±{c:.2f}"),
        ("mll",      "MLL",             lambda v: f"{v:.1f}",      lambda c: f"±{c:.1f}"),
        ("noise_var","noise_var",           lambda v: f"{v:.2e}",      lambda c: f"±{c:.0e}"),
    ]
    header = "| Method | n | " + " | ".join(h[1] for h in header_metrics) + " |"
    sep = "|" + "---|" * (2 + len(header_metrics))
    lines = [f"### {dataset_name}", "", header, sep]
    for _, row in summary.iterrows():
        cells = [row["method"]]
        n_ok_min = min(int(row[f"{k}_n_ok"]) for k, *_ in header_metrics)
        cells.append(f"{n_ok_min}/{int(row['n_total'])}")
        for key, _, fmt_v, fmt_c in header_metrics:
            mean = row[f"{key}_mean"]
            ci = row[f"{key}_ci95"]
            if pd.isna(mean):
                cells.append("—")
            elif pd.isna(ci):
                cells.append(fmt_v(mean))
            else:
                cells.append(f"{fmt_v(mean)} {fmt_c(ci)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_metrics(summary: pd.DataFrame, dataset_name: str, ax_grid):
    """Four panels in a row, one per metric (noise_var omitted, lives in table)."""
    panel_metrics = [
        ("k_error",  "K-error",          True),
        ("nlpd",     "NLPD",             False),
        ("kl",       "KL(fit‖oracle)",   True),
        ("mll",      "MLL",              False),
    ]
    methods = summary["method"].tolist()
    colors = sns.cubehelix_palette(n_colors=len(methods), reverse=True)
    for ax, (key, label, log_y) in zip(ax_grid, panel_metrics):
        means = summary[f"{key}_mean"].values
        if key == "k_error":
            means = means * 100
            label = "K-error %"
        ax.bar(methods, means, color=colors, edgecolor="black", width=0.7)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        if log_y:
            # Use symlog so negative/zero values still render.
            ax.set_yscale("symlog")
    ax_grid[0].set_ylabel(dataset_name, fontsize=11, fontweight="bold")


def make_summary_plot(summary_lsk, summary_hmk, path):
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    plot_metrics(summary_lsk, "Silverman LS", axes[0])
    plot_metrics(summary_hmk, "HMK",          axes[1])
    fig.suptitle("Kernel learning baselines — posterior-quality metrics",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    lsk = LocalStationaryKernel(a=0.5)
    # Specific seeds creates meaning full data / no constant data
    df_lsk = run_benchmark(
        lsk.kernel, "Silverman Locally Stationary",
        seeds=(42, 44, 45, 46, 47), x_lo=-5.0, x_hi=5.0, noise_var=1e-4, include_complex=False,
    )

    # Specific seeds creates meaning full data / no constant data
    df_hmk = run_benchmark(
        hmk_real_kernel, "Harmonizable Mixture Kernel",
        seeds=(42, 43, 45, 46, 47), x_lo=-2.0, x_hi=2.0, noise_var=1e-2,
    )

    summary_lsk = summarise(df_lsk)
    summary_hmk = summarise(df_hmk)

    print("\n" + render_table(summary_lsk, "Silverman LS"))
    print("\n" + render_table(summary_hmk, "HMK"))

    return df_lsk, df_hmk, summary_lsk, summary_hmk


if __name__ == "__main__":
    main()
