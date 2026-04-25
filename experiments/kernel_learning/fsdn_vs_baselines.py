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
from nsgp.metrics import (
    kl_posterior,
    log_pred_density_true_function,
    marginal_log_likelihood,
    negative_log_predictive_density,
    noise_variance,
    oracle_posterior,
)
from nsgp.models import (
    DKLGP,
    FactorizedSpectralDensityNetwork,
    NeuralGSMGP,
    StandardGP,
)


METRIC_KEYS = ["k_error", "nlpd", "kl", "lpd_true", "mll", "noise_var"]


@dataclass
class MethodSpec:
    """A method to evaluate. ``factory`` returns a fresh model on each call."""
    label: str
    factory: Callable[[], object]


def make_methods() -> List[MethodSpec]:
    return [
        MethodSpec("RBF", lambda: StandardGP()),
        MethodSpec("Neural-GSM", lambda: NeuralGSMGP(
            input_dim=1, n_components=2, hidden_dims=[32, 32], prior_variance=1.0,
        )),
        MethodSpec("DKL", lambda: DKLGP(input_dim=1)),
        MethodSpec("F-SDN (real)", lambda: FactorizedSpectralDensityNetwork(
            input_dim=1, hidden_dims=[128, 128], rank=8, n_features=256,
            omega_max=10.0, enforce_symmetry=False, spectral_real=True,
        )),
        MethodSpec("F-SDN (complex)", lambda: FactorizedSpectralDensityNetwork(
            input_dim=1, hidden_dims=[128, 128], rank=8, n_features=256,
            omega_max=10.0, enforce_symmetry=False, spectral_real=False,
        )),
    ]


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
    f_true_test: torch.Tensor,
    K_true_test: torch.Tensor,
    oracle_post,
    epochs: int,
) -> dict:
    """Fit ``spec.factory()`` and compute the six metrics. NaN on any failure."""
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
        out["lpd_true"] = float(log_pred_density_true_function(pred_dist, f_true_test).item())
        out["mll"] = marginal_log_likelihood(model, X_train, y_train)
        out["noise_var"] = noise_variance(model)
    except Exception as exc:  # noqa: BLE001 — single seeds are allowed to fail
        out["error_msg"] = f"{type(exc).__name__}: {exc}"
    return out


def run_single_comparison(
    kernel_fn,
    seed: int,
    n_train: int = 50,
    n_test: int = 100,
    epochs: int = 4000,
    noise_var: float = 1e-4,
) -> List[dict]:
    """Return one dict per method for the given seed (long format)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = torch.linspace(-5, 5, n_train).unsqueeze(-1)
    X_test = torch.linspace(-10, 10, n_test).unsqueeze(-1)

    K_true_train = kernel_fn(X_train, X_train)
    K_true_test = kernel_fn(X_test, X_test)

    L = torch.linalg.cholesky(K_true_train + noise_var * torch.eye(n_train))
    f_train = (L @ torch.randn(n_train)).squeeze()
    y_train = f_train + math.sqrt(noise_var) * torch.randn(n_train)

    # f_true(X_test) drawn jointly with f_train, so the oracle posterior and
    # f_true_test are mutually consistent.
    K_joint_cross = kernel_fn(X_test, X_train)
    K_joint_test = kernel_fn(X_test, X_test)
    A = K_true_train + noise_var * torch.eye(n_train)
    L_A = torch.linalg.cholesky(A)
    alpha = torch.cholesky_solve(f_train.unsqueeze(-1), L_A).squeeze(-1)
    cond_mean = K_joint_cross @ alpha
    V = torch.cholesky_solve(K_joint_cross.transpose(-1, -2), L_A)
    cond_cov = K_joint_test - K_joint_cross @ V
    cond_cov = 0.5 * (cond_cov + cond_cov.transpose(-1, -2))
    cond_cov = cond_cov + 1e-6 * torch.eye(n_test)
    f_true_test = cond_mean + torch.linalg.cholesky(cond_cov) @ torch.randn(n_test)
    y_test = f_true_test + math.sqrt(noise_var) * torch.randn(n_test)

    oracle_post = oracle_posterior(kernel_fn, X_train, y_train, X_test, noise_var=noise_var)

    rows = []
    for spec in make_methods():
        row = _evaluate_one_method(
            spec, X_train, y_train, X_test, y_test, f_true_test, K_true_test,
            oracle_post, epochs,
        )
        row["seed"] = seed
        rows.append(row)
    return rows


def run_benchmark(kernel_fn, kernel_name: str, n_seeds: int = 5) -> pd.DataFrame:
    print(f"Benchmark: {kernel_name}")
    all_rows = []
    for i in range(n_seeds):
        seed = i + 42
        print(f"Seed {i+1}/{n_seeds} (seed={seed})")
        rows = run_single_comparison(kernel_fn, seed=seed)
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
        ("lpd_true", "LPD(f_true)",     lambda v: f"{v:.1f}",      lambda c: f"±{c:.1f}"),
        ("mll",      "MLL",             lambda v: f"{v:.1f}",      lambda c: f"±{c:.1f}"),
        ("noise_var","σ²_noise",        lambda v: f"{v:.2e}",      lambda c: f"±{c:.0e}"),
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
    """Five panels in a row, one per metric (noise_var omitted, lives in table)."""
    panel_metrics = [
        ("k_error",  "K-error",          True),
        ("nlpd",     "NLPD",             False),
        ("kl",       "KL(fit‖oracle)",   True),
        ("lpd_true", "LPD(f_true)",      False),
        ("mll",      "MLL",              False),
    ]
    methods = summary["method"].tolist()
    colors = sns.cubehelix_palette(n_colors=len(methods), reverse=True)
    for ax, (key, label, log_y) in zip(ax_grid, panel_metrics):
        means = summary[f"{key}_mean"].values
        cis = np.where(pd.isna(summary[f"{key}_ci95"].values), 0.0,
                       summary[f"{key}_ci95"].values)
        if key == "k_error":
            means = means * 100
            cis = cis * 100
            label = "K-error %"
        ax.bar(methods, means, yerr=cis, capsize=4,
               color=colors, edgecolor="black", width=0.7)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        if log_y:
            # Use symlog so negative/zero values still render.
            ax.set_yscale("symlog")
    ax_grid[0].set_ylabel(dataset_name, fontsize=11, fontweight="bold")


def make_summary_plot(summary_lsk, summary_hmk, path):
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    plot_metrics(summary_lsk, "Silverman LS", axes[0])
    plot_metrics(summary_hmk, "HMK",          axes[1])
    fig.suptitle("Kernel learning baselines — posterior-quality metrics",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(n_seeds: int = 5):
    lsk = LocalStationaryKernel(a=0.5)
    df_lsk = run_benchmark(lsk.kernel, "Silverman Locally Stationary", n_seeds=n_seeds)
    df_hmk = run_benchmark(hmk_real_kernel, "Harmonizable Mixture Kernel", n_seeds=n_seeds)

    summary_lsk = summarise(df_lsk)
    summary_hmk = summarise(df_hmk)

    print("\n" + render_table(summary_lsk, "Silverman LS"))
    print("\n" + render_table(summary_hmk, "HMK"))

    return df_lsk, df_hmk, summary_lsk, summary_hmk


if __name__ == "__main__":
    main()
