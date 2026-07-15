import math
import time
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd
import torch
import linear_operator
from linear_operator.operators import (
    DenseLinearOperator,
    DiagLinearOperator,
    LowRankRootLinearOperator,
)

from nsgp.kernel import HarmonizableMixtureKernel, LocalStationaryKernel
from nsgp.lowrank import RegularNonstationaryFeatures

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

REPEATS = 20
WARMUP = 1
SIGMA2 = 0.1
N_GRID = [2000, 4000, 6000, 8000]
M_GRID = [100, 120, 140, 160, 180, 200]


def relative_error(K_true, K_approx):
    return (torch.norm(K_true - K_approx) / torch.norm(K_true)).item() * 100.0


def mean_ci(samples):
    """Mean and 95% Student-t CI half-width of the mean."""
    arr = np.asarray(samples)
    sem = arr.std(ddof=1) / np.sqrt(len(arr))
    return float(arr.mean()), float(stats.t.ppf(0.975, len(arr) - 1) * sem)


def build_hmk():
    eta = torch.tensor([[1.0]])
    B = torch.tensor([[2.0 + 0j, 0.5j], [-0.5j, 2.0 + 0j]])
    return HarmonizableMixtureKernel(
        sigma1=torch.eye(1) / math.pi**2,
        sigma2=torch.eye(1) / (2 * math.pi) ** 2,
        centers=[torch.zeros(1)],
        scalings=[torch.ones(1)],
        frequencies=[torch.cat([eta, -eta])],
        psd_matrices=[B],
    )


def run_kernel(name, true_kernel, spectral, spectral_real, x, omega_max, n):
    x_col = x.reshape(-1, 1)
    y = torch.randn(n)
    with torch.no_grad():
        K_true = true_kernel(x_col)

    def exact_solve(K):
        op = DenseLinearOperator(K) + DiagLinearOperator(
            SIGMA2 * torch.ones(n, dtype=K.dtype)
        )
        op.inv_quad_logdet(inv_quad_rhs=y.unsqueeze(-1), logdet=True)

    def exact_lsk_step():
        exact_solve(true_kernel(x_col))

    def exact_hmk_step():
        exact_solve(true_kernel(x_col).real)

    exact_step = exact_lsk_step if spectral_real else exact_hmk_step

    def lowrank_step(nff, spacing):
        L = nff.lowrank(x_col, spacing=spacing)
        op = LowRankRootLinearOperator(L) + DiagLinearOperator(
            SIGMA2 * torch.ones(n, dtype=L.dtype)
        )
        op.inv_quad_logdet(inv_quad_rhs=y.unsqueeze(-1), logdet=True)

    specs = []
    for m in M_GRID:
        spacing = omega_max / m
        if spacing >= torch.pi / x.abs().max().item():
            raise RuntimeError(f"aliasing violated: n={n}, m={m}")
        nff = RegularNonstationaryFeatures(
            spectral=spectral, spectral_real=spectral_real, num_feat=m
        )
        specs.append((m, nff, spacing))

    # timings
    exact_t = []
    low_t = {m: [] for m, _, _ in specs}
    for it in range(WARMUP + REPEATS):
        with torch.no_grad(), linear_operator.settings.max_cholesky_size(n + 1):
            t0 = time.perf_counter()
            exact_step()
            e = time.perf_counter() - t0
            per_m = {}
            for m, nff, spacing in specs:
                t1 = time.perf_counter()
                lowrank_step(nff, spacing)
                per_m[m] = time.perf_counter() - t1
        if it >= WARMUP:
            exact_t.append(e)
            for m, t in per_m.items():
                low_t[m].append(t)

    exact_arr = np.asarray(exact_t)
    rows = []
    for m, nff, spacing in specs:
        # speedup
        speedup, ci = mean_ci(exact_arr / np.asarray(low_t[m]))

        # relative error
        with torch.no_grad():
            error = relative_error(
                K_true, nff.kernel_estimate(x_col, x_col, spacing=spacing)
            )
        rows.append(
            {
                "kernel": name,
                "n": n,
                "m": m,
                "error_percent": error,
                "speedup": speedup,
                "speedup_lo": speedup - ci,
                "speedup_hi": speedup + ci,
                "exact_time_s": float(exact_arr.mean()),
                "lowrank_time_s": float(np.mean(low_t[m])),
            }
        )
    return rows


torch.manual_seed(0)
rows = []
lsk = LocalStationaryKernel(a=1.0)
for n in N_GRID:
    rows += run_kernel(
        "LSK",
        lambda xc: lsk.kernel(xc, xc),
        lsk.spectral,
        True,
        torch.linspace(0.0, 2.5, n),
        5.0,
        n,
    )
hmk = build_hmk()
for n in N_GRID:
    rows += run_kernel(
        "HMK",
        lambda xc: hmk.k_hmk(xc.squeeze(-1), xc.squeeze(-1)),
        hmk.s_khm,
        False,
        torch.linspace(-3.0, 3.0, n),
        20.0,
        n,
    )

df = pd.DataFrame(rows)
df.to_csv(DATA_DIR / "speedup_error.csv", index=False)
