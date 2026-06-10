import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from pathlib import Path
from nsgp.kernel import HarmonizableMixtureKernel
from nsgp.lowrank import RegularNonstationaryFeatures


def to_csv(obj, path):
    arr = (
        obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
    )
    pd.DataFrame(arr).to_csv(path, index=False, header=False, float_format="%.10e")
    return None


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Set up parameters
d = 1
# Q conjugate frequency pairs
Q = 1
# HMK components
P = 1

# Conjugate frequency pairs
eta = torch.tensor([[1.0]])
frequencies = torch.cat([eta, -eta], dim=0)

B = torch.tensor([[2.0 + 0.0j, 0.0 + 0.5j], [0.0 - 0.5j, 2.0 + 0.0j]])

# HMK kernel with specific parameters to match Silverman's kernel
sigma1 = torch.eye(d) * (1.0 / (math.pi**2))
sigma2 = torch.eye(d) * (1.0 / (2.0 * math.pi) ** 2)

centers = [torch.zeros(d)]
scalings = [torch.ones(d)]
frequencies_list = [frequencies]
psd_matrices = [B]

hmk = HarmonizableMixtureKernel(
    sigma1=sigma1,
    sigma2=sigma2,
    centers=centers,
    scalings=scalings,
    frequencies=frequencies_list,
    psd_matrices=psd_matrices,
)

omega_max = 20.0
num_feat = 100
delta_omega = omega_max / num_feat

# True spectral density
omega_range = torch.arange(-num_feat * 10 + 1, num_feat * 10) * delta_omega / 10
omega1_grid, omega2_grid = torch.meshgrid(omega_range, omega_range)
s_true = hmk.s_khm(omega_range, omega_range)

# True kernel
x_range = torch.arange(-300 + 1, 300) * 3.0 / 300
x1_grid, x2_grid = torch.meshgrid(x_range, x_range)
k_true = hmk.k_hmk(x_range, x_range)

# Verify aliasing
x_max = x_range.abs().max()
required_delta_omega = math.pi / x_max
print(f"Aliasing satisfied: {delta_omega <= required_delta_omega}")

# NFF approximation
nff = RegularNonstationaryFeatures(
    spectral=hmk.s_khm, spectral_real=False, num_feat=num_feat
)
k_estimate = nff.kernel_estimate(
    x_range.reshape(-1, 1), x_range.reshape(-1, 1), spacing=delta_omega
)

abs_error = torch.abs(k_true - k_estimate) * 1e5

# Save data to CSV files
omega_mesh_x, omega_mesh_y = np.meshgrid(omega_range.numpy(), omega_range.numpy())
x_mesh_x, x_mesh_y = np.meshgrid(x_range.numpy(), x_range.numpy())

file_map = {
    "hmk_spectral_real.csv": s_true.real,
    "hmk_spectral_imag.csv": s_true.imag,
    "hmk_spectral_abs.csv": torch.abs(s_true),
    "hmk_k_true.csv": k_true.real,
    "hmk_k_estimate.csv": k_estimate.real,
    "hmk_error.csv": abs_error,
    "hmk_omega_mesh_x.csv": omega_mesh_x,
    "hmk_omega_mesh_y.csv": omega_mesh_y,
    "hmk_x_mesh_x.csv": x_mesh_x,
    "hmk_x_mesh_y.csv": x_mesh_y,
}

for fname, obj in file_map.items():
    to_csv(obj, DATA_DIR / fname)

# Save mesh points for spectral density and kernel overlays
mesh_points_omega = np.column_stack([omega_mesh_x.flatten(), omega_mesh_y.flatten()])
mesh_points_x = np.column_stack([x_mesh_x.flatten(), x_mesh_y.flatten()])
np.savetxt(DATA_DIR / "hmk_omega_mesh_points.dat", mesh_points_omega, fmt="%.6f")

# Generate PNG figures for tikz
dpi = 300
N = 256
cubehelix_cmap = sns.cubehelix_palette(n_colors=N, as_cmap=True, reverse=True)
gray_cmap = sns.color_palette("gray", n_colors=N, as_cmap=True)

omega_min = omega_range.min().item()
omega_max_val = omega_range.max().item()
x_min = x_range.min().item()
x_max_val = x_range.max().item()

# Spectral density (real)
fig, ax = plt.subplots(
    figsize=(s_true.real.shape[1] / dpi, s_true.real.shape[0] / dpi), dpi=dpi
)
ax.imshow(
    s_true.real.numpy(),
    origin="lower",
    cmap=gray_cmap,
    vmin=-0.04,
    vmax=0.16,
    extent=[omega_min, omega_max_val, omega_min, omega_max_val],
    aspect="auto",
)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    DATA_DIR / "hmk_spectral_real.png", dpi=dpi, bbox_inches="tight", pad_inches=0
)
plt.close()

# Spectral density (imaginary)
fig, ax = plt.subplots(
    figsize=(s_true.imag.shape[1] / dpi, s_true.imag.shape[0] / dpi), dpi=dpi
)
ax.imshow(
    s_true.imag.numpy(),
    origin="lower",
    cmap=gray_cmap,
    vmin=-0.04,
    vmax=0.16,
    extent=[omega_min, omega_max_val, omega_min, omega_max_val],
    aspect="auto",
)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    DATA_DIR / "hmk_spectral_imag.png", dpi=dpi, bbox_inches="tight", pad_inches=0
)
plt.close()

# Spectral density (absolute)
fig, ax = plt.subplots(figsize=(s_true.shape[1] / dpi, s_true.shape[0] / dpi), dpi=dpi)
ax.imshow(
    torch.abs(s_true).numpy(),
    origin="lower",
    cmap=gray_cmap,
    vmin=-0.04,
    vmax=0.16,
    extent=[omega_min, omega_max_val, omega_min, omega_max_val],
    aspect="auto",
)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    DATA_DIR / "hmk_spectral_abs.png", dpi=dpi, bbox_inches="tight", pad_inches=0
)
plt.close()

# True kernel
fig, ax = plt.subplots(
    figsize=(k_true.real.shape[1] / dpi, k_true.real.shape[0] / dpi), dpi=dpi
)
ax.imshow(
    k_true.real.numpy(),
    origin="lower",
    cmap=gray_cmap,
    vmin=-4.9,
    vmax=4.9,
    extent=[x_min, x_max_val, x_min, x_max_val],
    aspect="auto",
)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(DATA_DIR / "hmk_k_true.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
plt.close()

# Approximated kernel
fig, ax = plt.subplots(
    figsize=(k_estimate.shape[1] / dpi, k_estimate.shape[0] / dpi), dpi=dpi
)
ax.imshow(
    k_estimate.numpy(),
    origin="lower",
    cmap=cubehelix_cmap,
    vmin=-4.9,
    vmax=4.9,
    extent=[x_min, x_max_val, x_min, x_max_val],
    aspect="auto",
)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(DATA_DIR / "hmk_k_estimate.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
plt.close()

# Error
fig, ax = plt.subplots(
    figsize=(abs_error.shape[0] / dpi, abs_error.shape[1] / dpi), dpi=dpi
)
ax.imshow(
    abs_error.numpy(),
    origin="lower",
    cmap=cubehelix_cmap,
    vmin=0,
    vmax=1.4,
    extent=[x_min, x_max_val, x_min, x_max_val],
    aspect="auto",
)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(DATA_DIR / "hmk_error.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
plt.close()
