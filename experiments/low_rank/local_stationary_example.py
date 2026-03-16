import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from nsgp.lowrank import NonstationaryFeatures
from nsgp.kernel import LocalStationaryKernel


def to_csv(obj, path):
    arr = (
        obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
    )
    pd.DataFrame(arr).to_csv(path, index=False, header=False, float_format="%.10e")

    return None


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

lsk = LocalStationaryKernel(a=1.0)

with torch.no_grad():
    cutoff1 = 5
    num_feat = 20
    num_feat_sk = 190
    delta_omega1 = cutoff1 / num_feat
    delta_x = 0.001
    n_xdata = 2500
    print("Check spectral spacing", delta_omega1 <= 1.0 * torch.pi / n_xdata / delta_x)

    x = torch.arange(n_xdata).reshape(-1, 1) * delta_x
    omega1 = torch.arange(num_feat).reshape(-1, 1) * delta_omega1
    omega = torch.arange(num_feat_sk).reshape(-1, 1) * delta_omega1 / 10.0

    nff = NonstationaryFeatures(
        spectral=lsk.spectral, spectral_real=True, num_feat=num_feat
    )
    kestimate1 = nff.kernel_estimate(x, x, spacing=delta_omega1)
    ktrue = lsk.kernel(x, x)

    spectral_true = lsk.spectral(omega, omega)
    spectral1 = lsk.spectral(omega1, omega1)

error1 = torch.abs(kestimate1 - ktrue)

oo1_x, oo1_y = np.meshgrid(omega1, omega1)
file_map = {
    "k_true.csv": ktrue,
    "k_estimate1.csv": kestimate1,
    "error1.csv": error1,
    "spectral_true.csv": spectral_true,
    "omega1_mesh_x.csv": oo1_x,
    "omega1_mesh_y.csv": oo1_y,
}

for fname, obj in file_map.items():
    to_csv(obj, DATA_DIR / fname)

# Create colormaps
N = 256
cubehelix_cmap = sns.cubehelix_palette(n_colors=N, as_cmap=True, reverse=True)
gray_cmap = sns.color_palette("gray", n_colors=N, as_cmap=True)

# Generate PNG figures
dpi = 300
x_max = (n_xdata - 1) * delta_x

# Approximation with cubehelix
fig, ax = plt.subplots(
    figsize=(kestimate1.shape[1] / dpi, kestimate1.shape[0] / dpi), dpi=dpi
)
ax.imshow(
    kestimate1.numpy(),
    origin="lower",
    cmap=cubehelix_cmap,
    vmin=0,
    vmax=1,
    extent=[0, x_max, 0, x_max],
    aspect="auto",
)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(
    "data/k_estimate1.png", dpi=dpi, bbox_inches="tight", pad_inches=0
)
plt.close()

# Error with cubehelix
fig, ax = plt.subplots(figsize=(error1.shape[0] / dpi, error1.shape[1] / dpi), dpi=dpi)
ax.imshow(
    error1.numpy() * 1000,
    origin="lower",
    cmap=cubehelix_cmap,
    vmin=0,
    vmax=1.2,
    extent=[0, x_max, 0, x_max],
    aspect="auto",
)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("data/error1.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
plt.close()

# True kernel with gray
fig, ax = plt.subplots(figsize=(ktrue.shape[1] / dpi, ktrue.shape[0] / dpi), dpi=dpi)
ax.imshow(
    ktrue.numpy(),
    origin="lower",
    cmap=gray_cmap,
    vmin=0,
    vmax=1,
    extent=[0, x_max, 0, x_max],
    aspect="auto",
)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("data/k_true.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
plt.close()

# Spectral density with gray
omega_max = (num_feat - 1) * delta_omega1
fig, ax = plt.subplots(
    figsize=(spectral_true.shape[1] / dpi, spectral_true.shape[0] / dpi), dpi=dpi
)
ax.imshow(
    spectral_true.numpy(),
    origin="lower",
    cmap=gray_cmap,
    vmin=0,
    vmax=0.08,
    extent=[0, omega_max, 0, omega_max],
    aspect="auto",
)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("data/spectral.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
plt.close()

# Save mesh points for spectral density overlay
mesh_points = np.column_stack([oo1_x.flatten(), oo1_y.flatten()])
np.savetxt("data/mesh_points.dat", mesh_points, fmt="%.6f")
