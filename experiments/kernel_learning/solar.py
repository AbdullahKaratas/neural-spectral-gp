# Copyright 2016 James Hensman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications: Copyright 2026 Arsalan Jawaid (MIT-licensed)


import math
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt

import gpytorch

from nsgp.models import FactorizedSpectralDensityNetwork, StandardGP

torch.set_default_dtype(torch.float64)
torch.manual_seed(67)


DATA_DIR = Path("experiments/kernel_learning/data")
data = pd.read_csv(
    DATA_DIR / "solar_data.txt", comment="#", sep=",", header=None
).to_numpy()
true_x = data[:, 0].reshape(-1, 1)
true_y = data[:, -1].reshape(-1, 1)

train_x = true_x.copy()
train_y = true_y.copy()

# remove some chunks of data
test_x, test_y = [], []

intervals = ((1620, 1650), (1700, 1720), (1780, 1800), (1850, 1870), (1930, 1950))
for low, up in intervals:
    ind = np.logical_and(train_x.flatten() > low, train_x.flatten() < up)
    test_x.append(train_x[ind])
    test_y.append(train_y[ind])
    train_x = np.delete(train_x, np.where(ind)[0], axis=0)
    train_y = np.delete(train_y, np.where(ind)[0], axis=0)
test_x, test_y = np.vstack(test_x), np.vstack(test_y)

Y_MEAN = train_y.mean()
Y_STD = train_y.std()
YEAR_MIN = train_x.min()
YEAR_MAX = train_x.max()

train_y = (train_y - Y_MEAN) / Y_STD
test_y = (test_y - Y_MEAN) / Y_STD


def to_x(year):
    return (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN) * 10.0 - 5.0


def to_year(x):
    return (x + 5.0) / 10.0 * (YEAR_MAX - YEAR_MIN) + YEAR_MIN


# map to to -5,5
train_x = torch.tensor(
    to_x(train_x.flatten()), dtype=torch.get_default_dtype()
).unsqueeze(-1)
train_y = torch.tensor(train_y.flatten(), dtype=torch.get_default_dtype())
test_x = torch.tensor(
    to_x(test_x.flatten()), dtype=torch.get_default_dtype()
).unsqueeze(-1)
test_y = torch.tensor(test_y.flatten(), dtype=torch.get_default_dtype())


def gap_metrics(model, test_y):
    with torch.no_grad():
        mvn = model._full_pred_dist(test_x, predictive_dist=True)
        return {
            "nlpd": gpytorch.metrics.negative_log_predictive_density(
                mvn, test_y
            ).item(),
            "mae": gpytorch.metrics.mean_absolute_error(mvn, test_y).item(),
            "mse": gpytorch.metrics.mean_squared_error(mvn, test_y).item(),
        }


def plot(model, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    xgrid = torch.linspace(-5.0, 5.0, 300).unsqueeze(-1)
    mean, std = model.predict(xgrid)
    mu, sd = mean.detach().numpy(), std.detach().numpy()
    years = to_year(xgrid.squeeze().numpy())
    for low, up in intervals:
        ax.axvspan(low, up, color="gray", zorder=0, alpha=0.2)
    # posterior mean +/- 2 sigma
    ax.plot(years, mu, color="blue", label="Mean")
    ax.fill_between(
        years,
        mu - 2 * sd,
        mu + 2 * sd,
        color="blue",
        alpha=0.2,
        lw=0,
        label=r"$\pm 2\sigma$",
    )
    # data
    ax.plot(
        to_year(train_x.numpy()).ravel(),
        train_y.numpy().ravel(),
        ".",
        color="black",
        label="Train",
    )
    ax.plot(
        to_year(test_x.numpy()).ravel(),
        test_y.numpy().ravel(),
        ".",
        color="orange",
        label="Test",
    )
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.legend(loc="upper left")


fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)
results = {}

# (a) RBF
gp = StandardGP()
gp.fit(train_x, train_y, epochs=2000, lr=0.01, verbose=True)
plot(gp, axes[0])
axes[0].set_title("(a) RBF")
results["RBF"] = gap_metrics(gp, test_y)

# (b) Neural Network Kernel
#

# (c) F-SDN (ours)
# omega_max: low / underfitting, high / overfitting
omega_max = 32.0
# compute n_features
xmax = train_x.abs().max()
# add double m then required
m = int(omega_max * xmax / math.pi) * 2
n_features = 2 * m - 1

fsdn = FactorizedSpectralDensityNetwork(
    input_dim=1,
    hidden_dims=[128, 128],
    rank=8,
    n_features=n_features,
    omega_max=omega_max,
    enforce_symmetry=False,
    spectral_real=False,
    prior_variance=1.0,
    embedding_dim=256,
    embedding_scale=15.0,
)
with torch.no_grad():
    fsdn.log_noise_var.data = torch.tensor(math.log(1**2))
fsdn.fit(train_x, train_y, epochs=2000, lr=0.01, verbose=True)
plot(fsdn, axes[2])
axes[2].set_title("(c) F-SDN (ours)")
results["F-SDN"] = gap_metrics(fsdn, test_y)

axes[2].set_xlabel("year")
fig.tight_layout()
out = DATA_DIR / "solar_fit.png"
# fig.savefig(out, dpi=100)

print(results)
plt.show()
