# Regular Fourier Features for Nonstationary Gaussian Processes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors**: Abdullah Karatas, Arsalan Jawaid, Joerg Seewig

---

## Overview

We propose regular Fourier features for harmonizable Gaussian processes that discretize the spectral representation on a frequency grid. Existing spectral approaches for nonstationary GPs limit the class of representable kernels. Our method yields a low-rank approximation without modifying the spectral density.

---

## Installation

```bash
git clone https://github.com/AbdullahKaratas/neural-spectral-gp.git
cd neural-spectral-gp
pip install -e .
```

---

## Project structure

```
neural-spectral-gp/
├── src/nsgp/
│   ├── kernel/           # Kernel functions (Silverman, HMK)
│   │   ├── local_stationary.py
│   │   └── hmk.py
│   ├── lowrank/          # Fourier feature approximations
│   │   ├── nff.py        # Regular nonstationary Fourier features
│   │   └── random_nff.py # Random Fourier Feature baseline
│   └── models/           # GP models and spectral density networks
│       ├── sdn_factorized.py
│       ├── standard_gp.py
│       └── remes_baseline.py
├── experiments/
│   ├── low_rank/         # Kernel approximation experiments
│   ├── kernel_learning/  # Spectral density learning experiments
├── tests/
└── pyproject.toml
```

---

## Running experiments

Low-rank kernel approximation (Silverman kernel, HMK):
```bash
python experiments/low_rank/local_stationary_example.py
python experiments/low_rank/hmk_example.py
```

Regular vs random Fourier features comparison:
```bash
python experiments/low_rank/regular_vs_random.py
```

Kernel learning:
```bash
python experiments/kernel_learning/fsdn_vs_rbf.py
```

Tests:
```bash
pytest tests/
```
