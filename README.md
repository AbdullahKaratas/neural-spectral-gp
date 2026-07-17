# Regular Fourier Features for Nonstationary Gaussian Processes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors**: Arsalan Jawaid, Abdullah Karatas, Joerg Seewig

---

## Overview

We propose regular Fourier features for harmonizable Gaussian processes that discretize the spectral representation on a frequency grid. Existing spectral approaches for nonstationary GPs limit the class of representable kernels. Our method yields a low-rank approximation without modifying the spectral density.

---

## Development / Installation

```bash
cd neural-spectral-gp
pip install -e ".[develop]"
pre-commit install
```

---

## Project structure

```
neural-spectral-gp/
├── src/nsgp/
│   ├── kernel/                 # Kernel functions (LS, HMK, Neural-GSM, NNK, DKL)
│   │   ├── local_stationary.py
│   │   ├── hmk.py
│   │   ├── neural_gsm.py
│   │   ├── neural_network_kernel.py
│   │   └── deep_kernel.py
│   ├── lowrank/                # Fourier feature approximations
│   │   ├── regular_nff.py      # Regular Fourier Features (Ours)
│   │   └── random_nff.py       # Random Fourier Feature
│   ├── models/                 # GP models
│   │   ├── sdn_factorized.py   # Factorized Spectral Density Network (Ours)
│   │   ├── standard_gp.py      # Exact GP Model with configurable kernel
│   │   ├── neural_gsm_gp.py    # Exact GP Model with Neural-GSM
│   │   └── dkl_gp.py           # DKL GP
│   ├── metrics.py
│   └── utils.py
├── experiments/
│   ├── low_rank/         # Kernel approximation experiments
│   └── kernel_learning/  # Kernel learning experiments
├── tests/
└── pyproject.toml
```

---

## Running experiments

Low-rank kernel approximation:
```bash
python experiments/low_rank/local_stationary_example.py # LS
python experiments/low_rank/hmk_example.py              # HMK
python experiments/low_rank/ablation_studies.py         # Ablation Studies
python experiments/low_rank/regular_vs_random.py        # Regular vs Random Fourier
python experiments/low_rank/speedup_error.py            # Speed-up against Cholesky
```

Kernel learning:
```bash
python experiments/kernel_learning/solar.py              # FSDN vs RBF vs NNK kernel on the solar dataset
python experiments/kernel_learning/fsdn_vs_baselines.py  # FSDN vs RBF, NNK, Neural-GSM, DKL on synthetic data (LS + HMK)
```

Tests:
```bash
pytest tests/
```

---

## Citation

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata, or use:

```bibtex
@article{jawaid2026regular,
  title={Regular Fourier Features for Nonstationary Gaussian Processes},
  author={Jawaid, Arsalan and Karatas, Abdullah and Seewig, Joerg},
  journal={arXiv preprint},
  year={2026}
}
```
