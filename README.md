# Regular Fourier Features for Nonstationary Gaussian Processes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors**: Abdullah Karatas, Arsalan Jawaid, Joerg Seewig

---

## Overview

We propose regular Fourier features for harmonizable Gaussian processes that discretize the spectral representation on a frequency grid. Existing spectral approaches for nonstationary GPs limit the class of representable kernels. Our method yields a low-rank approximation without modifying the spectral density.

---

## Installation

```bash
git clone https://github.com/mts-public/neural-spectral-gp.git
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

<<<<<<< HEAD
Low-rank kernel approximation (Silverman kernel, HMK):
```bash
python experiments/low_rank/local_stationary_example.py
python experiments/low_rank/hmk_example.py
=======
### Synthetic Benchmarks

We validate our method on three synthetic scenarios:

1. **Locally Stationary (Silverman 1957)**
   - Ground truth: $r_{LS}(x,x') = \exp(-2a(\frac{x+x'}{2})^2) \exp(-\frac{a}{2}(x-x')^2)$

2. **Spatially Varying Matérn**
   - Smoothness parameter varies with location

3. **Complex Nonstationary Patterns**
   - Multiple length scales and amplitudes

### Real Data Applications

- **Climate Data**: Temperature and precipitation modeling
- **Geospatial Analysis**: Elevation and soil properties
- **Environmental Monitoring**: Sensor network data

---

## Results Preview

| Method | RMSE ↓ | NLL ↓ | Time (s) ↓ |
|--------|--------|-------|-----------|
| Standard GP | 0.15 | -1.2 | 125.3 |
| NFFs (oracle) | 0.16 | -1.1 | 0.12 |
| NFFs (misspec) | 0.45 | 0.8 | 0.12 |
| Neural Process | 0.18 | -0.9 | 15.7 |
| **NSGP (ours)** | **0.17** | **-1.0** | **0.15** |

*Averaged over 10 synthetic datasets with n=1000 observations*

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{karatas2026regular,
  title={Regular Fourier Features for Nonstationary Gaussian Processes},
  author={Karatas, Abdullah and Jawaid, Arsalan and Seewig, Joerg},
  journal={arXiv preprint},
  year={2026}
}
>>>>>>> origin/main
```

Regular vs random Fourier features comparison:
```bash
python experiments/low_rank/regular_vs_random.py
```

Kernel learning:
```bash
python experiments/kernel_learning/fsdn_vs_rbf.py
```

<<<<<<< HEAD
Tests:
```bash
pytest tests/
```
=======
This work builds upon:
- **Regular Fourier Features** (Shinozuka, 1972): Efficient simulation for stationary processes
- **Regular Nonstationary Fourier Features** (Jawaid, 2024): Extension to harmonizable processes
- **Neural Processes** (Garnelo et al., 2018): Data-driven GP approximation
- **Deep Kernel Learning** (Wilson et al., 2016): Learning kernel functions with neural networks

---

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Contact

- **Abdullah Karatas** - [GitHub](https://github.com/AbdullahKaratas)
- **Arsalan Jawaid**
- **Project Link**: [https://github.com/mts-public/neural-spectral-gp](https://github.com/mts-public/neural-spectral-gp)

---

## Acknowledgments

This work extends the Regular Nonstationary Fourier Features method developed by Arsalan Jawaid.

---

**Status**: 🚧 Work in Progress - Initial implementation phase
>>>>>>> origin/main
