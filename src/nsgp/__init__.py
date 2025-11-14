"""
Neural Spectral Gaussian Processes (NSGP)

A hybrid framework combining deep learning with spectral methods
for efficient simulation of nonstationary Gaussian processes.

Authors: Abdullah Karatas, Arsalan Jawaid
"""

__version__ = "0.1.0"

from .models.sdn_factorized import FactorizedSpectralDensityNetwork
from .models.nffs import NFFs

__all__ = [
    "FactorizedSpectralDensityNetwork",
    "NFFs",
]
