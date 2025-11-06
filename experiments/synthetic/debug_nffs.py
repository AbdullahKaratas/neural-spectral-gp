"""
Debug NFFs implementation

Check intermediate values to find the bug.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nsgp.models.nffs import NFFs


def silverman_spectral_density(omega1: torch.Tensor, omega2: torch.Tensor, a: float = 0.5) -> torch.Tensor:
    """Silverman's spectral density."""
    if omega1.dim() == 1:
        omega1 = omega1.unsqueeze(-1)
    if omega2.dim() == 1:
        omega2 = omega2.unsqueeze(-1)

    omega_sum = (omega1 + omega2) / 2.0
    omega_diff = omega1 - omega2

    prefactor = 1.0 / (4.0 * np.pi * a)
    exp1 = torch.exp(-1.0 / (2.0 * a) * omega_sum**2)
    exp2 = torch.exp(-1.0 / (8.0 * a) * omega_diff**2)

    return (prefactor * exp1 * exp2).squeeze()


def debug_nffs():
    print("=" * 70)
    print("Debugging NFFs")
    print("=" * 70)

    a = 0.5
    n_features = 100
    omega_max = 10.0

    x = torch.linspace(-3, 3, 100).unsqueeze(-1)

    def spectral_density_fn(w1, w2):
        return silverman_spectral_density(w1, w2, a=a)

    nffs = NFFs(
        spectral_density=spectral_density_fn,
        n_features=n_features,
        omega_max=omega_max,
        input_dim=1
    )

    # Check spectral matrix
    S = nffs._compute_spectral_matrix()
    print(f"\nSpectral matrix S:")
    print(f"  Shape: {S.shape}")
    print(f"  Diagonal mean: {torch.diag(S).mean().item():.6f}")
    print(f"  Diagonal std: {torch.diag(S).std().item():.6f}")
    print(f"  Off-diagonal mean: {(S.sum() - torch.diag(S).sum()).item() / (S.numel() - S.shape[0]):.6f}")
    print(f"  Matrix norm: {torch.norm(S).item():.6f}")
    print(f"  Trace: {torch.trace(S).item():.6f}")

    # Check Cholesky
    L = nffs._whiten(S)
    print(f"\nCholesky factor L:")
    print(f"  Shape: {L.shape}")
    print(f"  Norm: {torch.norm(L).item():.6f}")

    # Verify L·L^T ≈ S
    S_reconstructed = L @ L.T
    error = torch.norm(S - S_reconstructed).item()
    print(f"  Reconstruction error ||S - LL^T||: {error:.6e}")

    # Check white noise
    M = n_features
    xi = torch.randn(1000, M, 2)  # Many samples for statistics
    print(f"\nWhite noise ξ:")
    print(f"  Shape: {xi.shape}")
    print(f"  Mean (Real): {xi[:, :, 0].mean().item():.6f}")
    print(f"  Mean (Imag): {xi[:, :, 1].mean().item():.6f}")
    print(f"  Var (Real): {xi[:, :, 0].var().item():.6f}")
    print(f"  Var (Imag): {xi[:, :, 1].var().item():.6f}")

    # Check transformed noise
    xi_transformed = torch.einsum('sm,nmc->nsc', L, xi)
    print(f"\nTransformed noise ξ̃ = L·ξ:")
    print(f"  Shape: {xi_transformed.shape}")
    print(f"  Mean (Real): {xi_transformed[:, :, 0].mean().item():.6f}")
    print(f"  Mean (Imag): {xi_transformed[:, :, 1].mean().item():.6f}")
    print(f"  Var (Real): {xi_transformed[:, :, 0].var().item():.6f}")
    print(f"  Var (Imag): {xi_transformed[:, :, 1].var().item():.6f}")

    # Check expected variance of transformed noise
    # Should be roughly trace(S) / M
    expected_var = torch.trace(S).item() / M
    print(f"  Expected var (trace(S)/M): {expected_var:.6f}")

    # Check volume normalization
    volume = (2 * omega_max) ** 1 / M
    print(f"\nVolume element Δω:")
    print(f"  Δω = {volume:.6f}")
    print(f"  sqrt(Δω) = {np.sqrt(volume):.6f}")

    # Generate samples and check variance
    samples = nffs.simulate(x, n_samples=1000, seed=42)
    print(f"\nGenerated samples:")
    print(f"  Shape: {samples.shape}")
    print(f"  Mean: {samples.mean().item():.6f}")
    print(f"  Var (empirical): {samples.var().item():.6f}")

    # True variance at x=0
    true_var_at_0 = np.exp(0)  # r(0,0) = exp(-2a·0)·exp(0) = 1
    print(f"  True var at x=0: {true_var_at_0:.6f}")

    # Compute theoretical variance from formula
    # Var(Z(x)) should be related to integral of s(ω,ω')
    print(f"\nTheoretical check:")
    print(f"  Volume × Trace(S) = {volume * torch.trace(S).item():.6f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    debug_nffs()
