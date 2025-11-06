"""
Test NFFs with Silverman's Locally Stationary Process

This script validates the NFFs implementation using the exact spectral
density from Silverman (1957) as described in Chapter 6.

Ground truth:
- ACVF: r_LS(x,x') = exp(-2a((x+x')/2)²) * exp(-a/2(x-x')²)
- Spectral: s_LS(ω,ω') = 1/(4πa) * exp(-1/(2a)((ω+ω')/2)²) * exp(-1/(8a)(ω-ω')²)

Authors: Abdullah Karatas, Arsalan Jawaid
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nsgp.models.nffs import NFFs


def silverman_spectral_density(omega1: torch.Tensor, omega2: torch.Tensor, a: float = 0.5) -> torch.Tensor:
    """
    Silverman's locally stationary spectral density from Chapter 6.

    s_LS(ω,ω') = 1/(4πa) * exp(-1/(2a)((ω+ω')/2)²) * exp(-1/(8a)(ω-ω')²)

    Parameters
    ----------
    omega1 : torch.Tensor, shape (n, 1) or (n,)
        First frequency
    omega2 : torch.Tensor, shape (n, 1) or (n,)
        Second frequency
    a : float
        Parameter controlling nonstationary strength (a > 0)

    Returns
    -------
    torch.Tensor, shape (n,)
        Spectral density values
    """
    # Ensure proper shapes
    if omega1.dim() == 1:
        omega1 = omega1.unsqueeze(-1)
    if omega2.dim() == 1:
        omega2 = omega2.unsqueeze(-1)

    # Compute (ω+ω')/2 and (ω-ω')
    omega_sum = (omega1 + omega2) / 2.0
    omega_diff = omega1 - omega2

    # Silverman's spectral density
    prefactor = 1.0 / (4.0 * np.pi * a)
    exp1 = torch.exp(-1.0 / (2.0 * a) * omega_sum**2)
    exp2 = torch.exp(-1.0 / (8.0 * a) * omega_diff**2)

    s = prefactor * exp1 * exp2

    return s.squeeze()


def silverman_acvf(x1: torch.Tensor, x2: torch.Tensor, a: float = 0.5) -> torch.Tensor:
    """
    Silverman's locally stationary ACVF (ground truth).

    r_LS(x,x') = exp(-2a((x+x')/2)²) * exp(-a/2(x-x')²)

    Parameters
    ----------
    x1, x2 : torch.Tensor
        Spatial locations
    a : float
        Parameter (same as spectral density)

    Returns
    -------
    torch.Tensor
        ACVF values
    """
    x_mean = (x1 + x2) / 2.0
    x_diff = x1 - x2

    r = torch.exp(-2.0 * a * x_mean**2) * torch.exp(-a / 2.0 * x_diff**2)

    return r


def test_nffs_simulation():
    """
    Test NFFs simulation with Silverman's spectral density.
    """
    print("=" * 70)
    print("Testing NFFs with Silverman's Locally Stationary Process")
    print("=" * 70)

    # Parameters
    a = 0.5  # Silverman parameter
    n_features = 100  # Number of Fourier features M
    omega_max = 10.0  # Cutoff frequency
    n_samples = 5  # Number of sample paths

    # Test locations
    x = torch.linspace(-3, 3, 100).unsqueeze(-1)

    print(f"\nParameters:")
    print(f"  a = {a}")
    print(f"  M (Fourier features) = {n_features}")
    print(f"  ω_max (cutoff) = {omega_max}")
    print(f"  n_samples = {n_samples}")
    print(f"  Test points: {len(x)}")

    # Create spectral density function
    def spectral_density_fn(w1, w2):
        return silverman_spectral_density(w1, w2, a=a)

    # Initialize NFFs
    print("\n" + "-" * 70)
    print("Initializing NFFs...")
    nffs = NFFs(
        spectral_density=spectral_density_fn,
        n_features=n_features,
        omega_max=omega_max,
        input_dim=1
    )
    print(f"✓ NFFs initialized with {len(nffs.omegas)} frequencies")

    # Simulate
    print("\n" + "-" * 70)
    print("Generating samples...")
    try:
        samples = nffs.simulate(x, n_samples=n_samples, seed=42)
        print(f"✓ Samples generated: shape = {samples.shape}")
    except Exception as e:
        print(f"✗ Error during simulation: {e}")
        raise

    # Compute empirical covariance from samples
    print("\n" + "-" * 70)
    print("Computing empirical statistics...")

    # Mean (should be ~0)
    empirical_mean = samples.mean(dim=0)
    print(f"  Empirical mean: {empirical_mean.mean().item():.6f} (should be ≈0)")

    # Variance at different locations
    empirical_var = samples.var(dim=0)

    # Ground truth variance (r(x,x))
    true_var = silverman_acvf(x.squeeze(), x.squeeze(), a=a)

    print(f"  Empirical variance (avg): {empirical_var.mean().item():.4f}")
    print(f"  True variance (avg): {true_var.mean().item():.4f}")

    # Visualization
    print("\n" + "-" * 70)
    print("Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Sample paths
    ax = axes[0, 0]
    for i in range(n_samples):
        ax.plot(x.squeeze().numpy(), samples[i].numpy(), alpha=0.7, label=f'Sample {i+1}')
    ax.set_xlabel('x')
    ax.set_ylabel('Z(x)')
    ax.set_title('Sample Paths from NFFs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Variance comparison
    ax = axes[0, 1]
    ax.plot(x.squeeze().numpy(), true_var.numpy(), 'r-', linewidth=2, label='True variance r(x,x)')
    ax.plot(x.squeeze().numpy(), empirical_var.numpy(), 'b--', linewidth=2, label='Empirical variance')
    ax.set_xlabel('x')
    ax.set_ylabel('Variance')
    ax.set_title('Variance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Spectral density heatmap
    ax = axes[1, 0]
    omega_grid = torch.linspace(-omega_max/2, omega_max/2, 50)
    W1, W2 = torch.meshgrid(omega_grid, omega_grid, indexing='ij')
    S = silverman_spectral_density(W1.flatten().unsqueeze(-1), W2.flatten().unsqueeze(-1), a=a)
    S = S.reshape(W1.shape)

    im = ax.contourf(W1.numpy(), W2.numpy(), S.numpy(), levels=20, cmap='viridis')
    ax.set_xlabel('ω')
    ax.set_ylabel("ω'")
    ax.set_title("Spectral Density s(ω, ω')")
    plt.colorbar(im, ax=ax)

    # 4. ACVF at center location (x=0)
    ax = axes[1, 1]
    x_center = 0.0
    x_test = x.squeeze()

    # True ACVF r(0, x')
    true_acvf = silverman_acvf(
        torch.tensor([x_center] * len(x_test)),
        x_test,
        a=a
    )

    # Empirical: Compute covariance between samples at x=0 and all other points
    center_idx = torch.argmin(torch.abs(x.squeeze()))
    center_samples = samples[:, center_idx]  # (n_samples,)

    empirical_acvf = torch.zeros(len(x_test))
    for i in range(len(x_test)):
        empirical_acvf[i] = torch.mean((center_samples - center_samples.mean()) *
                                       (samples[:, i] - samples[:, i].mean()))

    ax.plot(x_test.numpy(), true_acvf.numpy(), 'r-', linewidth=2, label='True r(0, x\')')
    ax.plot(x_test.numpy(), empirical_acvf.numpy(), 'b--', linewidth=2, label='Empirical')
    ax.set_xlabel('x\'')
    ax.set_ylabel('r(0, x\')')
    ax.set_title('ACVF at center location x=0')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "silverman_test_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")

    plt.show()

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_nffs_simulation()
