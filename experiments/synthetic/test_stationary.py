"""
Test NFFs with a simple stationary kernel

For stationary: s(ω,ω') = s_0(ω)·δ(ω - ω')
This should be easier to debug because S is diagonal!

We use Gaussian kernel: r(h) = σ²·exp(-h²/(2ℓ²))
Spectral density: s_0(ω) = σ²·ℓ·sqrt(2π)·exp(-ℓ²ω²/2)
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nsgp.models.nffs import NFFs


def gaussian_kernel_acvf(h, sigma=1.0, ell=1.0):
    """Gaussian (squared exponential) ACVF."""
    return sigma**2 * torch.exp(-h**2 / (2 * ell**2))


def gaussian_kernel_spectral_density(omega1, omega2, sigma=1.0, ell=1.0):
    """
    Spectral density for Gaussian kernel (STATIONARY).

    For stationary, s(ω,ω') = s_0(ω)·δ(ω - ω'), where:
    s_0(ω) = σ²·ℓ·sqrt(2π)·exp(-ℓ²ω²/2)

    We approximate δ(ω - ω') with a narrow Gaussian to keep S positive definite.
    """
    if omega1.dim() == 1:
        omega1 = omega1.unsqueeze(-1)
    if omega2.dim() == 1:
        omega2 = omega2.unsqueeze(-1)

    # Spectral density at (ω+ω')/2 (average frequency)
    omega_avg = (omega1 + omega2) / 2.0
    s_omega = sigma**2 * ell * np.sqrt(2 * np.pi) * torch.exp(-ell**2 * omega_avg**2 / 2.0)

    # Delta function approximation: narrow Gaussian
    omega_diff = omega1 - omega2
    delta_approx = torch.exp(-omega_diff**2 / 0.01)  # Very narrow

    return (s_omega * delta_approx).squeeze()


def test_stationary():
    print("=" * 70)
    print("Testing NFFs with Stationary Gaussian Kernel")
    print("=" * 70)

    # Parameters
    sigma = 1.0  # Marginal variance
    ell = 1.0    # Length scale
    n_features = 100
    omega_max = 10.0
    n_samples = 1000

    x = torch.linspace(-5, 5, 100).unsqueeze(-1)

    print(f"\nKernel parameters:")
    print(f"  σ² (variance) = {sigma**2}")
    print(f"  ℓ (length scale) = {ell}")
    print(f"  M (features) = {n_features}")
    print(f"  ω_max = {omega_max}")

    # True ACVF
    h = x - x[50]  # Distances from center point
    true_acvf = gaussian_kernel_acvf(h.squeeze(), sigma, ell)

    print(f"\nTrue statistics:")
    print(f"  r(0) = {gaussian_kernel_acvf(torch.tensor(0.0), sigma, ell).item():.6f}")
    print(f"  r(1) = {gaussian_kernel_acvf(torch.tensor(1.0), sigma, ell).item():.6f}")
    print(f"  r(2) = {gaussian_kernel_acvf(torch.tensor(2.0), sigma, ell).item():.6f}")

    # Initialize NFFs
    def spectral_density_fn(w1, w2):
        return gaussian_kernel_spectral_density(w1, w2, sigma, ell)

    nffs = NFFs(
        spectral_density=spectral_density_fn,
        n_features=n_features,
        omega_max=omega_max,
        input_dim=1
    )

    # Check spectral matrix (should be approximately diagonal!)
    S = nffs._compute_spectral_matrix()
    diag_sum = torch.diag(S).sum().item()
    total_sum = S.sum().item()
    off_diag_ratio = (total_sum - diag_sum) / total_sum

    print(f"\nSpectral matrix S:")
    print(f"  Trace(S) = {torch.trace(S).item():.6f}")
    print(f"  sum(S) = {total_sum:.6f}")
    print(f"  Off-diagonal ratio = {off_diag_ratio:.4f} (should be ~0 for stationary)")

    # Generate samples
    print(f"\nGenerating {n_samples} samples...")
    samples = nffs.simulate(x, n_samples=n_samples, seed=42)

    # Empirical statistics
    empirical_mean = samples.mean(dim=0)
    empirical_var = samples.var(dim=0)

    center_idx = 50
    print(f"\nEmpirical statistics at center (x=0):")
    print(f"  Mean = {empirical_mean[center_idx].item():.6f} (should be ~0)")
    print(f"  Variance = {empirical_var[center_idx].item():.6f} (should be ~{sigma**2})")

    # Compare ACVFs
    center_samples = samples[:, center_idx]
    empirical_acvf = torch.zeros(len(x))
    for i in range(len(x)):
        empirical_acvf[i] = torch.mean(
            (center_samples - center_samples.mean()) *
            (samples[:, i] - samples[:, i].mean())
        )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Sample paths
    ax = axes[0]
    for i in range(min(5, n_samples)):
        ax.plot(x.squeeze().numpy(), samples[i].numpy(), alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('Z(x)')
    ax.set_title('Sample Paths')
    ax.grid(True, alpha=0.3)

    # 2. Variance
    ax = axes[1]
    ax.axhline(sigma**2, color='r', linewidth=2, label=f'True σ² = {sigma**2}')
    ax.plot(x.squeeze().numpy(), empirical_var.numpy(), 'b--', linewidth=2, label='Empirical')
    ax.set_xlabel('x')
    ax.set_ylabel('Variance')
    ax.set_title('Variance (should be constant)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. ACVF
    ax = axes[2]
    ax.plot(h.squeeze().numpy(), true_acvf.numpy(), 'r-', linewidth=2, label='True r(h)')
    ax.plot(h.squeeze().numpy(), empirical_acvf.numpy(), 'b--', linewidth=2, label='Empirical')
    ax.set_xlabel('h (lag)')
    ax.set_ylabel('r(h)')
    ax.set_title('Autocovariance Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / "stationary_test_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_path}")
    plt.show()

    # Summary
    relative_var_error = (empirical_var[center_idx].item() - sigma**2) / sigma**2 * 100
    print(f"\n" + "=" * 70)
    print(f"Variance error: {relative_var_error:.1f}%")
    if abs(relative_var_error) < 10:
        print("✓ TEST PASSED: Variance is within 10% of true value")
    else:
        print("✗ TEST FAILED: Variance error too large")
    print("=" * 70)


if __name__ == "__main__":
    test_stationary()
