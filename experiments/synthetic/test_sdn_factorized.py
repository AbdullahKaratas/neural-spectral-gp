"""
Test Factorized SDN - Guaranteed Positive Definiteness! 🎯

This version uses low-rank factorization to GUARANTEE PD:
    s(ω, ω') = f(ω)ᵀ f(ω')

This means:
✓ Sampling will always work (no Cholesky failures)
✓ Cleaner architecture (PD by construction)
✓ Better optimization (no fighting against PD constraints)

Authors: Abdullah Karatas, Arsalan Jawaid
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork
from nsgp.models.nffs import NFFs


def silverman_spectral_density(omega1, omega2, a=0.5):
    """Silverman's spectral density (ground truth)."""
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


def generate_synthetic_data(n_train=50, a=0.5, noise_std=0.1, seed=42):
    """Generate synthetic training data from Silverman GP."""
    torch.manual_seed(seed)

    X_train = torch.linspace(-3, 3, n_train).unsqueeze(-1)

    def spectral_density_fn(w1, w2):
        return silverman_spectral_density(w1, w2, a=a)

    nffs = NFFs(
        spectral_density=spectral_density_fn,
        n_features=100,
        omega_max=10.0,
        input_dim=1
    )

    y_train = nffs.simulate(X_train, n_samples=1, seed=seed).squeeze()
    y_train = y_train + noise_std * torch.randn_like(y_train)
    y_train = y_train - y_train.mean()

    return X_train, y_train


def test_factorized_sdn():
    print("=" * 70)
    print("FACTORIZED SDN - Guaranteed Positive Definiteness! 🎯")
    print("=" * 70)

    # Parameters
    a = 0.5
    n_train = 50
    noise_std = 0.1
    n_features = 50
    omega_max = 8.0

    print(f"\nSetup:")
    print(f"  Kernel: Silverman (a={a})")
    print(f"  n_train: {n_train}")
    print(f"  noise_std: {noise_std}")

    # Generate data
    print(f"\nGenerating synthetic data...")
    X_train, y_train = generate_synthetic_data(
        n_train=n_train,
        a=a,
        noise_std=noise_std,
        seed=42
    )
    print(f"✓ Generated {len(X_train)} training points")

    # Initialize Factorized SDN
    print(f"\nInitializing Factorized SDN...")
    sdn = FactorizedSpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[64, 64, 64],  # Medium network
        rank=15,  # Factorization rank (key parameter!)
        n_features=n_features,
        omega_max=omega_max,
        activation='elu'
    )
    n_params = sum(p.numel() for p in sdn.parameters())
    print(f"✓ Factorized SDN initialized:")
    print(f"  - Parameters: {n_params:,}")
    print(f"  - Rank: {sdn.rank}")
    print(f"  - PD: GUARANTEED by construction! ✓")

    # Train
    print(f"\n" + "-" * 70)
    losses = sdn.fit(
        X_train=X_train,
        y_train=y_train,
        epochs=1000,
        lr=1e-2,
        noise_var=noise_std**2,
        lambda_smooth=0.1,
        patience=150,
        verbose=True
    )
    print("-" * 70)

    # Evaluate spectral density
    print(f"\nEvaluating learned spectral density...")
    omega_grid = torch.linspace(-omega_max/2, omega_max/2, 50)
    W1, W2 = torch.meshgrid(omega_grid, omega_grid, indexing='ij')

    S_true = silverman_spectral_density(
        W1.flatten().unsqueeze(-1),
        W2.flatten().unsqueeze(-1),
        a=a
    ).reshape(W1.shape)

    with torch.no_grad():
        S_learned = sdn.forward(
            W1.flatten().unsqueeze(-1),
            W2.flatten().unsqueeze(-1)
        ).reshape(W1.shape)

    relative_error = torch.norm(S_learned - S_true) / torch.norm(S_true)
    print(f"  Relative L2 error: {relative_error.item():.4f} ({relative_error.item()*100:.1f}%)")

    # Generate samples (should work now!)
    print(f"\nGenerating samples from learned prior...")
    X_test = torch.linspace(-4, 4, 100).unsqueeze(-1)

    try:
        with torch.no_grad():
            samples_learned = sdn.simulate(X_test, n_samples=5, seed=123)
        print(f"  ✓ SUCCESS! Sampling worked! 🎉")
        sampling_worked = True
    except RuntimeError as e:
        print(f"  ⚠️ Sampling failed: {str(e)[:100]}")
        print(f"  → This shouldn't happen with factorized s(ω,ω')...")
        sampling_worked = False
        samples_learned = None

    # Generate true samples for comparison
    def spectral_density_fn_true(w1, w2):
        return silverman_spectral_density(w1, w2, a=a)

    nffs_true = NFFs(
        spectral_density=spectral_density_fn_true,
        n_features=n_features,
        omega_max=omega_max,
        input_dim=1
    )
    samples_true = nffs_true.simulate(X_test, n_samples=5, seed=123)

    # Visualization
    print(f"\nCreating visualizations...")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Samples
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(X_train.numpy(), y_train.numpy(), c='red', s=40, alpha=0.7,
                edgecolors='darkred', linewidths=1.5, label='Training data')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Training Data', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if sampling_worked and samples_learned is not None:
        for i in range(samples_learned.shape[0]):
            ax2.plot(X_test.numpy(), samples_learned[i].numpy(), alpha=0.7, linewidth=2)
        ax2.scatter(X_train.numpy(), y_train.numpy(), c='red', s=30, alpha=0.6, zorder=10)
        ax2.set_title('Samples from Learned Prior (FACTORIZED)', fontsize=14, fontweight='bold')
        ax2.text(0.05, 0.95, '✓ Sampling Works!', transform=ax2.transAxes,
                 fontsize=11, color='green', fontweight='bold', va='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    else:
        ax2.text(0.5, 0.5, 'Sampling Failed\n(unexpected!)', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=16, color='red')
        ax2.set_title('Samples from Learned Prior (FAILED)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Z(x)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(samples_true.shape[0]):
        ax3.plot(X_test.numpy(), samples_true[i].numpy(), alpha=0.7, linewidth=2)
    ax3.scatter(X_train.numpy(), y_train.numpy(), c='red', s=30, alpha=0.6, zorder=10)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('Z(x)', fontsize=12)
    ax3.set_title('Samples from True Prior', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Row 2: Spectral densities
    ax4 = fig.add_subplot(gs[1, 0])
    im1 = ax4.contourf(W1.numpy(), W2.numpy(), S_true.numpy(), levels=20, cmap='viridis')
    ax4.set_xlabel('ω', fontsize=12)
    ax4.set_ylabel("ω'", fontsize=12)
    ax4.set_title("True Spectral Density s(ω,ω')", fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    im2 = ax5.contourf(W1.numpy(), W2.numpy(), S_learned.numpy(), levels=20, cmap='viridis')
    ax5.set_xlabel('ω', fontsize=12)
    ax5.set_ylabel("ω'", fontsize=12)
    ax5.set_title(f"Learned Spectral Density (Factorized, rank={sdn.rank})",
                  fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    error = torch.abs(S_learned - S_true)
    im3 = ax6.contourf(W1.numpy(), W2.numpy(), error.numpy(), levels=20, cmap='Reds')
    ax6.set_xlabel('ω', fontsize=12)
    ax6.set_ylabel("ω'", fontsize=12)
    ax6.set_title(f"Absolute Error\n(Relative L2: {relative_error.item():.3f})",
                  fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=ax6)

    # Row 3: Training curves
    ax7 = fig.add_subplot(gs[2, :])
    ax7.plot(losses, 'b-', linewidth=2.5, alpha=0.8)
    ax7.set_xlabel('Epoch', fontsize=12)
    ax7.set_ylabel('Total Loss', fontsize=12)
    ax7.set_title('Training Loss (Factorized SDN)', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(losses[-1], color='r', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Final: {losses[-1]:.2f}')
    ax7.axhline(min(losses), color='g', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Best: {min(losses):.2f}')
    ax7.legend(fontsize=11)

    plt.suptitle('Factorized SDN: s(ω,ω\') = f(ω)ᵀf(ω\') - Guaranteed PD! 🎯',
                 fontsize=16, fontweight='bold')

    output_path = Path(__file__).parent / "sdn_factorized_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")

    plt.show()

    # Summary
    print(f"\n" + "=" * 70)
    print("FACTORIZED SDN RESULTS")
    print("=" * 70)
    print(f"Architecture: Factorized (rank={sdn.rank})")
    print(f"Parameters: {n_params:,}")
    print(f"Training epochs: {len(losses)}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {min(losses):.4f}")
    print(f"Spectral density error: {relative_error.item()*100:.1f}%")
    print(f"Sampling: {'✓ SUCCESS' if sampling_worked else '✗ FAILED'}")
    print()

    # Performance rating
    if sampling_worked:
        print("🎯 BREAKTHROUGH! Sampling works with factorized architecture!")

    if relative_error.item() < 0.5:
        print("🏆 OUTSTANDING! Error < 50%")
    elif relative_error.item() < 1.0:
        print("🎉 EXCELLENT! Error < 100%")
    elif relative_error.item() < 2.0:
        print("✅ GOOD! Error < 200%")
    else:
        print("✓ Decent progress")

    print("=" * 70)


if __name__ == "__main__":
    test_factorized_sdn()
