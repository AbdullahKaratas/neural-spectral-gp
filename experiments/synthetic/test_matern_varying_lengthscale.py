"""
Matérn Kernel with Spatially-Varying Lengthscale

Test case: ℓ(x) = 0.5 + 0.3·sin(x)
This creates a nonstationary kernel where the lengthscale varies smoothly with x.

Based on Paciorek & Schervish (2004) framework for nonstationary kernels.
Since no closed-form spectral density exists, we evaluate via K-matching.

Authors: Abdullah Karatas, Arsalan Jawaid
Date: November 2025
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork


def lengthscale_fn(x):
    """Spatially-varying lengthscale: ℓ(x) = 0.5 + 0.3·sin(x)"""
    return 0.5 + 0.3 * torch.sin(x)


def matern_kernel_nonstationary(x, xprime, lengthscale_fn, nu=1.5, sigma_f=1.0):
    """
    Nonstationary Matérn kernel with varying lengthscale.

    Based on Paciorek & Schervish (2004):
    k(x, x') = σ_f² · |Σ(x)|^(1/4) · |Σ(x')|^(1/4) · k_ν(r̃)

    For 1D with scalar lengthscales: Σ(x) = ℓ²(x), so:
    |Σ(x)|^(1/4) · |Σ(x')|^(1/4) = √(ℓ(x)·ℓ(x'))

    where:
    - r̃ = ||x - x'|| / √((ℓ(x)² + ℓ(x')²)/2)
    - k_ν(r) = 2^(1-ν)/Γ(ν) · (√(2ν)·r)^ν · K_ν(√(2ν)·r)

    For ν = 1.5 (once differentiable):
    k_1.5(r) = (1 + √3·r) · exp(-√3·r)

    Args:
        x, xprime: Input locations
        lengthscale_fn: Function ℓ(x) returning lengthscale at x
        nu: Smoothness parameter (1.5 = once differentiable)
        sigma_f: Signal variance
    """
    l_x = lengthscale_fn(x)
    l_xp = lengthscale_fn(xprime)

    # Effective lengthscale (arithmetic mean for distance metric)
    l_eff = torch.sqrt((l_x**2 + l_xp**2) / 2)

    # Scaled distance
    dist = torch.abs(x - xprime)
    r_tilde = dist / l_eff

    # Matérn ν=1.5 correlation function
    sqrt3_r = np.sqrt(3) * r_tilde
    matern_corr = (1 + sqrt3_r) * torch.exp(-sqrt3_r)

    # Amplitude adjustment: |Σ(x)|^(1/4) · |Σ(x')|^(1/4) = √(ℓ(x)·ℓ(x'))
    # CORRECTED: Removed extra normalization factor that was causing scale issues
    amplitude_factor = torch.sqrt(l_x * l_xp)

    return sigma_f**2 * amplitude_factor * matern_corr


def generate_covariance_matrix(X, lengthscale_fn, nu=1.5, sigma_f=1.0, noise_var=1e-6):
    """
    Generate covariance matrix for nonstationary Matérn kernel.

    Returns:
        K: Covariance matrix [n x n]
    """
    n = len(X)
    K = torch.zeros(n, n)

    for i in range(n):
        for j in range(n):
            K[i, j] = matern_kernel_nonstationary(
                X[i], X[j], lengthscale_fn, nu=nu, sigma_f=sigma_f
            )

    # Add noise for numerical stability
    K = K + noise_var * torch.eye(n)

    return K


def generate_synthetic_data(n_train=50, nu=1.5, sigma_f=1.0, noise_std=0.1, seed=42):
    """
    Generate synthetic training data from nonstationary Matérn kernel.
    """
    torch.manual_seed(seed)

    X_train = torch.linspace(-3, 3, n_train).unsqueeze(-1)

    # Compute covariance matrix
    K = generate_covariance_matrix(X_train, lengthscale_fn, nu=nu, sigma_f=sigma_f)

    # Cholesky decomposition
    try:
        L = torch.linalg.cholesky(K)
    except RuntimeError:
        # Add more jitter if needed
        K = K + 1e-4 * torch.eye(len(K))
        L = torch.linalg.cholesky(K)

    # Sample from GP
    z = torch.randn(n_train)
    y_train = L @ z

    # Add observation noise
    y_train = y_train + noise_std * torch.randn_like(y_train)
    y_train = y_train - y_train.mean()

    return X_train, y_train


def test_matern_varying_lengthscale():
    print("=" * 70)
    print("Matérn Kernel with Spatially-Varying Lengthscale - F-SDN Test")
    print("=" * 70)

    # Parameters
    nu = 1.5  # Once differentiable
    sigma_f = 1.0
    n_train = 50
    noise_std = 0.1
    n_features = 50
    omega_max = 8.0

    print(f"\nSetup:")
    print(f"  Kernel: Matérn-{nu} with ℓ(x) = 0.5 + 0.3·sin(x)")
    print(f"  Signal variance: {sigma_f}")
    print(f"  n_train: {n_train}")
    print(f"  noise_std: {noise_std}")

    # Generate data
    print(f"\nGenerating synthetic data...")
    X_train, y_train = generate_synthetic_data(
        n_train=n_train,
        nu=nu,
        sigma_f=sigma_f,
        noise_std=noise_std,
        seed=42
    )
    print(f"✓ Generated {len(X_train)} training points")

    # Compute ground truth covariance (for evaluation)
    print(f"\nComputing ground truth covariance...")
    X_test = torch.linspace(-4, 4, 100).unsqueeze(-1)
    K_true = generate_covariance_matrix(X_test, lengthscale_fn, nu=nu, sigma_f=sigma_f)
    print(f"✓ Ground truth K computed: {K_true.shape}")

    # Initialize Factorized SDN
    print(f"\nInitializing Factorized SDN...")
    sdn = FactorizedSpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[64, 64, 64],
        rank=15,
        n_features=n_features,
        omega_max=omega_max,
        activation='elu'
    )
    n_params = sum(p.numel() for p in sdn.parameters())
    print(f"✓ SDN initialized:")
    print(f"  - Parameters: {n_params:,}")
    print(f"  - Rank: {sdn.rank}")
    print(f"  - Architecture: [64, 64, 64]")

    # Train with deterministic covariance (best accuracy)
    print(f"\n" + "-" * 70)
    losses = sdn.fit(
        X_train=X_train,
        y_train=y_train,
        epochs=1000,
        lr=1e-2,
        noise_var=noise_std**2,
        lambda_smooth=0.1,
        patience=150,
        use_mc_training=False,  # Deterministic for best accuracy
        verbose=True
    )
    print("-" * 70)

    # Evaluate: K-error (NO post-hoc correction - should work now!)
    print(f"\nEvaluating learned covariance...")
    with torch.no_grad():
        K_learned = sdn.compute_covariance_deterministic(X_test, noise_var=1e-6)
        K_train_learned = sdn.compute_covariance_deterministic(X_train, noise_var=0.0)

    # Check scale (for diagnostic purposes)
    empirical_var = y_train.var().item()
    learned_var = torch.diag(K_train_learned).mean().item()
    scale_ratio = learned_var / empirical_var if empirical_var > 1e-10 else 1.0

    print(f"  Scale diagnostic:")
    print(f"    - Empirical var(y_train): {empirical_var:.4f}")
    print(f"    - Learned var (mean diag): {learned_var:.4f}")
    print(f"    - Ratio (learned/empirical): {scale_ratio:.4f}")

    k_error = torch.norm(K_learned - K_true) / torch.norm(K_true)
    print(f"  Covariance error (K): {k_error.item():.4f} ({k_error.item()*100:.1f}%)")

    # Generate samples (test if sampling works!)
    print(f"\nGenerating samples from learned prior...")
    try:
        with torch.no_grad():
            samples_learned = sdn.simulate(X_test, n_samples=5, seed=123)
        print(f"  ✓ SUCCESS! Sampling worked! 🎉")
        sampling_worked = True
    except RuntimeError as e:
        print(f"  ⚠️ Sampling failed: {str(e)[:100]}")
        sampling_worked = False
        samples_learned = None

    # Generate true samples for comparison
    print(f"\nGenerating samples from true prior...")
    K_test_train = generate_covariance_matrix(
        torch.cat([X_test, X_train]), lengthscale_fn, nu=nu, sigma_f=sigma_f
    )
    n_test = len(X_test)
    n_train_actual = len(X_train)

    # True posterior samples
    samples_true = []
    for _ in range(5):
        z_all = torch.randn(n_test + n_train_actual)
        try:
            L_all = torch.linalg.cholesky(K_test_train + 1e-5 * torch.eye(len(K_test_train)))
            y_all = L_all @ z_all
            samples_true.append(y_all[:n_test])
        except:
            samples_true.append(torch.zeros(n_test))
    samples_true = torch.stack(samples_true)

    # Visualization
    print(f"\nCreating visualizations...")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

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
        ax2.set_title('Samples from Learned Prior', fontsize=14, fontweight='bold')
        ax2.text(0.05, 0.95, '✓ Sampling Works!', transform=ax2.transAxes,
                 fontsize=11, color='green', fontweight='bold', va='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    else:
        ax2.text(0.5, 0.5, 'Sampling Failed', ha='center', va='center',
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

    # Row 2: Covariances and lengthscale
    ax4 = fig.add_subplot(gs[1, 0])
    im1 = ax4.imshow(K_true.numpy(), cmap='viridis', aspect='auto')
    ax4.set_xlabel('x index', fontsize=12)
    ax4.set_ylabel('x index', fontsize=12)
    ax4.set_title("True Covariance K(x,x')", fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    im2 = ax5.imshow(K_learned.detach().numpy(), cmap='viridis', aspect='auto')
    ax5.set_xlabel('x index', fontsize=12)
    ax5.set_ylabel('x index', fontsize=12)
    ax5.set_title("Learned Covariance K(x,x')", fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    error = torch.abs(K_learned - K_true)
    im3 = ax6.imshow(error.detach().numpy(), cmap='Reds', aspect='auto')
    ax6.set_xlabel('x index', fontsize=12)
    ax6.set_ylabel('x index', fontsize=12)
    ax6.set_title(f"Absolute Error\n(Relative L2: {k_error.item():.3f})",
                  fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=ax6)

    plt.suptitle('Matérn-1.5 Kernel with Varying Lengthscale: ℓ(x) = 0.5 + 0.3·sin(x)',
                 fontsize=16, fontweight='bold')

    output_path = Path(__file__).parent / "matern_varying_lengthscale_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")

    plt.show()

    # Summary
    print(f"\n" + "=" * 70)
    print("MATÉRN VARYING LENGTHSCALE RESULTS")
    print("=" * 70)
    print(f"Network: {n_params:,} parameters (rank={sdn.rank})")
    print(f"Training epochs: {len(losses)}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {min(losses):.4f}")
    print(f"Covariance error (K): {k_error.item()*100:.1f}%")
    print(f"Sampling: {'✓ Works' if sampling_worked else '✗ Failed'}")
    print()

    if k_error.item() < 0.20:
        print("🏆 OUTSTANDING! K-error < 20%")
    elif k_error.item() < 0.30:
        print("🎉 EXCELLENT! K-error < 30%")
    elif k_error.item() < 0.50:
        print("✅ GOOD! K-error < 50%")
    else:
        print("✓ Decent progress, room for improvement")

    print("=" * 70)

    # Return results for potential further analysis
    return {
        'k_error': k_error.item(),
        's_error': None,  # Not available for Matérn (no closed-form)
        'sampling_works': sampling_worked,
        'n_epochs': len(losses),
        'final_loss': losses[-1],
        'best_loss': min(losses)
    }


if __name__ == "__main__":
    results = test_matern_varying_lengthscale()
    print(f"\nResults dict: {results}")
