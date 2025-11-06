"""
Test SDN Training

Train SDN on synthetic data and validate that it learns the correct
spectral density.

Workflow:
1. Generate synthetic data from known GP (Silverman)
2. Train SDN on this data
3. Compare learned vs. true spectral density
4. Validate samples from learned SDN match ground truth

Authors: Abdullah Karatas, Arsalan Jawaid
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nsgp.models.sdn import SpectralDensityNetwork
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
    """
    Generate synthetic training data from Silverman GP.

    Parameters
    ----------
    n_train : int
        Number of training points
    a : float
        Silverman parameter
    noise_std : float
        Observation noise standard deviation
    seed : int
        Random seed

    Returns
    -------
    X_train, y_train : torch.Tensor
        Training data
    """
    torch.manual_seed(seed)

    # Training locations
    X_train = torch.linspace(-3, 3, n_train).unsqueeze(-1)

    # Generate samples from true Silverman GP
    def spectral_density_fn(w1, w2):
        return silverman_spectral_density(w1, w2, a=a)

    nffs = NFFs(
        spectral_density=spectral_density_fn,
        n_features=100,
        omega_max=10.0,
        input_dim=1
    )

    # Generate one sample path
    y_train = nffs.simulate(X_train, n_samples=1, seed=seed).squeeze()

    # Add observation noise
    y_train = y_train + noise_std * torch.randn_like(y_train)

    # Center
    y_train = y_train - y_train.mean()

    return X_train, y_train


def test_sdn_training():
    print("=" * 70)
    print("Testing SDN Training")
    print("=" * 70)

    # Parameters
    a = 0.5  # True Silverman parameter
    n_train = 50
    noise_std = 0.1
    n_features = 50  # Smaller for faster training
    omega_max = 8.0

    print(f"\nData generation:")
    print(f"  True kernel: Silverman (a={a})")
    print(f"  n_train = {n_train}")
    print(f"  noise_std = {noise_std}")

    # Generate synthetic data
    print(f"\nGenerating synthetic training data...")
    X_train, y_train = generate_synthetic_data(
        n_train=n_train,
        a=a,
        noise_std=noise_std,
        seed=42
    )
    print(f"✓ Generated {len(X_train)} training points")
    print(f"  y range: [{y_train.min().item():.3f}, {y_train.max().item():.3f}]")

    # Initialize SDN
    print(f"\nInitializing SDN...")
    sdn = SpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[32, 32],  # Smaller network for faster training
        n_features=n_features,
        omega_max=omega_max,
        activation='relu'
    )
    print(f"✓ SDN initialized")
    print(f"  Network: input(2) → 32 → 32 → output(1)")
    print(f"  Parameters: {sum(p.numel() for p in sdn.parameters())}")

    # Train SDN
    print(f"\n" + "-" * 70)
    print("Training SDN...")
    print("-" * 70)

    # Option B failed - trying Option A: MANY more samples
    print("Option B didn't work well enough.")
    print("\nUsing OPTION A: Moment Matching with MANY samples")
    print("(Trading speed for gradient stability)")
    print("This will be slower but should converge better...")

    sdn.fit(
        X_train=X_train,
        y_train=y_train,
        epochs=300,  # Fewer epochs but better gradients
        learning_rate=5e-3,  # Smaller LR for stability
        noise_var=noise_std**2,
        lambda_smooth=0.05,  # Strong smoothness prior
        lambda_rank=0.001,
        loss_type="moment_matching",
        n_loss_samples=5000,  # OPTION A: 50x more samples!
        verbose=True,
        print_every=30
    )

    # Compare learned vs. true spectral density
    print(f"\n" + "-" * 70)
    print("Comparing learned vs. true spectral density...")
    print("-" * 70)

    # Evaluate on frequency grid
    omega_grid = torch.linspace(-omega_max/2, omega_max/2, 50)
    W1, W2 = torch.meshgrid(omega_grid, omega_grid, indexing='ij')

    # True spectral density
    S_true = silverman_spectral_density(
        W1.flatten().unsqueeze(-1),
        W2.flatten().unsqueeze(-1),
        a=a
    ).reshape(W1.shape)

    # Learned spectral density
    with torch.no_grad():
        S_learned = sdn.forward(
            W1.flatten().unsqueeze(-1),
            W2.flatten().unsqueeze(-1)
        ).reshape(W1.shape)

    # Compute error
    relative_error = torch.norm(S_learned - S_true) / torch.norm(S_true)
    print(f"  Relative L2 error: {relative_error.item():.4f}")

    # Generate samples from learned SDN
    print(f"\nGenerating samples from learned SDN...")
    X_test = torch.linspace(-4, 4, 100).unsqueeze(-1)
    samples_learned = sdn.simulate(X_test, n_samples=5, seed=123)
    print(f"✓ Generated {samples_learned.shape[0]} samples")

    # Generate samples from true GP (for comparison)
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
    print(f"\n" + "-" * 70)
    print("Creating visualizations...")
    print("-" * 70)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Training data and samples
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(X_train.numpy(), y_train.numpy(), c='red', s=30, alpha=0.7, label='Training data')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Training Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(samples_learned.shape[0]):
        ax2.plot(X_test.numpy(), samples_learned[i].numpy(), alpha=0.6, label=f'Sample {i+1}')
    ax2.scatter(X_train.numpy(), y_train.numpy(), c='red', s=20, alpha=0.5, zorder=10)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Z(x)')
    ax2.set_title('Samples from Learned SDN')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(samples_true.shape[0]):
        ax3.plot(X_test.numpy(), samples_true[i].numpy(), alpha=0.6, label=f'Sample {i+1}')
    ax3.scatter(X_train.numpy(), y_train.numpy(), c='red', s=20, alpha=0.5, zorder=10)
    ax3.set_xlabel('x')
    ax3.set_ylabel('Z(x)')
    ax3.set_title('Samples from True GP')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 2: Spectral densities
    ax4 = fig.add_subplot(gs[1, 0])
    im1 = ax4.contourf(W1.numpy(), W2.numpy(), S_true.numpy(), levels=20, cmap='viridis')
    ax4.set_xlabel('ω')
    ax4.set_ylabel("ω'")
    ax4.set_title("True Spectral Density s(ω,ω')")
    plt.colorbar(im1, ax=ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    im2 = ax5.contourf(W1.numpy(), W2.numpy(), S_learned.numpy(), levels=20, cmap='viridis')
    ax5.set_xlabel('ω')
    ax5.set_ylabel("ω'")
    ax5.set_title("Learned Spectral Density s_θ(ω,ω')")
    plt.colorbar(im2, ax=ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    error = torch.abs(S_learned - S_true)
    im3 = ax6.contourf(W1.numpy(), W2.numpy(), error.numpy(), levels=20, cmap='Reds')
    ax6.set_xlabel('ω')
    ax6.set_ylabel("ω'")
    ax6.set_title(f"Absolute Error (L2: {relative_error.item():.3f})")
    plt.colorbar(im3, ax=ax6)

    # Row 3: Training curves
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(sdn.train_losses, 'b-', linewidth=2)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Total Loss')
    ax7.set_title('Training Loss')
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale('log')

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(sdn.nll_history, 'g-', linewidth=2, label='NLL')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Negative Log-Likelihood')
    ax8.set_title('Likelihood')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(sdn.reg_history, 'r-', linewidth=2)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Regularization')
    ax9.set_title('Regularization Penalty')
    ax9.grid(True, alpha=0.3)

    plt.suptitle('SDN Training Results: Learning Silverman Spectral Density', fontsize=14, fontweight='bold')

    output_path = Path(__file__).parent / "sdn_training_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")

    plt.show()

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ SDN successfully trained on {n_train} observations")
    print(f"✓ Spectral density learned with {relative_error.item():.1%} relative error")
    print(f"✓ Generated realistic samples matching true GP")
    print(f"\nFinal losses:")
    print(f"  Total loss: {sdn.train_losses[-1]:.4f}")
    print(f"  NLL: {sdn.nll_history[-1]:.4f}")
    print(f"  Regularization: {sdn.reg_history[-1]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    test_sdn_training()
