"""
Test SDN Training - OPTION B: Big Network + Advanced Training

Improvements over basic version:
- Much larger network (128×128×128 vs 32×32)
- Learning rate scheduling
- Better initialization
- Early stopping
- Gradient clipping

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
    """Generate synthetic training data from Silverman GP."""
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


def train_sdn_with_scheduler(
    sdn,
    X_train,
    y_train,
    epochs=500,
    initial_lr=1e-2,
    noise_std=0.1,
    patience=50,
    verbose=True
):
    """
    Train SDN with advanced tricks:
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    """
    # Center observations
    y_train = y_train - y_train.mean()

    # Optimizer with learning rate scheduler
    optimizer = torch.optim.Adam(sdn.parameters(), lr=initial_lr)

    # Cosine annealing scheduler: LR goes down smoothly
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=initial_lr / 100
    )

    # Early stopping
    best_loss = float('inf')
    patience_counter = 0

    # Training history
    losses = []

    print("Training with advanced techniques:")
    print(f"  - Initial LR: {initial_lr}")
    print(f"  - LR scheduling: Cosine annealing")
    print(f"  - Early stopping patience: {patience}")
    print(f"  - Gradient clipping: 1.0")
    print()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Data loss: Moment matching with many samples
        data_loss = sdn.spectral_moment_matching_loss(
            X_train,
            y_train,
            n_samples=5000  # High for low noise
        )

        # Regularization
        smooth_penalty = sdn.spectral_smoothness_penalty()
        rank_penalty = sdn.low_rank_penalty()

        # Total loss
        loss = data_loss + 0.05 * smooth_penalty + 0.001 * rank_penalty

        # Backpropagation
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(sdn.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Track
        losses.append(loss.item())
        current_lr = optimizer.param_groups[0]['lr']

        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Data: {data_loss.item():.4f} | LR: {current_lr:.6f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\n✓ Training completed! Best loss: {best_loss:.4f}")

    return losses


def test_option_b():
    print("=" * 70)
    print("OPTION B: Big Network + Advanced Training")
    print("=" * 70)

    # Parameters
    a = 0.5
    n_train = 50
    noise_std = 0.1
    n_features = 50
    omega_max = 8.0

    print(f"\nData generation:")
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

    # Initialize BIG SDN
    print(f"\nInitializing BIG SDN...")
    sdn = SpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[128, 128, 128],  # MUCH bigger!
        n_features=n_features,
        omega_max=omega_max,
        activation='elu'  # ELU often better than ReLU
    )
    n_params = sum(p.numel() for p in sdn.parameters())
    print(f"✓ SDN initialized")
    print(f"  Architecture: input(2) → 128 → 128 → 128 → output(1)")
    print(f"  Parameters: {n_params:,}")
    print(f"  (Previous version had only 1,185 parameters)")

    # Train
    print(f"\n" + "-" * 70)
    losses = train_sdn_with_scheduler(
        sdn=sdn,
        X_train=X_train,
        y_train=y_train,
        epochs=500,
        initial_lr=1e-2,
        noise_std=noise_std,
        patience=100,
        verbose=True
    )
    print("-" * 70)

    # Evaluate
    print(f"\nComparing learned vs. true spectral density...")
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

    # Generate samples
    print(f"\nGenerating samples...")
    X_test = torch.linspace(-4, 4, 100).unsqueeze(-1)
    samples_learned = sdn.simulate(X_test, n_samples=5, seed=123)

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
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Samples
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(X_train.numpy(), y_train.numpy(), c='red', s=30, alpha=0.7, label='Training data')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Training Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(samples_learned.shape[0]):
        ax2.plot(X_test.numpy(), samples_learned[i].numpy(), alpha=0.6)
    ax2.scatter(X_train.numpy(), y_train.numpy(), c='red', s=20, alpha=0.5, zorder=10)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Z(x)')
    ax2.set_title('Samples from Learned SDN (Big Network)')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(samples_true.shape[0]):
        ax3.plot(X_test.numpy(), samples_true[i].numpy(), alpha=0.6)
    ax3.scatter(X_train.numpy(), y_train.numpy(), c='red', s=20, alpha=0.5, zorder=10)
    ax3.set_xlabel('x')
    ax3.set_ylabel('Z(x)')
    ax3.set_title('Samples from True GP')
    ax3.grid(True, alpha=0.3)

    # Row 2: Spectral densities
    ax4 = fig.add_subplot(gs[1, 0])
    im1 = ax4.contourf(W1.numpy(), W2.numpy(), S_true.numpy(), levels=20, cmap='viridis')
    ax4.set_xlabel('ω')
    ax4.set_ylabel("ω'")
    ax4.set_title("True Spectral Density")
    plt.colorbar(im1, ax=ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    im2 = ax5.contourf(W1.numpy(), W2.numpy(), S_learned.numpy(), levels=20, cmap='viridis')
    ax5.set_xlabel('ω')
    ax5.set_ylabel("ω'")
    ax5.set_title("Learned Spectral Density (Big Net)")
    plt.colorbar(im2, ax=ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    error = torch.abs(S_learned - S_true)
    im3 = ax6.contourf(W1.numpy(), W2.numpy(), error.numpy(), levels=20, cmap='Reds')
    ax6.set_xlabel('ω')
    ax6.set_ylabel("ω'")
    ax6.set_title(f"Absolute Error (L2: {relative_error.item():.3f})")
    plt.colorbar(im3, ax=ax6)

    # Row 3: Training curve
    ax7 = fig.add_subplot(gs[2, :])
    ax7.plot(losses, 'b-', linewidth=2)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Total Loss')
    ax7.set_title('Training Loss (Big Network + Advanced Techniques)')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(losses[-1], color='r', linestyle='--', alpha=0.5, label=f'Final: {losses[-1]:.2f}')
    ax7.legend()

    plt.suptitle('Option B: Big Network (128×128×128) with Advanced Training', fontsize=14, fontweight='bold')

    output_path = Path(__file__).parent / "sdn_option_b_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")

    plt.show()

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Network size: {n_params:,} parameters (vs 1,185 previously)")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Spectral density error: {relative_error.item()*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    test_option_b()
