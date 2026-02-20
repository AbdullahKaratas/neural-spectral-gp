"""
Compare Spatial Kernels (K matrices)

This script focuses on comparing the learned covariance matrix K(x, x') directly against the ground truth.
It avoids simulation/sampling and focuses on the accuracy of the kernel approximation.

Metrics:
- Relative Frobenius Error: ||K_pred - K_true||_F / ||K_true||_F
- Visual Heatmaps: K_pred vs K_true

Authors: Abdullah Karatas, Arsalan Jawaid
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork
from nsgp.models.standard_gp import StandardGP

from nsgp.kernel import LocalStationaryKernel

# ============================================================================
# COMPARISON LOGIC
# ============================================================================

def compare_kernels(
    kernel_fn,
    kernel_name: str,
    n_train: int = 50,
    n_test: int = 100,
    epochs: int = 1000,
    noise_var: float = 0.05,
    seed: int = 42
):
    print(f"\n{'='*80}")
    print(f"Comparing Kernels: {kernel_name}")
    print(f"{'='*80}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Data
    X_train = torch.linspace(-5, 5, n_train).unsqueeze(-1)
    X_test = torch.linspace(-8, 8, n_test).unsqueeze(-1)
    
    # Ground Truth
    K_true_train = kernel_fn(X_train, X_train)
    K_true_test = kernel_fn(X_test, X_test)
    
    # Generate noisy y
    L = torch.linalg.cholesky(K_true_train + 1e-6 * torch.eye(n_train))
    y_train = (L @ torch.randn(n_train)).squeeze() + math.sqrt(noise_var) * torch.randn(n_train)

    # Normalize inputs
    x_scale = torch.abs(X_train).max()
    X_train_normalized = X_train / x_scale
    X_test_normalized = X_test / x_scale

    # Standardize outputs
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_std = (y_train - y_mean) / y_std

    results = {}

    # 1. Standard GP Baseline
    print("\n[Standard GP] Training...")
    gp = StandardGP()
    gp.fit(X_train_normalized, y_train_std, epochs=epochs, lr=0.01, verbose=True)
    K_gp = gp.compute_covariance(X_test_normalized, X_test_normalized)
    results['Standard GP'] = K_gp * y_std**2
    
    # 2. F-SDN (Ours)
    print("[F-SDN] Training...")
    sdn = FactorizedSpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[64, 64],
        rank=4,
        n_features=80,
        omega_max=20.0,
        enforce_symmetry=True,
        learn_log_scale=True
    )
    sdn.fit(X_train_normalized, y_train_std, epochs=epochs, lr=0.01, verbose=True, patience=epochs)
    # Compute covariance in standardized space and un-standardize
    K_sdn_std = sdn.compute_covariance(X_test_normalized)
    K_sdn = K_sdn_std * y_std**2
    results['F-SDN'] = K_sdn

    # Posterior predictive predictions
    mean_sdn_std, std_sdn_std = sdn.predict(X_test_normalized, X_train_normalized, y_train_std, predictive_dist=True)
    mean_sdn = mean_sdn_std * y_std + y_mean
    std_sdn = std_sdn_std * y_std

    # Standard GP predictions
    mean_gp_std, std_gp_std = gp.predict(X_test_normalized, predictive_dist=True)
    mean_gp = mean_gp_std * y_std + y_mean
    std_gp = std_gp_std * y_std

    # True posterior
    K_true_test_train = kernel_fn(X_test, X_train)
    Sigma_true = K_true_train + noise_var * torch.eye(n_train)
    L_true = torch.linalg.cholesky(Sigma_true)

    # True posterior mean
    alpha_true = torch.cholesky_solve(y_train.unsqueeze(-1), L_true)
    mean_true = (K_true_test_train @ alpha_true).squeeze()

    # True (predictive) posterior variance
    v_true = torch.cholesky_solve(K_true_test_train.T, L_true)
    var_true = torch.diag(K_true_test - K_true_test_train @ v_true) + noise_var
    std_true = torch.sqrt(torch.clamp(var_true, min=1e-6))

    # Plot 1: Posterior predictions with uncertainty
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    X_plot = X_test.squeeze().detach().numpy()
    mean_true_plot = mean_true.detach().numpy()
    std_true_plot = std_true.detach().numpy()
    mean_sdn_plot = mean_sdn.detach().numpy()
    std_sdn_plot = std_sdn.detach().numpy()
    mean_gp_plot = mean_gp.detach().numpy()
    std_gp_plot = std_gp.detach().numpy()

    X_train_plot = X_train.squeeze().detach().numpy()
    y_train_plot = y_train.detach().numpy()

    # Panel 1: True Posterior
    axes1[0].plot(X_plot, mean_true_plot, 'k-', linewidth=2, label='True posterior mean')
    axes1[0].fill_between(
        X_plot,
        mean_true_plot - 2 * std_true_plot,
        mean_true_plot + 2 * std_true_plot,
        alpha=0.3,
        color='gray',
        label=r'True posterior $\pm 2\sigma$'
    )
    axes1[0].scatter(X_train_plot, y_train_plot, c='red', s=20, alpha=0.6, label='Training data', zorder=5)
    axes1[0].set_xlabel('x', fontsize=12)
    axes1[0].set_ylabel('y', fontsize=12)
    axes1[0].set_title(f'True Posterior ({kernel_name})', fontsize=13)
    axes1[0].legend(loc='best', fontsize=10)
    axes1[0].grid(True, alpha=0.3)

    # Panel 2: F-SDN
    axes1[1].plot(X_plot, mean_true_plot, 'gray', linewidth=1.5, linestyle=':', label='True posterior', alpha=0.6)
    axes1[1].plot(X_plot, mean_sdn_plot, 'b-', linewidth=2, label='F-SDN mean')
    axes1[1].fill_between(
        X_plot,
        mean_sdn_plot - 2 * std_sdn_plot,
        mean_sdn_plot + 2 * std_sdn_plot,
        alpha=0.3,
        color='blue',
        label=r'F-SDN $\pm 2\sigma$'
    )
    axes1[1].scatter(X_train_plot, y_train_plot, c='red', s=20, alpha=0.6, label='Training data', zorder=5)
    axes1[1].set_xlabel('x', fontsize=12)
    axes1[1].set_ylabel('y', fontsize=12)
    axes1[1].set_title(f'F-SDN', fontsize=13)
    axes1[1].legend(loc='best', fontsize=10)
    axes1[1].grid(True, alpha=0.3)

    # Panel 3: Standard GP
    axes1[2].plot(X_plot, mean_true_plot, 'gray', linewidth=1.5, linestyle=':', label='True posterior', alpha=0.6)
    axes1[2].plot(X_plot, mean_gp_plot, 'g-', linewidth=2, label='RBF GP mean')
    axes1[2].fill_between(
        X_plot,
        mean_gp_plot - 2 * std_gp_plot,
        mean_gp_plot + 2 * std_gp_plot,
        alpha=0.3,
        color='green',
        label=r'RBF GP $\pm 2\sigma$'
    )
    axes1[2].scatter(X_train_plot, y_train_plot, c='red', s=20, alpha=0.6, label='Training data', zorder=5)
    axes1[2].set_xlabel('x', fontsize=12)
    axes1[2].set_ylabel('y', fontsize=12)
    axes1[2].set_title(f'Baseline RBF', fontsize=13)
    axes1[2].legend(loc='best', fontsize=10)
    axes1[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Plot 2: Kernel heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot Ground Truth
    im0 = axes[0, 0].imshow(K_true_test.numpy(), cmap='viridis')
    axes[0, 0].set_title(f"True Kernel\n{kernel_name}")
    plt.colorbar(im0, ax=axes[0, 0])
    axes[1, 0].axis('off') # No error plot for ground truth

    methods = ['Standard GP', 'F-SDN']

    print("\nResults (Relative Frobenius Error):")

    for i, method in enumerate(methods):
        K_pred = results[method].detach()

        # Error Metric
        error_norm = torch.norm(K_pred - K_true_test)
        true_norm = torch.norm(K_true_test)
        rel_error = error_norm / true_norm

        print(f"  {method:<15}: {rel_error.item():.2%}")
        
        # Plot Kernel
        im = axes[0, i+1].imshow(K_pred.numpy(), cmap='viridis')
        axes[0, i+1].set_title(f"{method}\nError: {rel_error.item():.1%}")
        plt.colorbar(im, ax=axes[0, i+1])
        
        # Plot Error Difference
        diff = torch.abs(K_pred - K_true_test)
        im_diff = axes[1, i+1].imshow(diff.numpy(), cmap='hot')
        axes[1, i+1].set_title(f"|{method} - True|")
        plt.colorbar(im_diff, ax=axes[1, i+1])

    plt.tight_layout()

    # Show both figures
    plt.show()

    return results

if __name__ == "__main__":
    # Test on Non-Stationary Kernel (Silverman)
    # F-SDN should beat Standard GP
    lsk = LocalStationaryKernel(a=0.5)
    compare_kernels(
        lsk.kernel,
        kernel_name="Silverman Non-Stationary",
    )
