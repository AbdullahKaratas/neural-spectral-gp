"""
Compare Spatial Kernels (K matrices)

This script focuses on comparing the learned covariance matrix K(x, x') directly against the ground truth.
It avoids simulation/sampling and focuses on the accuracy of the kernel approximation.

Metrics:
- Relative Frobenius Error: ||K_pred - K_true||_F / ||K_true||_F
- Visual Heatmaps: K_pred vs K_true

Authors: Abdullah Karatas, Arsalan Jawaid
"""

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
    is_stationary: bool = False,
    n_train: int = 100,
    n_test: int = 50,
    epochs: int = 500,
    seed: int = 42
):
    print(f"\n{'='*80}")
    print(f"Comparing Kernels: {kernel_name}")
    print(f"{'='*80}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Data
    X_train = torch.linspace(-3, 3, n_train).unsqueeze(-1)
    X_test = torch.linspace(-3, 3, n_test).unsqueeze(-1)
    
    # Ground Truth
    K_true_train = kernel_fn(X_train, X_train)
    K_true_test = kernel_fn(X_test, X_test)
    
    # Generate y
    L = torch.linalg.cholesky(K_true_train + 1e-4 * torch.eye(n_train))
    y_train = (L @ torch.randn(n_train)).squeeze()

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
    gp = StandardGP(kernel_type='rbf' if is_stationary else 'rbf') # Use RBF as baseline for everything
    gp.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=True)
    K_gp = gp.forward(X_test, X_test)
    results['Standard GP'] = K_gp
    
    # 2. F-SDN (Ours)
    print("[F-SDN] Training...")
    sdn = FactorizedSpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[64, 64],
        rank=10,
        n_features=40,
        omega_max=10.0,
        enforce_symmetry=True
    )
    sdn.fit(X_train_normalized, y_train_std, epochs=epochs, lr=0.01, verbose=True)
    print(f"F-SDN Final Log Scale: {sdn.log_scale.item()}")
    # Compute covariance in standardized space and un-standardize
    K_sdn_std = sdn.compute_covariance(X_test_normalized)
    K_sdn = K_sdn_std * y_std**2
    results['F-SDN'] = K_sdn

    # ============================================================================
    # VISUALIZATION & METRICS
    # ============================================================================

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
        
        print(f"[{method}] Stats:")
        print(f"  True: min={K_true_test.min():.4f}, max={K_true_test.max():.4f}, mean={K_true_test.mean():.4f}")
        print(f"  Pred: min={K_pred.min():.4f}, max={K_pred.max():.4f}, mean={K_pred.mean():.4f}")
        
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
    plt.show()
    
    return results

if __name__ == "__main__":
    # Test on Non-Stationary Kernel (Silverman)
    # F-SDN should beat Standard GP
    lsk = LocalStationaryKernel(a=0.5)
    compare_kernels(
        lsk.kernel,
        kernel_name="Silverman Non-Stationary",
        is_stationary=False,
        epochs=500
    )
