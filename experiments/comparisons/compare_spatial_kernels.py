"""
Compare Spatial Kernels (K matrices)

This script focuses on comparing the learned covariance matrix K(x, x') directly against the ground truth.
It avoids simulation/sampling and focuses on the accuracy of the kernel approximation.

Metrics:
- Relative Frobenius Error: ||K_pred - K_true||_F / ||K_true||_F
- Visual Heatmaps: K_pred vs K_true

Authors: Abdullah Karatas, Arsalan Jawaid
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork
from nsgp.models.remes_baseline import RemesNeuralSpectralKernel
from nsgp.models.standard_gp import StandardGP

# ============================================================================
# KERNEL DEFINITIONS (Ground Truth)
# ============================================================================

def silverman_kernel(X1, X2, a=0.5):
    """Silverman locally stationary kernel."""
    if X1.dim() == 2: X1 = X1.squeeze(-1)
    if X2.dim() == 2: X2 = X2.squeeze(-1)
    x_mean = (X1.unsqueeze(1) + X2.unsqueeze(0)) / 2.0
    x_diff = X1.unsqueeze(1) - X2.unsqueeze(0)
    return torch.exp(-2.0 * a * x_mean**2) * torch.exp(-a / 2.0 * x_diff**2)

def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """Standard RBF Kernel."""
    if X1.dim() == 2: X1 = X1.squeeze(-1)
    if X2.dim() == 2: X2 = X2.squeeze(-1)
    dist_sq = (X1.unsqueeze(1) - X2.unsqueeze(0))**2
    return variance * torch.exp(-dist_sq / (2 * lengthscale**2))

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
        omega_max=10.0
    )
    sdn.fit(X_train, y_train, epochs=epochs, lr=1e-3, verbose=True, use_mc_training=False,
            use_diversity=True, lambda_diversity=0.5) # Diversity regularization to prevent rank collapse
    print(f"F-SDN Final Log Scale: {sdn.log_scale.item()}")
    K_sdn = sdn.compute_covariance_deterministic(X_test, noise_var=0.0)
    results['F-SDN'] = K_sdn
    
    # 3. Remes (Baseline) - Official Code
    print("[Remes] Training (Official Code)...")
    try:
        import subprocess
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_x_path = Path(tmpdir) / "train_x.npy"
            train_y_path = Path(tmpdir) / "train_y.npy"
            test_x_path = Path(tmpdir) / "test_x.npy"
            output_path = Path(tmpdir) / "output.npy"
            
            np.save(train_x_path, X_train.numpy())
            np.save(train_y_path, y_train.numpy())
            np.save(test_x_path, X_test.numpy())
            
            wrapper_path = Path(__file__).parent.parent / "baselines" / "run_remes.py"
            
            cmd = [
                sys.executable, str(wrapper_path),
                "--train_x", str(train_x_path),
                "--train_y", str(train_y_path),
                "--test_x", str(test_x_path),
                "--output", str(output_path),
                "--epochs", str(epochs)
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Verify output file was created
            if not output_path.exists():
                raise FileNotFoundError(f"Remes baseline did not create output file: {output_path}")

            K_remes = torch.tensor(np.load(output_path))
            results['Remes'] = K_remes

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run Remes baseline (exit code {e.returncode}): {e}")
        print(f"Command: {' '.join(cmd)}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
    except FileNotFoundError as e:
        print(f"❌ Remes baseline file not found: {e}")
        # Fallback to internal implementation or zeros
        print("Falling back to internal implementation...")
        remes = RemesNeuralSpectralKernel(input_dim=1, hidden_dims=[32, 32])
        remes.fit(X_train, y_train, epochs=epochs, verbose=False)
        K_remes, _ = remes.compute_covariance(X_test, noise_var=0.0)
        results['Remes'] = K_remes
    
    # ============================================================================
    # VISUALIZATION & METRICS
    # ============================================================================
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Plot Ground Truth
    im0 = axes[0, 0].imshow(K_true_test.numpy(), cmap='viridis')
    axes[0, 0].set_title(f"True Kernel\n{kernel_name}")
    plt.colorbar(im0, ax=axes[0, 0])
    axes[1, 0].axis('off') # No error plot for ground truth
    
    methods = ['Standard GP', 'F-SDN', 'Remes']
    
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
    output_path = Path(__file__).parent / f"comparison_{kernel_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_path)
    print(f"\nSaved plot to {output_path}")
    
    return results

if __name__ == "__main__":
    # 1. Test on Stationary Kernel (RBF)
    # Standard GP should be perfect here (0% error)
    compare_kernels(
        lambda x1, x2: rbf_kernel(x1, x2, lengthscale=1.0),
        kernel_name="Stationary RBF",
        is_stationary=True,
        epochs=200
    )
    
    # 2. Test on Non-Stationary Kernel (Silverman)
    # F-SDN should beat Standard GP
    compare_kernels(
        silverman_kernel,
        kernel_name="Silverman Non-Stationary",
        is_stationary=False,
        epochs=500
    )
