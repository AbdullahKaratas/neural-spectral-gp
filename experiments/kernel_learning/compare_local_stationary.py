"""
Posterior extrapolation example (preprint)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from nsgp.models import (
    DKLGP,
    FactorizedSpectralDensityNetwork,
    NeuralGSMGP,
    StandardGP
)

from nsgp.kernel import LocalStationaryKernel
from nsgp.metrics import kl_posterior, marginal_log_likelihood, noise_variance, oracle_posterior


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def compare_kernels(
    kernel_fn,
    kernel_name: str,
    n_train: int = 50,
    n_test: int = 100,
    epochs: int = 4000,
    noise_var: float = 1e-4,
    seed: int = 42
):
    print(f"\n{'='*80}")
    print(f"Comparing Kernels: {kernel_name}")
    print(f"{'='*80}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    X_train = torch.linspace(-5, 5, n_train).unsqueeze(-1)
    X_test = torch.linspace(-10, 10, n_test).unsqueeze(-1)

    # Ground Truth
    K_true_train = kernel_fn(X_train, X_train)

    # Generate noisy y
    L = torch.linalg.cholesky(K_true_train + noise_var * torch.eye(n_train))
    y_train = (L @ torch.randn(n_train)).squeeze()

    # Compute ground truth test kernel
    K_true_test = kernel_fn(X_test, X_test)

    oracle_post = oracle_posterior(kernel_fn, X_train, y_train, X_test, noise_var=noise_var)

    results = {}
    metrics = {}

    # 1. Standard GP Baseline
    print("[Standard GP] Training...")
    gp = StandardGP()
    gp.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=True)
    K_gp = gp.compute_covariance(X_test, X_test)
    results['Standard GP'] = K_gp
    pred_dist_gp = gp._full_pred_dist(X_test)
    metrics['Standard GP'] = {
        "kl":        float(kl_posterior(pred_dist_gp, oracle_post).item()),
        "mll":       marginal_log_likelihood(gp),
        "noise_var": noise_variance(gp),
    }

    # 2. F-SDN (Ours)
    print("[F-SDN] Training...")
    sdn = FactorizedSpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[128, 128],
        rank=8,
        n_features=256,
        omega_max=10.0,
        enforce_symmetry=False,
    )
    sdn.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=True)
    K_sdn = sdn.compute_covariance(X_test)
    results['F-SDN'] = K_sdn
    pred_dist_sdn = sdn._full_pred_dist(X_test)
    metrics['F-SDN'] = {
        "kl":        float(kl_posterior(pred_dist_sdn, oracle_post).item()),
        "mll":       marginal_log_likelihood(sdn),
        "noise_var": noise_variance(sdn),
    }

    # 3. Neural-GSM
    print("[Neural-GSM] Training...")
    ngsm = NeuralGSMGP(input_dim=1, n_components=2, hidden_dims=[32, 32], prior_variance=1.0)
    ngsm.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=True)
    K_ngsm = ngsm.compute_covariance(X_test)
    results['Neural-GSM'] = K_ngsm
    pred_dist_ngsm = ngsm._full_pred_dist(X_test)
    metrics['Neural-GSM'] = {
        "kl":        float(kl_posterior(pred_dist_ngsm, oracle_post).item()),
        "mll":       marginal_log_likelihood(ngsm),
        "noise_var": noise_variance(ngsm),
    }

    # 4. DKL
    print("[DKL] Training...")
    dkl = DKLGP(input_dim=1)
    dkl.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=True)
    K_dkl = dkl.compute_covariance(X_test)
    results['DKL'] = K_dkl
    pred_dist_dkl = dkl._full_pred_dist(X_test)
    metrics['DKL'] = {
        "kl":        float(kl_posterior(pred_dist_dkl, oracle_post).item()),
        "mll":       marginal_log_likelihood(dkl),
        "noise_var": noise_variance(dkl),
    }

    # Predictions
    mean_sdn, std_sdn = sdn.predict(X_test, predictive_dist=True)
    mean_gp, std_gp = gp.predict(X_test, predictive_dist=True)
    mean_ngsm, std_ngsm = ngsm.predict(X_test, predictive_dist=True)
    mean_dkl, std_dkl = dkl.predict(X_test, predictive_dist=True)

    # True posterior
    K_true_test_train = kernel_fn(X_test, X_train)
    Sigma_true = K_true_train + noise_var * torch.eye(n_train)
    L_true = torch.linalg.cholesky(Sigma_true)

    # True posterior mean
    alpha_true = torch.cholesky_solve(y_train.unsqueeze(-1), L_true)
    mean_true = (K_true_test_train @ alpha_true).squeeze()

    # True posterior variance
    v_true = torch.cholesky_solve(K_true_test_train.T, L_true)
    var_true = torch.diag(K_true_test - K_true_test_train @ v_true) + noise_var
    std_true = torch.sqrt(torch.clamp(var_true, min=1e-6))

    # Save posterior prediction data to CSV
    X_plot = X_test.squeeze().detach().numpy()
    posterior_data = pd.DataFrame({
        'x': X_plot,
        'true_mean': mean_true.detach().numpy(),
        'true_lower': (mean_true - 2 * std_true).detach().numpy(),
        'true_upper': (mean_true + 2 * std_true).detach().numpy(),
        'fsdn_mean': mean_sdn.detach().numpy(),
        'fsdn_lower': (mean_sdn - 2 * std_sdn).detach().numpy(),
        'fsdn_upper': (mean_sdn + 2 * std_sdn).detach().numpy(),
        'rbf_mean': mean_gp.detach().numpy(),
        'rbf_lower': (mean_gp - 2 * std_gp).detach().numpy(),
        'rbf_upper': (mean_gp + 2 * std_gp).detach().numpy(),
    })
    posterior_data.to_csv(DATA_DIR / 'kernel_learning_posterior.csv', index=False)

    # Save training data
    X_train_plot = X_train.squeeze().detach().numpy()
    y_train_plot = y_train.detach().numpy()
    training_data = pd.DataFrame({
        'x_train': X_train_plot,
        'y_train': y_train_plot,
    })
    training_data.to_csv(DATA_DIR / 'kernel_learning_training_data.csv', index=False)

    # Plot posterior predictions
    fig1, axes1 = plt.subplots(1, 5, figsize=(30, 5))

    X_plot = X_test.squeeze().detach().numpy()
    X_train_plot = X_train.squeeze().detach().numpy()
    y_train_plot = y_train.detach().numpy()

    # True posterior
    axes1[0].plot(X_plot, mean_true.detach().numpy(), 'k-', linewidth=2, label='True Posterior')
    axes1[0].fill_between(
        X_plot,
        (mean_true - 2 * std_true).detach().numpy(),
        (mean_true + 2 * std_true).detach().numpy(),
        alpha=0.3,
        color='gray',
        label=r'$\pm 2\sigma$'
    )
    axes1[0].scatter(X_train_plot, y_train_plot, c='red', s=20, alpha=0.3, label='Training data')
    axes1[0].set_xlabel('x')
    axes1[0].set_ylabel('y')
    axes1[0].set_title(f'True Posterior ({kernel_name})')
    axes1[0].legend(loc='best')
    axes1[0].grid(True)

    # F-SDN
    axes1[1].plot(X_plot, mean_true.detach().numpy(), 'gray', linewidth=2, linestyle='dashed', label='True', alpha=0.3)
    axes1[1].plot(X_plot, mean_sdn.detach().numpy(), 'b-', linewidth=2, label='F-SDN Posterior')
    axes1[1].fill_between(
        X_plot,
        (mean_sdn - 2 * std_sdn).detach().numpy(),
        (mean_sdn + 2 * std_sdn).detach().numpy(),
        alpha=0.3,
        color='blue',
        label=r'$\pm 2\sigma$'
    )
    axes1[1].scatter(X_train_plot, y_train_plot, c='red', s=20, alpha=0.3, label='Training data')
    axes1[1].set_xlabel('x')
    axes1[1].set_ylabel('y')
    axes1[1].set_title(f'F-SDN')
    axes1[1].legend(loc='best')
    axes1[1].grid(True)

    # Standard GP
    axes1[2].plot(X_plot, mean_true.detach().numpy(), 'gray', linewidth=2, linestyle='dashed', label='True', alpha=0.3)
    axes1[2].plot(X_plot, mean_gp.detach().numpy(), 'g-', linewidth=2, label='RBF Posterior')
    axes1[2].fill_between(
        X_plot,
        (mean_gp - 2 * std_gp).detach().numpy(),
        (mean_gp + 2 * std_gp).detach().numpy(),
        alpha=0.3,
        color='green',
        label=r'$\pm 2\sigma$'
    )
    axes1[2].scatter(X_train_plot, y_train_plot, c='red', s=20, alpha=0.3, label='Training data')
    axes1[2].set_xlabel('x')
    axes1[2].set_ylabel('y')
    axes1[2].set_title(f'RBF')
    axes1[2].legend(loc='best')
    axes1[2].grid(True)

    # Neural-GSM
    axes1[3].plot(X_plot, mean_true.detach().numpy(), 'gray', linewidth=2, linestyle='dashed', label='True', alpha=0.3)
    axes1[3].plot(X_plot, mean_ngsm.detach().numpy(), color='orange', linewidth=2, label='Neural-GSM Posterior')
    axes1[3].fill_between(
        X_plot,
        (mean_ngsm - 2 * std_ngsm).detach().numpy(),
        (mean_ngsm + 2 * std_ngsm).detach().numpy(),
        alpha=0.3,
        color='orange',
        label=r'$\pm 2\sigma$'
    )
    axes1[3].scatter(X_train_plot, y_train_plot, c='red', s=20, alpha=0.3, label='Training data')
    axes1[3].set_xlabel('x')
    axes1[3].set_ylabel('y')
    axes1[3].set_title('Neural-GSM')
    axes1[3].legend(loc='best')
    axes1[3].grid(True)

    # DKL
    axes1[4].plot(X_plot, mean_true.detach().numpy(), 'gray', linewidth=2, linestyle='dashed', label='True', alpha=0.3)
    axes1[4].plot(X_plot, mean_dkl.detach().numpy(), color='purple', linewidth=2, label='DKL Posterior')
    axes1[4].fill_between(
        X_plot,
        (mean_dkl - 2 * std_dkl).detach().numpy(),
        (mean_dkl + 2 * std_dkl).detach().numpy(),
        alpha=0.3,
        color='purple',
        label=r'$\pm 2\sigma$'
    )
    axes1[4].scatter(X_train_plot, y_train_plot, c='red', s=20, alpha=0.3, label='Training data')
    axes1[4].set_xlabel('x')
    axes1[4].set_ylabel('y')
    axes1[4].set_title('DKL')
    axes1[4].legend(loc='best')
    axes1[4].grid(True)

    plt.tight_layout()

    # Plot kernel heatmaps
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    # Ground truth
    im0 = axes[0, 0].imshow(K_true_test.numpy(), cmap='viridis')
    axes[0, 0].set_title(f"True Kernel {kernel_name}")
    plt.colorbar(im0, ax=axes[0, 0])
    axes[1, 0].axis('off')

    methods = ['Standard GP', 'F-SDN', 'Neural-GSM', 'DKL']

    print("Results:")
    print(f"  {'Method':<15}  {'K-error':>10}  {'KL':>8}  {'MLL':>8}  {'noise_var':>10}")
    for i, method in enumerate(methods):
        K_pred = results[method].detach()

        rel_error = (torch.norm(K_pred - K_true_test) / torch.norm(K_true_test)).item()
        m = metrics[method]
        print(f"  {method:<15}  {rel_error:>10.2%}  {m['kl']:>8.3f}  {m['mll']:>8.3f}  {m['noise_var']:>10.2e}")

        # Plot kernel and error
        im = axes[0, i+1].imshow(K_pred.numpy(), cmap='viridis')
        axes[0, i+1].set_title(f"{method} Error: {rel_error:.1%}")
        plt.colorbar(im, ax=axes[0, i+1])

        diff = torch.abs(K_pred - K_true_test)
        im_diff = axes[1, i+1].imshow(diff.numpy(), cmap='hot')
        axes[1, i+1].set_title(f"|{method} - True|")
        plt.colorbar(im_diff, ax=axes[1, i+1])

    plt.tight_layout()
    plt.show()

    return results, metrics

if __name__ == "__main__":
    # Silverman non-stationary kernel
    lsk = LocalStationaryKernel(a=0.5)
    compare_kernels(
        lsk.kernel,
        kernel_name="Silverman Non-Stationary",
    )
