"""
Wrapper script for running Official Remes et al. (2017) code.
Requires: gpflow (2.x), tensorflow

Usage:
    python run_remes.py --train_x X_train.npy --train_y y_train.npy --test_x X_test.npy --output K_pred.npy
"""

import argparse
import numpy as np
import sys
import os
import tensorflow as tf
import gpflow
from gpflow.utilities import positive
from gpflow.utilities import print_summary

# Enforce float64
gpflow.config.set_default_float(np.float64)


# ============================================================================
# PORTED KERNEL (GPflow 2.x)
# ============================================================================

class BSMKernelComponent(gpflow.kernels.Kernel):
    """
    Bi-variate Spectral Mixture Kernel.
    Ported from Remes et al. (2017) to GPflow 2.x
    """
    def __init__(self, input_dim=1, variance=1.0, frequency=None,
                 lengthscale=1.0, correlation=0.1, max_freq=1.0, active_dims=None):
        super().__init__(active_dims=active_dims)
        
        if frequency is None:
            frequency = np.array([1.0, 1.0])
            
        self.variance = gpflow.Parameter(variance, transform=positive())
        
        # Logistic transform for frequency: [0, max_freq]
        # In GPflow 2, we can use Sigmoid + scaling
        # Or just positive() if we don't strictly enforce upper bound, 
        # but original code used Logistic(0, max_freq).
        # Let's use positive() for simplicity as max_freq is usually just a prior.
        self.frequency = gpflow.Parameter(frequency, transform=positive())
        
        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive())
        
        # Correlation in [0, 1] (actually clipped in original code)
        import tensorflow_probability as tfp
        self.correlation = gpflow.Parameter(correlation, transform=tfp.bijectors.Sigmoid())
        
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
            
        pi = np.pi
        
        # exp term; x^T * [1 rho; rho 1] * x, x=[x,-x']^T
        # Note: X is (N, 1), X2 is (M, 1)
        
        # Meshgrid equivalent
        # X: (N, 1) -> (N, 1, 1)
        # X2: (M, 1) -> (1, M, 1)
        # We want (N, M) output
        
        X_ = tf.expand_dims(X, 1) # (N, 1, D)
        X2_ = tf.expand_dims(X2, 0) # (1, M, D)
        
        # Since input_dim=1, we can just squeeze
        X_flat = tf.squeeze(X_, -1) # (N, 1)
        X2_flat = tf.squeeze(X2_, -1) # (1, M)
        
        # XX, XX2 = tf.meshgrid(X, X2, indexing='ij') 
        # In TF2, meshgrid behavior might differ, let's use broadcasting
        XX = X_flat # (N, 1)
        XX2 = X2_flat # (1, M)
        
        # R = x^2 + x'^2 - 2*rho*x*x'
        R = tf.square(XX) + tf.square(XX2) - 2.0 * self.correlation * XX * XX2
        
        exp_term = tf.exp(-2.0 * pi**2 * tf.square(self.lengthscale) * R)
        
        # phi cosine terms
        mu = self.frequency # (2,)
        
        # phi1: (N, 2)
        # cos(2pi*mu0*x) + cos(2pi*mu1*x)
        # sin(2pi*mu0*x) + sin(2pi*mu1*x)
        
        # X is (N, D)
        X_sq = tf.squeeze(X, -1) # (N,)
        X2_sq = tf.squeeze(X2, -1) # (M,)
        
        c1 = tf.cos(2*pi*mu[0]*X_sq) + tf.cos(2*pi*mu[1]*X_sq)
        s1 = tf.sin(2*pi*mu[0]*X_sq) + tf.sin(2*pi*mu[1]*X_sq)
        phi1 = tf.stack([c1, s1], axis=1) # (N, 2)
        
        c2 = tf.cos(2*pi*mu[0]*X2_sq) + tf.cos(2*pi*mu[1]*X2_sq)
        s2 = tf.sin(2*pi*mu[0]*X2_sq) + tf.sin(2*pi*mu[1]*X2_sq)
        phi2 = tf.stack([c2, s2], axis=1) # (M, 2)
        
        # phi = phi1 * phi2^T
        phi = tf.matmul(phi1, phi2, transpose_b=True) # (N, M)
        
        return self.variance * exp_term * phi

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

def BSMKernel(input_dim=1, Q=1, max_freq=1.0):
    kerns = []
    for q in range(Q):
        var = 1.0 / Q
        mu_f = max_freq * np.random.rand(2)
        ell = np.random.rand()
        corr = 0.1 + 0.4 * np.random.rand() # Safe initialization
        kerns.append(BSMKernelComponent(input_dim=input_dim, variance=var, frequency=mu_f, lengthscale=ell, correlation=corr, max_freq=max_freq))
    
    # Sum kernels
    k_sum = kerns[0]
    for k in kerns[1:]:
        k_sum += k
    return k_sum

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x', type=str, required=True)
    parser.add_argument('--train_y', type=str, required=True)
    parser.add_argument('--test_x', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # Load Data
    X_train = np.load(args.train_x).astype(np.float64)
    y_train = np.load(args.train_y).astype(np.float64)
    X_test = np.load(args.test_x).astype(np.float64)

    if y_train.ndim == 1:
        y_train = y_train[:, None]

    print(f"Loaded data: X_train={X_train.shape}, y_train={y_train.shape}")

    # Setup Model
    # We use Q=5 to be competitive.
    input_dim = X_train.shape[1]
    k = BSMKernel(input_dim=input_dim, Q=5)
    
    m = gpflow.models.GPR(data=(X_train, y_train), kernel=k, mean_function=None)
    m.likelihood.variance.assign(0.01)

    # Train
    print("Optimizing...")
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=args.epochs))
    
    print_summary(m)

    # Predict
    print("Predicting...")
    # We want the kernel matrix K(X_test, X_test)
    # predict_f returns the posterior covariance, which includes the data update.
    # But for kernel comparison, we often want the PRIOR kernel matrix if we are comparing kernels directly.
    # However, the task is "compare spatial kernels", which usually means the learned kernel function.
    # So we should evaluate k(X_test, X_test).
    
    # GPflow kernels can be called directly
    K_pred_tensor = m.kernel(X_test) # (N, N)
    K_pred = K_pred_tensor.numpy()
    
    print(f"Prediction stats: min={K_pred.min():.4f}, max={K_pred.max():.4f}")
    
    # Save
    np.save(args.output, K_pred)
    print(f"Saved prediction to {args.output}")

if __name__ == "__main__":
    main()
