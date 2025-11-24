"""
Sanity Check: Test consistency between MC, Deterministic, and Low-Rank methods.

This test verifies that all three covariance computation methods produce
the same result (up to numerical error).
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork

def test_consistency():
    print("=" * 80)
    print("SANITY CHECK: Method Consistency Test")
    print("=" * 80)

    # Create model
    torch.manual_seed(42)
    model = FactorizedSpectralDensityNetwork(
        input_dim=1,
        hidden_dims=[64, 64],
        rank=10,
        n_features=200,  # High resolution for accuracy
        omega_max=10.0
    )
    model.eval()

    # Setup test data
    X = torch.linspace(0, 5, 20).unsqueeze(1)
    omega_grid = torch.linspace(0, model.omega_max, model.n_features).unsqueeze(-1)

    print(f"\nTest setup:")
    print(f"  X shape: {X.shape}")
    print(f"  Frequency grid: {model.n_features} points from 0 to {model.omega_max}")
    print(f"  log_scale: {model.log_scale.item():.4f} (scale factor: {torch.exp(model.log_scale).item():.4f})")

    # 1. Low-Rank Features
    print(f"\n{'='*80}")
    print("METHOD 1: Low-Rank Features (used for TRAINING)")
    print('='*80)
    with torch.no_grad():
        L = model.compute_lowrank_features(X, omega_grid)
        K_lowrank = L @ L.T

    print(f"  L shape: {L.shape}")
    print(f"  K shape: {K_lowrank.shape}")
    print(f"  K stats: min={K_lowrank.min():.6f}, max={K_lowrank.max():.6f}")
    print(f"  Diagonal (variance): mean={K_lowrank.diag().mean():.6f}, std={K_lowrank.diag().std():.6f}")

    # 2. Deterministic Quadrature
    print(f"\n{'='*80}")
    print("METHOD 2: Deterministic Quadrature (used for EVALUATION)")
    print('='*80)
    with torch.no_grad():
        K_det = model.compute_covariance_deterministic(X, noise_var=0.0)

    print(f"  K shape: {K_det.shape}")
    print(f"  K stats: min={K_det.min():.6f}, max={K_det.max():.6f}")
    print(f"  Diagonal (variance): mean={K_det.diag().mean():.6f}, std={K_det.diag().std():.6f}")

    # 3. Monte Carlo Integration
    print(f"\n{'='*80}")
    print("METHOD 3: Monte Carlo Integration (for comparison)")
    print('='*80)
    with torch.no_grad():
        K_mc = model.compute_covariance_mc(X, noise_var=0.0, n_samples=5000)

    print(f"  K shape: {K_mc.shape}")
    print(f"  K stats: min={K_mc.min():.6f}, max={K_mc.max():.6f}")
    print(f"  Diagonal (variance): mean={K_mc.diag().mean():.6f}, std={K_mc.diag().std():.6f}")

    # Compute ratios
    print(f"\n{'='*80}")
    print("CONSISTENCY ANALYSIS")
    print('='*80)

    var_lowrank = K_lowrank.diag().mean().item()
    var_det = K_det.diag().mean().item()
    var_mc = K_mc.diag().mean().item()

    ratio_lowrank_det = var_lowrank / var_det
    ratio_mc_det = var_mc / var_det
    ratio_mc_lowrank = var_mc / var_lowrank

    print(f"\nVariance ratios:")
    print(f"  Low-Rank / Deterministic: {ratio_lowrank_det:.4f}")
    print(f"  MC / Deterministic:       {ratio_mc_det:.4f}")
    print(f"  MC / Low-Rank:            {ratio_mc_lowrank:.4f}")

    # Frobenius norm differences
    diff_lowrank_det = torch.norm(K_lowrank - K_det) / torch.norm(K_det)
    diff_mc_det = torch.norm(K_mc - K_det) / torch.norm(K_det)
    diff_mc_lowrank = torch.norm(K_mc - K_lowrank) / torch.norm(K_lowrank)

    print(f"\nRelative Frobenius differences:")
    print(f"  |Low-Rank - Deterministic| / |Deterministic|: {diff_lowrank_det:.4f} ({diff_lowrank_det*100:.2f}%)")
    print(f"  |MC - Deterministic| / |Deterministic|:       {diff_mc_det:.4f} ({diff_mc_det*100:.2f}%)")
    print(f"  |MC - Low-Rank| / |Low-Rank|:                 {diff_mc_lowrank:.4f} ({diff_mc_lowrank*100:.2f}%)")

    # Diagnose
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print('='*80)

    if diff_lowrank_det < 0.01:
        print("✅ Low-Rank and Deterministic are CONSISTENT (< 1% difference)")
    else:
        print(f"⚠️  Low-Rank and Deterministic differ by {diff_lowrank_det*100:.2f}%")

    if 0.9 < ratio_mc_det < 1.1:
        print("✅ MC and Deterministic have CONSISTENT SCALING (ratio ~1)")
    elif 3.8 < ratio_mc_det < 4.2:
        print("❌ MC has FACTOR 4 SCALING relative to Deterministic!")
        print("   → Solution: Remove factor 4 from compute_covariance_mc")
    else:
        print(f"⚠️  MC has INCONSISTENT SCALING (ratio = {ratio_mc_det:.4f})")

    if diff_mc_lowrank < 0.1:
        print("✅ MC and Low-Rank are reasonably close (< 10% difference)")
    elif diff_mc_lowrank > 0.5:
        print("❌ MC and Low-Rank have LARGE DIFFERENCES!")
        print("   This suggests a fundamental inconsistency in the implementations.")

if __name__ == "__main__":
    test_consistency()
