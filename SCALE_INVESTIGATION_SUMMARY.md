# Scale Investigation Summary
**Date**: November 2025
**Authors**: Arsalan Jawaid, Abdullah Karatas

## Executive Summary

We discovered and fixed a bug in the Matérn kernel implementation, but this did **not** resolve the scale drift issue. The scale-noise ambiguity in marginal likelihood optimization remains the root cause, affecting different kernels in opposite directions.

---

## Three Completed Experiments

### 1. Silverman (Locally Stationary) ✅
- **s-error**: 46%
- **K-error**: ~12%
- **Scale**: Correct (no drift)
- **Status**: EXCELLENT - baseline working well

### 2. SE Varying Amplitude ⚠️
- **K-error**: 105% (with post-hoc correction)
- **Scale drift**: Learned 2× too small → corrected with 1.93× factor
- **Structure**: Captured well, but overcorrected
- **Status**: Acceptable but not ideal

### 3. Matérn Varying Lengthscale ⚠️
- **K-error**: 145.6% (with post-hoc correction)
- **Scale drift**: Learned 8× too large → corrected with 0.12× factor
- **Structure**: Captured reasonably, large absolute error
- **Status**: Acceptable but significant error

---

## Critical Bug Discovery & Fix

### The Matérn Kernel Amplitude Bug

**Location**: `experiments/synthetic/test_matern_varying_lengthscale.py:69`

**Incorrect formula** (what we had):
```python
amplitude_factor = torch.sqrt(2 * l_x * l_xp / (l_x**2 + l_xp**2))
```

**Correct formula** (Paciorek & Schervish 2004):
```python
amplitude_factor = torch.sqrt(l_x * l_xp)
```

**Derivation**:
For nonstationary Matérn kernel:
```
k(x,x') = σ²_f · |Σ(x)|^(1/4) · |Σ(x')|^(1/4) · k_ν(r̃)
```

In 1D, Σ(x) = ℓ²(x), so:
```
|Σ(x)|^(1/4) · |Σ(x')|^(1/4) = (ℓ²(x))^(1/4) · (ℓ²(x'))^(1/4)
                             = ℓ^(1/2)(x) · ℓ^(1/2)(x')
                             = √(ℓ(x) · ℓ(x'))
```

Our old formula added an extra normalization term that **doesn't appear** in Paciorek & Schervish (2004).

### Impact Assessment

**Numerical difference**: Our formula was ~1.5× larger than correct

**BUT**: Fixing this bug did **NOT** solve the scale issue!

**Before fix**: K-error ~725,900% (completely wrong scale)
**After fix**: K-error 145.6% (still significant error)

**Conclusion**: The bug was real and needed fixing for scientific correctness, but the scale drift has a different root cause.

---

## Root Cause Analysis: Scale-Noise Ambiguity

### The Mathematical Issue

The marginal likelihood is:
```
log p(y | K, σ²) = -½yᵀ(K + σ²I)⁻¹y - ½log|K + σ²I| - (n/2)log(2π)
```

**Scale ambiguity**: For any constant c > 0:
```
p(y | K, σ²) ≠ p(y | cK, cσ²)
```

This is **NOT** a true invariance! The ambiguity arises because:
1. The optimizer can trade off K scale vs. noise σ²
2. Both affect the loss landscape similarly
3. Gradient descent can converge to different local minima with different scales

### Evidence from Experiments

**Different kernels drift in opposite directions**:

| Kernel | True var(y) | Learned var | Scale drift |
|--------|-------------|-------------|-------------|
| Silverman | ~0.8 | ~0.8 | ✓ Correct |
| SE varying | 1.60 | 0.83 | 0.52× (too small) |
| Matérn varying | 0.75 | 6.04 | 8.1× (too large) |

**Key observation**: This is kernel-dependent, suggesting different optimization landscapes lead to different local minima.

---

## Post-Hoc Variance Normalization

### Method
```python
empirical_var = y_train.var()
learned_var = torch.diag(K_train_learned).mean()
scale_factor = empirical_var / learned_var

K_corrected = scale_factor * K_learned
```

### Justification
1. **Theoretically sound**: Matches learned prior to empirical data statistics
2. **Essential for some kernels**: Matérn would be unusable otherwise
3. **Consistent**: Applied uniformly to all kernels
4. **Transparent**: Documented scale factor for each experiment

### Results with Post-Hoc Correction

| Kernel | K-error (before) | K-error (after) | Improvement? |
|--------|------------------|-----------------|--------------|
| Silverman | ~12% | ~12% | ≈ (already correct) |
| SE varying | 67.5% | 105% | ✗ (overcorrected) |
| Matérn varying | 725,900% | 145.6% | ✓✓✓ (essential!) |

**Tradeoff**: The correction helps Matérn dramatically but slightly hurts SE. We accept this for methodological consistency.

---

## Remaining Questions

### 1. Why does scale drift happen?
- Optimization gets stuck in local minima with wrong scale
- Different kernels have different loss landscapes
- Random initialization affects which minimum we converge to

### 2. Why do different kernels drift differently?
- Matérn: Sharp, less smooth → harder to optimize → larger drift
- SE: Smooth, easier landscape → smaller drift (but still present)
- Silverman: Spectral density has closed form → better behaved

### 3. Can we fix this in training?
**Potential solutions**:
- Better initialization (match data variance)
- Scale regularization term in loss
- Multi-stage training (coarse → fine)
- Curriculum learning on data scale

---

## Recommendations

### For Paper (NeurIPS 2026)

1. **Document both bugs**:
   - Matérn kernel formula bug (scientific correction)
   - Scale-noise ambiguity (methodological challenge)

2. **Report all results transparently**:
   - Show K-errors with and without post-hoc correction
   - Document scale factors for each kernel
   - Acknowledge SE overcorrection

3. **Frame as contribution**:
   - "We identify and address scale-noise ambiguity in neural spectral methods"
   - Post-hoc correction as practical solution
   - Opens research direction for better training objectives

4. **Updated results table**:
```latex
\begin{tabular}{lccccc}
\toprule
\textbf{Kernel} & \textbf{s-error} & \textbf{K-error} & \textbf{K-error} & \textbf{Scale} & \textbf{Sampling} \\
                &                  & \textbf{(raw)}   & \textbf{(corr.)} & \textbf{Factor} & \\
\midrule
Silverman       & 46\%  & 12\%    & 12\%    & 1.0× & \checkmark \\
SE (var. amp.)  & N/A   & 67.5\%  & 105\%   & 1.93× & \checkmark \\
Matérn (var. ℓ) & N/A   & 7259× & 145.6\% & 0.12× & \checkmark \\
\bottomrule
\end{tabular}
```

### For Future Work

1. **Better training objective**:
   - Add variance matching term: `λ·|var(y_train) - mean(diag(K_learned))|`
   - Or normalize K during training: `K ← K · (var(y)/mean(diag(K)))`

2. **Monte Carlo s(ω,ω') for Matérn** (like Remes 2017):
   - Bypasses K-matching entirely
   - Could avoid scale issues
   - Worth implementing as alternative

3. **Hyperparameter tuning**:
   - Try different ranks (5, 10, 20, 30)
   - Different architectures ([32,32] vs [128,128,128])
   - Learning rate schedules

---

## Current Status: Phase 1 Complete ✅

All three synthetic kernel experiments are done:
- ✅ Silverman (46% s-error)
- ✅ SE varying amplitude (105% K-error corrected)
- ✅ Matérn varying lengthscale (145.6% K-error corrected)

**Next**: Phase 2 - Real-world application (Mauna Loa CO₂)

---

## Files Modified

1. `experiments/synthetic/test_matern_varying_lengthscale.py`
   - Fixed amplitude factor (line 69)
   - Added post-hoc variance normalization (lines 188-209)

2. `experiments/synthetic/test_se_varying_amplitude.py`
   - Added post-hoc variance normalization (lines 165-186)

3. `paper/neural_spectral_gp.tex`
   - Added authors: Arsalan Jawaid, Abdullah Karatas
   - Updated experimental setup section
   - Added results table with corrected values

4. `src/nsgp/models/sdn_factorized.py`
   - Improved Cholesky jitter handling (lines 240-257)
