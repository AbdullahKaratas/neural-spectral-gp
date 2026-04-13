import math
import pytest
import torch

from nsgp.lowrank.random_nff import RandomNonstationaryFeatures


def silverman_sampler(m, a=1.0):
    std = math.sqrt(2.0 * a)
    omega1 = torch.randn(m, 1) * std
    omega2 = torch.randn(m, 1) * std
    return omega1, omega2


@pytest.fixture
def rnff():
    return RandomNonstationaryFeatures(spectral_sampler=silverman_sampler, n_feat=100)


@pytest.fixture
def x():
    return torch.linspace(0, 2.5, 50).unsqueeze(-1)


def test_features_before_sampling_raises(rnff):
    with pytest.raises(RuntimeError):
        rnff.compute_features(torch.randn(10, 1))


def test_feature_shape(rnff, x):
    rnff.sample_frequencies(seed=0)
    assert rnff.compute_features(x).shape == (50, 200)


def test_kernel_is_psd(rnff, x):
    L = rnff.lowrank(x, seed=0)
    eigs = torch.linalg.eigvalsh(L @ L.T)
    assert (eigs >= -1e-5).all()


def test_kernel_converges():
    """Random NFF converges to the symmetrized kernel, not the true Silverman kernel.

    The symmetrized kernel is:
        k_sym(x,x') = 0.5 * exp(-a(x^2+x'^2)) + 0.5 * exp(-a(x-x')^2)
    """
    a = 1.0
    x = torch.linspace(0, 2, 30).unsqueeze(-1)
    xx, yy = x, x.T
    K_sym = 0.5 * torch.exp(-a * (xx**2 + yy**2)) + 0.5 * torch.exp(-a * (xx - yy) ** 2)

    K_sum = torch.zeros_like(K_sym)
    for seed in range(50):
        rnff = RandomNonstationaryFeatures(
            spectral_sampler=lambda n: silverman_sampler(n, a=a), n_feat=500
        )
        rnff.sample_frequencies(seed=seed)
        K_sum += rnff.kernel_estimate(x)

    rel_err = torch.norm(K_sum / 50 - K_sym) / torch.norm(K_sym)
    assert rel_err < 0.15


def test_cross_kernel(rnff):
    x1 = torch.linspace(0, 1, 20).unsqueeze(-1)
    x2 = torch.linspace(1, 2, 15).unsqueeze(-1)
    rnff.sample_frequencies(seed=0)
    K12 = rnff.kernel_estimate(x1, x2)
    assert K12.shape == (20, 15)
    rnff.sample_frequencies(seed=0)
    K21 = rnff.kernel_estimate(x2, x1)
    assert torch.allclose(K12, K21.T, atol=1e-6)


def test_seed_reproducibility():
    rnff = RandomNonstationaryFeatures(spectral_sampler=silverman_sampler, n_feat=50)
    x = torch.linspace(0, 1, 20).unsqueeze(-1)
    L1 = rnff.lowrank(x, seed=42)
    L2 = rnff.lowrank(x, seed=42)
    assert torch.equal(L1, L2)


def test_simulation_shape(rnff, x):
    assert rnff.simulation(x, n_samples=5, seed=0).shape == (5, 50)
