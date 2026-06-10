import pytest
import torch

from nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enforce_symmetry", [False, True])
def test_omega_grid_respects_default_dtype(dtype, enforce_symmetry):
    """omega_grid must follow torch's default dtype."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = FactorizedSpectralDensityNetwork(
            input_dim=1,
            enforce_symmetry=enforce_symmetry,
        )

        assert model.omega_grid.dtype == dtype

        # And a forward pass with inputs of that dtype must not raise.
        X = torch.linspace(0.0, 1.0, 10, dtype=dtype).reshape(-1, 1)
        L = model.compute_lowrank_features(X)
        assert L.dtype == dtype
        assert torch.isfinite(L).all()
    finally:
        torch.set_default_dtype(prev)


@pytest.mark.parametrize("ed", [0, 8])
def test_fourier_embedding(ed):
    """Fourier embedding (Tancik et al., 2020) is off when embedding_dim=0
    and on otherwise."""
    model = FactorizedSpectralDensityNetwork(
        input_dim=1,
        embedding_dim=ed,
    )
    # B (the random projection) exists iff the embedding is on.
    assert hasattr(model, "B") == (ed > 0)

    feat = model.compute_features(torch.linspace(-8, 8, 21).unsqueeze(-1))
    assert feat.shape == (21, model.rank)
    assert torch.isfinite(feat).all()
