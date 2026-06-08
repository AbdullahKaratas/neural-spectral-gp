import pytest
import torch

from src.nsgp.models.sdn_factorized import FactorizedSpectralDensityNetwork


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enforce_symmetry", [False, True])
def test_omega_grid_respects_default_dtype(dtype, enforce_symmetry):
    """Regression for #25: omega_grid must follow torch's default dtype,
    not be pinned to float32, so float64 inputs don't raise a mismatch."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = FactorizedSpectralDensityNetwork(
            input_dim=1,
            hidden_dims=[8],
            rank=4,
            n_features=11,
            enforce_symmetry=enforce_symmetry,
        )

        # The buffer itself must match the active default dtype.
        assert model.omega_grid.dtype == dtype

        # And a forward pass with inputs of that dtype must not raise.
        X = torch.linspace(0.0, 1.0, 16, dtype=dtype).reshape(-1, 1)
        L = model.compute_lowrank_features(X)
        assert L.dtype == dtype
        assert torch.isfinite(L).all()
    finally:
        torch.set_default_dtype(prev)
