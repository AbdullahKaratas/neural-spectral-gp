import math
from typing import Optional, Callable
import torch
from warnings import warn


class RegularNonstationaryFeatures:
    def __init__(
        self,
        spectral: Callable[..., torch.Tensor],
        spectral_real: Optional[bool] = True,
        num_feat: Optional[int] = 100,
    ):
        self.spectral = spectral
        self.spectral_real = spectral_real
        self.num_feat = num_feat

    def matrix_decomposition(
        self, matrix, jitter: float = 1e-8, cholesky_max: int = 3
    ):
        for i in range(cholesky_max):
            try:
                matrix_root = torch.linalg.cholesky(
                    matrix + i * jitter * torch.eye(matrix.shape[0])
                )
                if i > 0:
                    warn(f"Jitter added: {i*jitter:.0e}")
                break
            except Exception:
                pass
            if i == cholesky_max - 1:
                raise RuntimeError(
                    "Repeatedly adding jitter did not result in positive-definiteness"
                )
        return matrix_root

    def lowrank(
        self,
        x1: torch.Tensor,
        spacing: float = 1.0,
        return_extras=False,
        **kwargs,
    ):
        x_max = x1.abs().max().item()
        if spacing > torch.pi / x_max:
            spacing = torch.pi / x_max - 1e-6
            warn(f"spacing set to {spacing:.2e} to avoid time-periodicity.")

        if self.spectral_real:
            omega = torch.arange(0, self.num_feat).reshape(-1, 1) * spacing
            matrix = self.spectral(omega, omega) * (spacing) ** 2
            matrix_root = self.matrix_decomposition(matrix, **kwargs)
            b1 = torch.cos(omega.reshape(1, -1) * x1.reshape(-1, 1))
            # Correction for zero-th element
            b1[:, 0] *= 0.5
            kernel_root = 2.0 * b1.matmul(matrix_root)
        else:
            omega = (
                torch.arange(-self.num_feat + 1, self.num_feat).reshape(-1, 1) * spacing
            )

            matrix = self.spectral(omega, omega) * (spacing) ** 2
            matrix_root = self.matrix_decomposition(matrix, **kwargs).to(
                torch.promote_types(matrix.dtype, torch.complex64)
            )
            b1 = torch.cos(omega.reshape(1, -1) * x1.reshape(-1, 1))
            b2 = torch.sin(omega.reshape(1, -1) * x1.reshape(-1, 1))
            b1[:, self.num_feat - 1] *= 0.5
            b2[:, self.num_feat - 1] *= 0.0

            phi_real = b1.matmul(matrix_root.real) - b2.matmul(matrix_root.imag)
            phi_im = b1.matmul(matrix_root.imag) + b2.matmul(matrix_root.real)

            # Factor sqrt(2.0) missing, because we naively integrate omega1,omega2 in R
            kernel_root = torch.cat([phi_real, phi_im], dim=1)

        if return_extras:
            return kernel_root, matrix_root, omega
        else:
            return kernel_root

    def simulation(
        self, x1: torch.Tensor, spacing: float = 1.0, n_samples: int = 1, seed=None, **kwargs
    ):
        if seed is not None:
            torch.manual_seed(seed)
        kernel_root = self.lowrank(x1=x1, spacing=spacing, **kwargs)
        rvs = torch.randn(kernel_root.shape[1], n_samples, dtype=kernel_root.dtype)
        if not self.spectral_real:
            rvs /= math.sqrt(2)
        return kernel_root.matmul(rvs).T

    def kernel_estimate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        spacing: float = 1.0,
    ):
        x1_eq_x2 = torch.equal(x1, x2)
        kernel_root, matrix_root, omega = self.lowrank(
            x1=x1, spacing=spacing, return_extras=True
        )
        if x1_eq_x2:
            if self.spectral_real:
                return kernel_root.matmul(kernel_root.transpose(0, 1))
            else:
                return kernel_root.matmul(kernel_root.transpose(0, 1).conj())
        else:
            if self.spectral_real:
                warn("Same spacing of x1 and x2 is assumed")
                b2 = torch.cos(omega.reshape(-1, 1) * x2.reshape(1, -1))
                # Correction for zero-th element
                b2[0, :] *= 0.5
                return kernel_root.matmul(matrix_root.transpose(0, 1)).matmul(b2)
            else:
                warn("Same spacing of x1 and x2 is assumed")
                b2 = torch.exp(-1j * omega.reshape(-1, 1) * x2.reshape(1, -1))
                return kernel_root.matmul(matrix_root.transpose(0, 1).conj()).matmul(b2)
