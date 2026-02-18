import torch
import math
from typing import List


class HarmonizableMixtureKernel:
    """
    Harmonizable Mixture Kernel (HMK).
    """

    def __init__(
        self,
        sigma1,
        sigma2,
        centers: List[torch.Tensor],
        scalings: List[torch.Tensor],
        frequencies: List[torch.Tensor],
        psd_matrices: List[torch.Tensor],
    ):
        # Cholesky decomposition: sigma = L L^top
        self.L1 = torch.linalg.cholesky(sigma1)
        self.L2 = torch.linalg.cholesky(sigma2)

        # Inverse Cholesky transpose for spectral density
        self.L1_inv_T = torch.linalg.inv(self.L1).T
        self.L2_inv_T = torch.linalg.inv(self.L2).T
        # Log determinants
        self.neg_log_det_L1 = -torch.sum(torch.log(torch.diag(self.L1)))
        self.neg_log_det_L2 = -torch.sum(torch.log(torch.diag(self.L2)))

        self.num_components = len(centers)
        # x_p
        self.centers = centers
        # gamma_p
        self.scalings = scalings
        # mu_p vectors
        self.frequencies = frequencies
        # B_p matrices
        self.psd_matrices = psd_matrices

    def _sq_exp(self, x1, x2, dist=True):
        x1_eq_x2 = torch.equal(x1, x2)

        if dist:
            adjustment = x1.mean(-2, keepdim=True)
        else:
            adjustment = 0.0
        x1 = x1 - adjustment

        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            x2, x2_norm, x2_pad = x1, x1_norm, x1_pad
        else:
            x2 = (
                x2 - adjustment
            )  # x1 and x2 should be identical in all dims except -2 at this point
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            x2_pad = torch.ones_like(x2_norm)
        if dist:
            x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
        else:
            x1_ = torch.cat([2.0 * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad and dist:
            res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values
        return res.clamp_min_(0)

    def k_lsg(self, x1, x2):
        """
        Locally stationary Gaussian kernel.
        """
        if x1.dim() == 1:
            x1 = x1.unsqueeze(-1)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(-1)

        # For bar{x}^T sigma1 bar{x}:
        # transform by L1^T/2 so _sq_exp with dist=False gives |bar{x} @ L1^T|^2
        x1_t1 = x1 @ self.L1.T / 2.0
        x2_t1 = x2 @ self.L1.T / 2.0
        add_mat = self._sq_exp(x1_t1, x2_t1, dist=False)
        add_mat.mul_(-2 * math.pi**2).exp_()

        # For tau^T sigma2 tau:
        # transform by L2^T so _sq_exp with dist=True gives |tau @ L2^T|^2
        x1_t2 = x1 @ self.L2.T
        x2_t2 = x2 @ self.L2.T
        dist_mat = self._sq_exp(x1_t2, x2_t2, dist=True)
        dist_mat.mul_(-2 * math.pi**2).exp_()

        return add_mat * dist_mat

    def k_p(self, x1, x2, gamma_p, mu_p, B_p):
        """
        Single component kernel.
        """
        if x1.dim() == 1:
            x1 = x1.unsqueeze(-1)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(-1)

        # (x1 - x_p) * gamma_p
        x1_scaled = x1 * gamma_p
        x2_scaled = x2 * gamma_p

        k_lsg_mat = self.k_lsg(x1_scaled, x2_scaled)

        # spectral feature maps phi_p(x)
        phi_x1 = torch.exp(1j * 2 * math.pi * (x1 @ mu_p.T))
        phi_x2 = torch.exp(-1j * 2 * math.pi * (x2 @ mu_p.T))

        # phi_p(x1) B_p phi_p(x2)^H
        spectral_term = phi_x1 @ B_p @ phi_x2.transpose(-2, -1)

        return k_lsg_mat * spectral_term

    def k_hmk(self, x1, x2):
        """
        Harmonizable Mixture Kernel.
        """
        if x1.dim() == 1:
            x1 = x1.unsqueeze(-1)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(-1)

        kernel = 0.0

        for p in range(self.num_components):
            # Shift inputs
            x1_shifted = x1 - self.centers[p]
            x2_shifted = x2 - self.centers[p]

            # Component kernel
            component_kernel = self.k_p(
                x1_shifted,
                x2_shifted,
                self.scalings[p],
                self.frequencies[p],
                self.psd_matrices[p],
            )
            kernel += component_kernel

        return kernel

    def s_lsg(self, omega1, omega2):
        """
        Spectral density of the locally stationary Gaussian kernel.
        """
        if omega1.dim() == 1:
            omega1 = omega1.unsqueeze(-1)
        if omega2.dim() == 1:
            omega2 = omega2.unsqueeze(-1)

        d = self.L1.shape[0]

        # Convert from angular frequency omega to frequency f: f = omega/(2pi)
        omega1_ = omega1 / (2.0 * math.pi)
        omega2_ = omega2 / (2.0 * math.pi)

        # Normalization: log((2pi)^(-3d) |Sigma1|^(-1/2) |Sigma2|^(-1/2))
        log_norm = (
            -3 * d * math.log(2 * math.pi) + self.neg_log_det_L1 + self.neg_log_det_L2
        )

        # Transform by L1_inv_T so _sq_exp with dist=True gives
        # |(omega1 - omega2) @ L1_inv_T|^2
        omega1_t1 = omega1_ @ self.L1_inv_T
        omega2_t1 = omega2_ @ self.L1_inv_T
        diff_sq = self._sq_exp(omega1_t1, omega2_t1, dist=True)

        # Transform by L2_inv_T/2 so _sq_exp with dist=False gives |mean @ L2_inv_T|^2
        omega1_t2 = omega1_ @ self.L2_inv_T / 2.0
        omega2_t2 = omega2_ @ self.L2_inv_T / 2.0
        mean_sq = self._sq_exp(omega1_t2, omega2_t2, dist=False)

        result = torch.exp(log_norm - 0.5 * diff_sq - 0.5 * mean_sq)

        return result

    def s_kp(self, omega1, omega2, gamma_p, mu_p, B_p):
        """
        Component spectral density S_kp(omega1, omega2) for component p.
        """
        if omega1.dim() == 1:
            omega1 = omega1.unsqueeze(-1)
        if omega2.dim() == 1:
            omega2 = omega2.unsqueeze(-1)

        # Number of frequencies
        Q_p = mu_p.shape[0]

        # Scaling normalization: 1 / prod_d gamma_pd^2
        gamma_normalization = 1.0 / torch.prod(gamma_p**2)

        # Initialize
        n1, n2 = omega1.shape[0], omega2.shape[0]
        spectral_sum = torch.zeros(n1, n2, dtype=torch.complex128, device=omega1.device)

        # Sum over all pairs
        for i in range(Q_p):
            for j in range(Q_p):
                # S_ij = S_kLSG((omega1 - mu_i) ./ gamma, (omega2 - mu_j) ./ gamma)
                omega1_ = (omega1 - 2.0 * math.pi * mu_p[i]) / gamma_p
                omega2_ = (omega2 - 2.0 * math.pi * mu_p[j]) / gamma_p

                S_ij = self.s_lsg(omega1_, omega2_)

                # weighted by B_p[i, j]
                spectral_sum += B_p[i, j] * S_ij

        return gamma_normalization * spectral_sum

    def s_khm(self, omega1, omega2):
        """
        Harmonizable Mixture Kernel spectral density.
        """
        if omega1.dim() == 1:
            omega1 = omega1.unsqueeze(-1)
        if omega2.dim() == 1:
            omega2 = omega2.unsqueeze(-1)

        n1, n2 = omega1.shape[0], omega2.shape[0]
        spectral_density = torch.zeros(
            n1, n2, dtype=torch.complex128, device=omega1.device
        )

        omega_diff = omega1[:, None, :] - omega2[None, :, :]

        # Sum over P components
        for p in range(self.num_components):
            S_kp = self.s_kp(
                omega1,
                omega2,
                self.scalings[p],
                self.frequencies[p],
                self.psd_matrices[p],
            )

            phase = -1j * torch.sum(self.centers[p] * omega_diff, dim=-1)
            phase_term = torch.exp(phase)

            spectral_density += S_kp * phase_term

        return spectral_density
