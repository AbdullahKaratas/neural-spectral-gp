import torch


class LocalStationaryKernel:
    """
    Silverman's local stationary kernel.
    """

    def __init__(self, a=1.0):
        self.a = a

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

    def stationary_kernel(self, x1, x2):
        dist_mat = self._sq_exp(x1, x2, dist=True)
        return dist_mat.div_(-2).exp_()

    def kernel(self, x1, x2):
        add_mat = self._sq_exp(x1 / 2.0, x2 / 2.0, dist=False)
        add_mat.mul_(-2).mul_(self.a).exp_()

        dist_mat = self._sq_exp(x1, x2, dist=True)
        dist_mat.div_(-2).mul_(self.a).exp_()

        return add_mat * dist_mat

    def spectral(self, omega1, omega2):
        add_mat = self._sq_exp(omega1 / 2.0, omega2 / 2.0, dist=False)
        add_mat.mul_(-0.5).div_(self.a).exp_()

        dist_mat = self._sq_exp(omega1, omega2, dist=True)
        dist_mat.mul_(-0.125).div_(self.a).exp_()

        return 0.25 / torch.pi / self.a * add_mat * dist_mat
