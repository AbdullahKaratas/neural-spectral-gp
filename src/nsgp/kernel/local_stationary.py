import torch
from ..utils import sq_exp


class LocalStationaryKernel:
    """
    Silverman's local stationary kernel.
    """

    def __init__(self, a=1.0):
        self.a = a

    def stationary_kernel(self, x1, x2):
        dist_mat = sq_exp(x1, x2, dist=True)
        return dist_mat.div_(-2).exp_()

    def kernel(self, x1, x2):
        add_mat = sq_exp(x1 / 2.0, x2 / 2.0, dist=False)
        add_mat.mul_(-2).mul_(self.a).exp_()

        dist_mat = sq_exp(x1, x2, dist=True)
        dist_mat.div_(-2).mul_(self.a).exp_()

        return add_mat * dist_mat

    def spectral(self, omega1, omega2):
        add_mat = sq_exp(omega1 / 2.0, omega2 / 2.0, dist=False)
        add_mat.mul_(-0.5).div_(self.a).exp_()

        dist_mat = sq_exp(omega1, omega2, dist=True)
        dist_mat.mul_(-0.125).div_(self.a).exp_()

        return 0.25 / torch.pi / self.a * add_mat * dist_mat
