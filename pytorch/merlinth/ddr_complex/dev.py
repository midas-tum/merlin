
import torch
from merlinth import mytorch
import unittest
from optoth.activations import TrainableActivation
import numpy as np
from .complex_loss import *

class ComplexModPReLU_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, bias, slope, eps):
        ctx.save_for_backward(z)
        ctx.bias = bias
        ctx.slope = slope
        ctx.eps = eps
        magn = mytorch.complex.complex_abs(z, eps=eps, keepdim=True)
        return (torch.clamp(magn + bias, min=0) + slope * torch.clamp(magn + bias, max=0)) * z / magn

    @staticmethod
    def backward(ctx, grad_in):
        z = ctx.saved_tensors[0]
        bias = ctx.bias
        slope = ctx.slope
        eps = ctx.eps

        magn = mytorch.complex.complex_abs(z, eps=eps, keepdim=True)

        grad_inH = mytorch.complex.complex_conj(grad_in)
        dz = 1 + bias / (2 * magn)
        dz = torch.stack([dz[...,0], torch.zeros(*dz.shape[:-1], dtype=z.dtype, device=z.device)], dim=-1)

        dzH = - bias * mytorch.complex.complex_mult(z, z) / (2 * magn**3)
        grad_out = mytorch.complex.complex_mult(grad_in, dz) + mytorch.complex.complex_mult(grad_inH, dzH)
       
        masked_slope = torch.where((magn + bias) > 0, torch.ones(*magn.shape, dtype=magn.dtype, device=magn.device), slope * torch.ones(*magn.shape, dtype=magn.dtype, device=magn.device))
        grad_in = grad_out * masked_slope

        dbias = z / magn * masked_slope
        dbiasH = mytorch.complex.complex_conj(dbias)
        grad_bias = mytorch.complex.complex_mult(grad_inH, dbias) + mytorch.complex.complex_mult(grad_in, dbiasH)
        grad_bias = grad_bias[...,0].unsqueeze_(-1)
        #grad_bias = grad_bias.sum(-1, keepdim=True)
        dim=(0,*[l for l in range(2,z.ndim)])
        grad_bias = grad_bias.sum(dim=dim, keepdim=True)

        dslope = (bias + magn) * z / magn
        dslopeH = mytorch.complex.complex_conj(dslope)
        grad_slope = mytorch.complex.complex_mult(grad_inH, dslope) + mytorch.complex.complex_mult(grad_in, dslopeH)
        #grad_slope = grad_slope[...,0].unsqueeze_(-1)
        grad_slope = grad_slope.sum(-1, keepdim=True)
        grad_slope = torch.where(magn + bias <= 0, grad_slope, torch.zeros(*magn.shape, dtype=magn.dtype, device=magn.device) )
        dim=(0,*[l for l in range(2,z.ndim)])
        grad_slope = grad_slope.sum(dim=dim, keepdim=True)

        return grad_in, grad_bias, grad_slope, None

class TestModPReLU(unittest.TestCase):   
    def _test_gradient(self, shape):
        # setup the hyper parameters for each test
        dtype = torch.float64
        nf=5

        # perform a gradient check:
        epsilon = 1e-6

        # prefactors
        a = 1.1
        b = 1.1
        c = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x = torch.randn(*shape).to(dtype=dtype, device=cuda)
        th_v = torch.randn(nf).to(dtype=dtype, device=cuda)
        th_v = th_v.view(1, -1, *[1 for d in range(th_x.ndim-2)])
        th_w = torch.randn(nf).to(dtype=dtype, device=cuda)
        th_w = th_w.view(1, -1, *[1 for d in range(th_x.ndim-2)])

        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        th_b = torch.tensor(b, requires_grad=True, dtype=th_x.dtype, device=cuda)
        th_c = torch.tensor(c, requires_grad=True, dtype=th_x.dtype, device=cuda)

        op = ComplexModPReLU_fun()
        loss = ComplexL2Loss()
        # setup the model
        compute_loss = lambda a, b, c: loss(op.apply(th_x*a, th_v*b, th_w*c, 1e-12))
        th_loss = compute_loss(th_a, th_b, th_c)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()
        grad_b = th_b.grad.cpu().numpy()
        grad_c = th_c.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon, th_b, th_c).cpu().numpy()
            l_an = compute_loss(th_a-epsilon, th_b, th_c).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        #self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

        with torch.no_grad():
            l_bp = compute_loss(th_a, th_b+epsilon, th_c).cpu().numpy()
            l_bn = compute_loss(th_a, th_b-epsilon, th_c).cpu().numpy()
            grad_b_num = (l_bp - l_bn) / (2 * epsilon)

        print("grad_v: {:.7f} num_grad_v {:.7f} success: {}".format(
       
            grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
      #  self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)

        with torch.no_grad():
            l_cp = compute_loss(th_a, th_b, th_c+epsilon).cpu().numpy()
            l_cn = compute_loss(th_a, th_b, th_c-epsilon).cpu().numpy()
            grad_c_num = (l_cp - l_cn) / (2 * epsilon)

        print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
            grad_c, grad_c_num, np.abs(grad_c - grad_c_num) < 1e-4))
        #self.assertTrue(np.abs(grad_c - grad_c_num) < 1e-4)

    def test_gradient1(self):
        self._test_gradient((2,5,10,10,2))
    def test_gradient2(self):
        self._test_gradient((2,5,10,2))

if __name__ == "__main__":
    unittest.test()