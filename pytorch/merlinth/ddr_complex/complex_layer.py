import torch
import torch.nn.functional as F
import optoth.pad2d

import numpy as np

import unittest
from merlinth import mytorch
from .complex_loss import *

class ComplexNormalization_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, eps):
        ctx.save_for_backward(z)
        ctx.eps = eps
        magn = mytorch.complex.complex_abs(z, eps=eps, keepdim=True)
        return z / magn

    @staticmethod
    def backward(ctx, grad_out):
        z = ctx.saved_tensors[0]
        z_conj = mytorch.complex.complex_conj(z)
        grad_out_conj = mytorch.complex.complex_conj(grad_out)
        magn = mytorch.complex.complex_abs(z, eps=ctx.eps, keepdim=True)
        frac = mytorch.complex.complex_div(z, z_conj)
        return  1 / (2 * magn) * (grad_out_conj - mytorch.complex.complex_mult(grad_out, frac)), None

class ComplexNormalization(torch.nn.Module):
    def __init__(self):
        super(ComplexNormalization, self).__init__()

    def forward(self, z, eps=1e-9):
        assert z.shape[-1] == 2
        return ComplexNormalization_fun.apply(z, eps)

class ComplexMagnitude_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, eps):
        ctx.save_for_backward(z)
        ctx.eps = eps
        magn = mytorch.complex.complex_abs(z, eps=eps, keepdim=True)
        return magn

    @staticmethod
    def backward(ctx, grad_out):
        z = ctx.saved_tensors[0]
        eps = ctx.eps
        magn = mytorch.complex.complex_abs(z, eps=eps, keepdim=True)
        norm = z / magn
        return  grad_out * norm, None

class ComplexMagnitude(torch.nn.Module):
    def __init__(self):
        super(ComplexMagnitude, self).__init__()

    def forward(self, z, eps=1e-9):
        assert z.shape[-1] == 2
        return ComplexMagnitude_fun.apply(z, eps)

class RealLayer_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        return z[...,0].clone().unsqueeze_(-1)

    @staticmethod
    def backward(ctx, grad_out):
        return torch.cat([grad_out, torch.zeros(*grad_out.shape, device=grad_out.device, dtype=grad_out.dtype)], -1)

class RealLayer(torch.nn.Module):
    def __init__(self):
        super(RealLayer, self).__init__()

    def forward(self, z):
        assert z.shape[-1] == 2
        return RealLayer_fun.apply(z)

class ImagLayer_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        comp = z[...,1].clone().unsqueeze_(-1)
        return comp

    @staticmethod
    def backward(ctx, grad_out):
        return torch.cat([torch.zeros(*grad_out.shape, device=grad_out.device, dtype=grad_out.dtype), grad_out], -1)

class ImagLayer(torch.nn.Module):
    def __init__(self):
        super(ImagLayer, self).__init__()

    def forward(self, z):
        assert z.shape[-1] == 2
        return ImagLayer_fun.apply(z)

class TestMagnitudeLayer(unittest.TestCase):   
    def _test_gradient(self, shape):
        # setup the hyper parameters for each test
        dtype = torch.float64

        # perform a gradient check:
        epsilon = 1e-6

        # prefactors
        a = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x =  torch.randn(*shape).to(dtype=dtype, device=cuda)
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        op = ComplexMagnitude_fun()

        # setup the model
        compute_loss = lambda a: 0.5 * torch.sum(op.apply(th_x*a, 1e-9)**2)
        th_loss = compute_loss(th_a)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon).cpu().numpy()
            l_an = compute_loss(th_a-epsilon).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

    def test_gradient1(self):
        self._test_gradient((2,5,2))
    def test_gradient2(self):
        self._test_gradient((2,5,10,10,2))

class TestNormalizationLayer(unittest.TestCase):   
    def _test_gradient(self, shape):
        # setup the hyper parameters for each test
        dtype = torch.float64

        # perform a gradient check:
        epsilon = 1e-6

        # prefactors
        a = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x =  torch.randn(*shape).to(dtype=dtype, device=cuda)
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        op = ComplexNormalization_fun()
        lossop = ComplexL2Loss_fun()
        # setup the model
        compute_loss = lambda a: 0.5 * lossop.apply(op.apply(th_x*a, 1e-9))
        th_loss = compute_loss(th_a)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon).cpu().numpy()
            l_an = compute_loss(th_a-epsilon).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

    def test_gradient1(self):
        self._test_gradient((2,5,2))
    def test_gradient2(self):
        self._test_gradient((2,5,10,10,2))

class TestRealLayer(unittest.TestCase):   
    def _test_gradient(self, shape):
        # setup the hyper parameters for each test
        dtype = torch.float64

        # perform a gradient check:
        epsilon = 1e-6

        # prefactors
        a = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x =  torch.randn(*shape).to(dtype=dtype, device=cuda)
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        op = RealLayer_fun()

        # setup the model
        compute_loss = lambda a: 0.5 * torch.sum(op.apply(th_x*a)**2)
        th_loss = compute_loss(th_a)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon).cpu().numpy()
            l_an = compute_loss(th_a-epsilon).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

    def test_gradient1(self):
        self._test_gradient((2,5,2))
    def test_gradient2(self):
        self._test_gradient((2,5,10,10,2))

class TestImagLayer(unittest.TestCase):   
    def _test_gradient(self, shape):
        # setup the hyper parameters for each test
        dtype = torch.float64

        # perform a gradient check:
        epsilon = 1e-6

        # prefactors
        a = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x =  torch.randn(*shape).to(dtype=dtype, device=cuda)
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        op = ImagLayer_fun()

        # setup the model
        compute_loss = lambda a: 0.5 * torch.sum(op.apply(th_x*a)**2)
        th_loss = compute_loss(th_a)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon).cpu().numpy()
            l_an = compute_loss(th_a-epsilon).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

    def test_gradient1(self):
        self._test_gradient((2,5,2))
    def test_gradient2(self):
        self._test_gradient((2,5,10,10,2))

if __name__ == "__main__":
    unittest.test()
