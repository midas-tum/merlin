import unittest
import torch
import numpy as np
import merlinth
from merlinth.layers.convolutional.complex_conv import (
    ComplexConv1d,
    ComplexConv2d,
    ComplexConv3d,
    ComplexConvTranspose1d,
    ComplexConvTranspose2d,
    ComplexConvTranspose3d
)
from merlinth.layers.convolutional.padconv import (
    PadConv1d,
    PadConv2d,
    PadConv3d,
    PadConvScale2d,
    PadConvScale3d,
    PadConvScaleTranspose2d,
    PadConvScaleTranspose3d,
)
from merlinth.layers.convolutional.complex_padconv import (
    ComplexPadConv1d,
    ComplexPadConv2d,
    ComplexPadConv3d,
    ComplexPadConvScale2d,
    ComplexPadConvScale3d,
    ComplexPadConvScaleTranspose2d,
    ComplexPadConvScaleTranspose3d,
    ComplexPadConvRealWeight1d,
    ComplexPadConvRealWeight2d,
    ComplexPadConvRealWeight3d,
)

# complex conv tests
class Test1D(unittest.TestCase):
    def _test(self, conv_fun):
        nBatch = 5
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = conv_fun(nf_in, nf_out, kernel_size=3).cuda()
        
        x = merlinth.random_normal_complex((nBatch, nf_in, N), dtype=torch.get_default_dtype()).cuda()
        Kx = model(x)
        
        self.assertTrue(True)

    def test_conv(self):
        self._test(ComplexConv1d)

    def test_convT(self):
        self._test(ComplexConvTranspose1d)

class Test2D(unittest.TestCase):
    def _test(self, conv_fun):
        nBatch = 5
        M = 320
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = conv_fun(nf_in, nf_out, kernel_size=3).cuda()
        
        x = merlinth.random_normal_complex((nBatch, nf_in, M, N), dtype=torch.get_default_dtype()).cuda()
        Kx = model(x)
        
        self.assertTrue(True)

    def test_conv(self):
        self._test(ComplexConv2d)

    def test_convT(self):
        self._test(ComplexConvTranspose2d)

class Test3D(unittest.TestCase):
    def _test(self, conv_fun):
        nBatch = 5
        D = 16
        M = 320
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = conv_fun(nf_in, nf_out, kernel_size=3).cuda()
        
        x = merlinth.random_normal_complex((nBatch, nf_in, D, M, N), dtype=torch.get_default_dtype()).cuda()
        Kx = model(x)
        
        self.assertTrue(True)

    def test_conv(self):
        self._test(ComplexConv3d)

    def test_convT(self):
        self._test(ComplexConvTranspose3d)

# padconv.py tests
class PadConv1dTest(unittest.TestCase):
    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = PadConv1d(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True).cuda()
        np_weight = model.weight.detach().cpu().numpy()
        reduction_dim = model.weight.reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))
        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test_grad(self, conv_fun, kernel_size, strides, dilation_rate, padding):
        nBatch = 5
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, nf_in, N]

        model = conv_fun(nf_in, nf_out, kernel_size=kernel_size, stride=strides, padding=padding, zero_mean=False, bound_norm=False)
        model.cuda()
        x = torch.randn(shape).cuda()
        x.requires_grad_(True)

        Kx = model(x)
        loss = 0.5 * torch.sum(Kx ** 2)
        loss.backward()
        grad_x = x.grad
        x_autograd = grad_x.cpu().numpy()

        KHKx = model.backward(Kx, x.shape)
        x_bwd = KHKx.detach().cpu().numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test1(self):
        self._test_grad(PadConv1d, 5, 1, 1, 'symmetric')

class PadConv2dTest(unittest.TestCase):
    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = PadConv2d(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True)
        np_weight = model.weight.detach().numpy()
        reduction_dim = model.weight.reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))
        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test_grad(self, conv_fun, kernel_size, strides, dilation_rate, padding):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, nf_in, M, N]

        model = conv_fun(nf_in, nf_out, kernel_size=kernel_size, stride=strides, padding=padding, zero_mean=False, bound_norm=False)
        model.cuda()
        x = torch.randn(shape).cuda()
        x.requires_grad_(True)
        Kx = model(x)
        loss = 0.5 * torch.sum(Kx ** 2)
        loss.backward()
        grad_x = x.grad
        x_autograd = grad_x.detach().cpu().numpy()

        KHKx = model.backward(Kx, x.shape)
        x_bwd = KHKx.detach().cpu().numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test1(self):
        self._test_grad(PadConv2d, 5, 1, 1, 'symmetric')

    def test2(self):
        self._test_grad(PadConv2d, [3, 5], 1, 1, 'symmetric')

    def test3(self):
        self._test_grad(PadConvScale2d, 3, 2, 1, 'symmetric')
        
class PadConvScaleTranspose2dTest(unittest.TestCase):
    def test_conv_transpose2d(self):
        nBatch = 5
        M = 320
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = PadConvScaleTranspose2d(nf_in, nf_out, kernel_size=3).cuda()
        
        x = torch.randn(nBatch, nf_in, M, N).cuda()
        Kx = model.backward(x)
        
        y = torch.randn(*Kx.shape).cuda()
        KHy = model.forward(y, output_shape=x.shape)

        rhs = torch.sum(Kx * y).detach().cpu().numpy()
        lhs = torch.sum(x * KHy).detach().cpu().numpy()

        self.assertTrue(rhs, lhs)

class PadConv3dTest(unittest.TestCase):
    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = PadConv3d(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True).cuda()
        np_weight = model.weight.detach().cpu().numpy()
        reduction_dim = model.weight.reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))

        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test_grad(self, conv_fun, kernel_size, strides, dilation_rate, padding):
        nBatch = 5
        M = 256
        N = 256
        D = 10
        nf_in = 2
        nf_out = 16
        shape = [nBatch, nf_in, D, M, N]

        model = conv_fun(nf_in, nf_out, kernel_size=kernel_size, stride=strides, padding=padding, zero_mean=False, bound_norm=False)
        model.double().cuda()
        x = torch.randn(shape).double().cuda()
        x.requires_grad_(True)

        Kx = model(x)
        loss = 0.5 * torch.sum(Kx ** 2)
        loss.backward()
        grad_x = x.grad
        x_autograd = grad_x.detach().cpu().numpy()

        KHKx = model.backward(Kx, x.shape)
        x_bwd = KHKx.detach().cpu().numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test1(self):
        self._test_grad(PadConv3d, 5, 1, 1, 'symmetric')

    def test2(self):
        self._test_grad(PadConv3d, [3, 5, 5], 1, 1, 'symmetric')

    def test3(self):
        self._test_grad(PadConvScale3d, 3, (1, 2, 2), 1, 'symmetric')

class PadConvScaleTranspose3dTest(unittest.TestCase):
    def test_conv_transpose3d(self):
        nBatch = 5
        M = 320
        N = 320
        D = 10
        nf_in = 1
        nf_out = 32
        
        model = PadConvScaleTranspose3d(nf_in, nf_out, kernel_size=3).cuda()
        
        x = torch.randn(nBatch, nf_in, D, M, N).cuda()
        Kx = model.backward(x)
        
        y = torch.randn(*Kx.shape).cuda()
        KHy = model.forward(y, output_shape=x.shape)

        rhs = torch.sum(Kx * y).detach().cpu().numpy()
        lhs = torch.sum(x * KHy).detach().cpu().numpy()

        self.assertTrue(rhs, lhs)

# complex_padconv.py
class ComplexPadConv1dTest(unittest.TestCase):
    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConv1d(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True).cuda()
        np_weight = model.weight.detach().cpu().numpy()
        reduction_dim = model.weight.reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))
        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test_grad(self, conv_fun, kernel_size, strides, dilation_rate, padding):
        nBatch = 2
        N = 12
        nf_in = 1
        nf_out = 1
        shape = [nBatch, nf_in, N]

        model = conv_fun(nf_in, nf_out, kernel_size=kernel_size, stride=strides, padding=padding, zero_mean=False, bound_norm=False)
        model.cuda()
        x = merlinth.random_normal_complex(shape, dtype=torch.get_default_dtype()).cuda()
        x.requires_grad_(True)
        Kx = model(x)
        loss = 0.5 * torch.sum(torch.conj(Kx) * Kx)
        loss.backward()
        grad_x = x.grad
        x_autograd = grad_x.cpu().numpy()

        KHKx = model.backward(Kx, x.shape)
        x_bwd = KHKx.detach().cpu().numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test1(self):
        self._test_grad(ComplexPadConv1d, 5, 1, 1, 'symmetric')

    def test2(self):
        self._test_grad(ComplexPadConvRealWeight1d, 5, 1, 1, 'symmetric')

class ComplexPadConv2dTest(unittest.TestCase):
    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConv2d(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True)
        np_weight = model.weight.detach().numpy()
        reduction_dim = model.weight.reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))
        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test_grad(self, conv_fun, kernel_size, strides, dilation_rate, padding):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, nf_in, M, N]

        model = conv_fun(nf_in, nf_out, kernel_size=kernel_size, stride=strides, padding=padding, zero_mean=False, bound_norm=False)
        model.cuda()
        x = merlinth.random_normal_complex(shape, dtype=torch.get_default_dtype()).cuda()
        x.requires_grad_(True)
        Kx = model(x)
        loss = 0.5 * torch.sum(torch.conj(Kx) * Kx)
        loss.backward()
        grad_x = x.grad
        x_autograd = grad_x.detach().cpu().numpy()

        KHKx = model.backward(Kx, x.shape)
        x_bwd = KHKx.detach().cpu().numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test1(self):
        self._test_grad(ComplexPadConv2d, 5, 1, 1, 'symmetric')

    def test2(self):
        self._test_grad(ComplexPadConv2d, [3, 5], 1, 1, 'symmetric')

    def test3(self):
        self._test_grad(ComplexPadConvScale2d, 3, 2, 1, 'symmetric')

    def test4(self):
        self._test_grad(ComplexPadConvRealWeight2d, 5, 1, 1, 'symmetric')

class ComplexPadConvScaleTranspose2dTest(unittest.TestCase):
    def test_conv_transpose2d(self):
        nBatch = 5
        M = 320
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConvScaleTranspose2d(nf_in, nf_out, kernel_size=3).cuda()
        
        x = merlinth.random_normal_complex((nBatch, nf_in, M, N), dtype=torch.get_default_dtype()).cuda()
        Kx = model.backward(x)
        
        y = merlinth.random_normal_complex(Kx.shape, dtype=torch.get_default_dtype()).cuda()
        KHy = model.forward(y, output_shape=x.shape)

        rhs = torch.sum(torch.conj(Kx)  * y).detach().cpu().numpy()
        lhs = torch.sum(torch.conj(x) * KHy).detach().cpu().numpy()

        self.assertTrue(rhs, lhs)

class ComplexPadConv3dTest(unittest.TestCase):
    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConv3d(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True).cuda()
        np_weight = model.weight.detach().cpu().numpy()
        reduction_dim = model.weight.reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))

        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test_grad(self, conv_fun, kernel_size, strides, dilation_rate, padding):
        nBatch = 5
        M = 256
        N = 256
        D = 10
        nf_in = 2
        nf_out = 16
        shape = [nBatch, nf_in, D, M, N]

        model = conv_fun(nf_in, nf_out, kernel_size=kernel_size, stride=strides, padding=padding, zero_mean=False, bound_norm=False)
        model.double().cuda()
        x = merlinth.random_normal_complex(shape, dtype=torch.double).cuda()
        x.requires_grad_(True)

        Kx = model(x)
        loss = 0.5 * torch.sum(torch.conj(Kx) * Kx)
        loss.backward()
        grad_x = x.grad
        x_autograd = grad_x.detach().cpu().numpy()

        KHKx = model.backward(Kx, x.shape)
        x_bwd = KHKx.detach().cpu().numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test1(self):
        self._test_grad(ComplexPadConv3d, 5, 1, 1, 'symmetric')

    def test2(self):
        self._test_grad(ComplexPadConv3d, [3, 5, 5], 1, 1, 'symmetric')

    def test3(self):
        self._test_grad(ComplexPadConvScale3d, 3, (1, 2, 2), 1, 'symmetric')

    def test4(self):
        self._test_grad(ComplexPadConvRealWeight3d, 5, 1, 1, 'symmetric')

class ComplexPadConvScaleTranspose3dTest(unittest.TestCase):
    def test_conv_transpose3d(self):
        nBatch = 5
        M = 320
        N = 320
        D = 10
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConvScaleTranspose3d(nf_in, nf_out, kernel_size=3).cuda()
        
        x = merlinth.random_normal_complex((nBatch, nf_in, D, M, N), dtype=torch.get_default_dtype()).cuda()
        Kx = model.backward(x)
        
        y = merlinth.random_normal_complex(Kx.shape, dtype=torch.get_default_dtype()).cuda()
        KHy = model.forward(y, output_shape=x.shape)

        rhs = torch.sum(torch.conj(Kx) * y).detach().cpu().numpy()
        lhs = torch.sum(torch.conj(x) * KHy).detach().cpu().numpy()

        self.assertTrue(rhs, lhs)

if __name__ == "__main__":
    unittest.main()
