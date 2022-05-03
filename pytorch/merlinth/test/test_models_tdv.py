import unittest
import torch
import numpy as np
from merlinth.models.tdv import TDV, ComplexMicroBlock, MacroBlock

class GradientTest(unittest.TestCase):
    def _test_tdv_gradient(self, dim):
        # setup the data
        if dim == '2D':
            shape = (2,1,64,64)
        elif dim == '3D':
            shape = (2,1,10,64,64)
        else:
            raise ValueError
        x = np.random.rand(*shape)
        x = torch.from_numpy(x).cuda()

        # define the TDV regularizer
        config ={
            'dim': dim,
            'is_complex': False,
            'in_channels': 1,
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 3,
            'num_mb': 2,
            'multiplier': 2,
        }
        R = TDV(config).double().cuda()

        def compute_loss(scale):
            return torch.sum(R.energy(scale*x))
        
        scale = 1.
        
        # compute the gradient using the implementation
        grad_scale = torch.sum(x*R.grad(scale*x)).item()

        # check it numerically
        epsilon = 1e-4
        with torch.no_grad():
            l_p = compute_loss(scale+epsilon).item()
            l_n = compute_loss(scale-epsilon).item()
            grad_scale_num = (l_p - l_n) / (2 * epsilon)

        condition = np.abs(grad_scale - grad_scale_num) < 1e-3
        print(f'grad_scale: {grad_scale:.7f} num_grad_scale {grad_scale_num:.7f} success: {condition}')
        self.assertTrue(condition)

    def test_tdv_gradient_2D(self):
        self._test_tdv_gradient('2D')

    @unittest.expectedFailure
    def test_tdv_gradient_3D(self):
        self._test_tdv_gradient('3D')

class GradientTesComplexGradientTest(unittest.TestCase):
    def _test_tdv_gradient(self, dim):
        # setup the data
        if dim == '2D':
            shape = (2,1,64,64)
        elif dim == '3D':
            shape = (2,1,10,64,64)
        else:
            raise ValueError
        
        x = np.random.rand(*shape) + 1j*np.random.rand(*shape)
        x = torch.from_numpy(x).cuda()
        x.requires_grad_(True)

        # define the TDV regularizer
        config ={
            'dim': dim,
            'is_complex': True,
            'in_channels': 1,
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 3,
            'num_mb': 2,
            'multiplier': 1,
        }
        R = TDV(config).cuda().double()
                
        loss = 0.5 * torch.sum(R.energy(x))
        loss.backward()
        x_autograd = x.grad.data.cpu().numpy()
        KHKx = R.grad(x)
        x_bwd = KHKx.data.cpu().numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    #@unittest.expectedFailure
    def test_tdv_gradient_2D_complex(self):
        self._test_tdv_gradient('2D')

    @unittest.expectedFailure
    def test_tdv_gradient_3D_complex(self):
        self._test_tdv_gradient('3D')

class TestComplexMicroBlock(unittest.TestCase):
    def _test_gradient(self, dim):
        # setup the data
        nf = 32
        if dim == '2D':
            shape = (2,nf,64,64)
        elif dim == '3D':
            shape = (2,nf,10,64,64)
        else:
            raise ValueError
        x = np.random.rand(*shape) + 1j*np.random.rand(*shape)
        x = torch.from_numpy(x).cuda()
        x.requires_grad_(True)
        
        R = ComplexMicroBlock(dim, nf).cuda().double()
        Kx = R(x)
        loss = 0.5 * torch.sum(torch.conj(Kx) * Kx)
        loss.backward()
        x_autograd = x.grad.detach().cpu().numpy()
        
        KHKx = R.backward(Kx)
        x_bwd = KHKx.detach().cpu().numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_2D(self):
        self._test_gradient('2D')
        
    def test_3D(self):
        self._test_gradient('3D')

class TestComplexMacroBlock(unittest.TestCase):
    def _test_gradient(self, dim):
        # setup the data
        nf = 32
        if dim == '2D':
            shape = (2,nf,64,64)
        elif dim == '3D':
            shape = (2,nf,10,64,64)
        else:
            raise ValueError
        
        x = np.random.rand(*shape) + 1j*np.random.rand(*shape)
        x = torch.from_numpy(x).cuda()
        x.requires_grad_(True)

        R = MacroBlock(dim, nf, num_scales=1, is_complex=True).cuda().double()
        Kx = R([x])[0]
        loss = 0.5 * torch.sum(torch.conj(Kx) * Kx)
        loss.backward()
        x_autograd = x.grad.detach().cpu().numpy()

        KHKx = R.backward([Kx])
        x_bwd = KHKx[0].detach().cpu().numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)
    
    def test_2D(self):
        self._test_gradient('2D')

    def test_3D(self):
        self._test_gradient('3D')

class TestEnergy(unittest.TestCase):
    def _test_gradient(self, dim):
        # setup the data
        if dim == '2D':
            shape = (1,1,1,1)
        elif dim == '3D':
            shape = (1,1,1,1,1)
        else:
            raise ValueError

        x = np.random.rand(*shape) + 1j*np.random.rand(*shape)
        x = torch.from_numpy(x).cuda()
        x.requires_grad_(True)
        
        # define the TDV regularizer
        config ={
            'in_channels': 1,
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 2,
            'num_mb': 1,
            'multiplier': 2,
            'dim': dim,
            'is_complex': True,
        }
        R = TDV(config)
        
        Kx = R._potential(x)
        loss = 0.5 * sum(Kx)
        loss.backward()
        
        x_autograd = x.grad.detach().cpu().numpy()

        KHKx = R._activation(x)
        x_bwd = KHKx.detach().cpu().numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_2D(self):
        self._test_gradient('2D')

    def test_3D(self):
        self._test_gradient('3D')

class TestTransformation(unittest.TestCase):
    def _test_gradient(self, dim, is_complex):
        # setup the data
        if dim == '2D':
            shape = (2,1,64,64)
        elif dim == '3D':
            shape = (2,1,10,64,64)
        else:
            raise ValueError
        if is_complex:
            x = np.random.rand(*shape) + 1j*np.random.rand(*shape)
        else:
            x = np.random.rand(*shape)
        x = torch.from_numpy(x).cuda()
        x.requires_grad_(True)

        # define the TDV regularizer
        config ={
            'dim': dim,
            'is_complex': is_complex,
            'in_channels': 1,
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 2,
            'num_mb': 1,
            'multiplier': 2,
        }
        R = TDV(config).cuda().double()

        Kx = R._transformation(x)
        if is_complex:
            loss = 0.5 * torch.sum(torch.conj(Kx) * Kx)
        else:
            loss = 0.5 * torch.sum(Kx * Kx)
        
        loss.backward()
        x_autograd = x.grad.detach().cpu().numpy()

        KHKx = R._transformation_T(Kx)
        x_bwd = KHKx.detach().cpu().numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_2D(self):
        self._test_gradient('2D', False)

    def test_3D(self):
        self._test_gradient('3D', False)

    #@unittest.expectedFailure
    def test_2D_complex(self):
        self._test_gradient('2D', True)

    #@unittest.expectedFailure
    def test_3D_complex(self):
        self._test_gradient('3D', True)

if __name__ == "__main__":
    unittest.main()