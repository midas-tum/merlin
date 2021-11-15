import torch
from merlinth.layers import PadConv1d, PadConv2d, PadConv3d
from merlinth.layers import ComplexPadConv1d, ComplexPadConv2d, ComplexPadConv3d
from merlinth.layers import ComplexPadConv2Dt
from optoth.activations import TrainableActivation
import merlinth
import unittest
import numpy as np

__all__ = ['Regularizer',
           'MagnitudeFoE',
           'PolarFoE',
           'ComplexFoE',
           'FoE',
           'Real2chFoE'
          ]

class Regularizer(torch.nn.Module):
    """
    Basic regularization function
    """

    def __init__(self):
        super(Regularizer, self).__init__()

    def forward(self, x):
        return self.grad(x)

    def energy(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

    def get_theta(self):
        """
        return all parameters of the regularization
        """
        return self.named_parameters()

    def get_vis(self):
        raise NotImplementedError

class FoEBase(Regularizer):
    def __init__(self, config=None):
        super(FoEBase, self).__init__()
        self.config = config

    def _transformation(self, x):
        return self.K1(x)

    def _transformation_T(self, grad_out):
        return self.K1.backward(grad_out)

    def _activation(self):
        return NotImplementedError

    def grad(self, x):
        x = self._transformation(x)
        x = self._activation(x)
        x = self._transformation_T(x)
        return x

class FoE(FoEBase):
    def __init__(self, config=None):
        super().__init__(config=config)
        if config['dim'] == '2D':
                self.K1 = PadConv2d(**self.config["K1"])
        elif config['dim'] == '3D':
            self.K1 = PadConv3d(**self.config["K1"])
        else:
            raise RuntimeError(f"FoE regularizer not defined for {config['dim']}!")

        self.f1 = TrainableActivation(**self.config["f1"])

    def _activation(self, x):
        return self.f1(x) / x.shape[1]

class Real2chFoE(FoE):
    def grad(self, x):
        xreal = merlinth.complex2real(x)
        xreal = super().grad(xreal)
        return merlinth.real2complex(xreal)

class PolarFoE(FoEBase):
    def __init__(self, config=None):
        super(PolarFoE, self).__init__(config=config)

        # setup the modules
        if config['dim'] == '2D':
            self.K1 = ComplexPadConv2d(**self.config["K1"])
        elif config['dim'] == '3D':
            self.K1 = ComplexPadConv3d(**self.config['K1'])
        elif config['dim'] == '2Dt':
            self.K1 = ComplexPadConv2Dt(**self.config["K1"])
        else:
            raise RuntimeError(f"FoE regularizer not defined for {config['dim']}!")
        
        self.f1_abs = TrainableActivation(**self.config["f1_abs"])
        self.f1_phi = TrainableActivation(**self.config["f1_phi"])

    def _activation(self, x):
        magn = self.f1_abs(merlinth.complex_abs(x, eps=1e-6)) #/ x.shape[1]
        angle = self.f1_phi(merlinth.complex_angle(x, eps=1e-6))

        re = magn * torch.cos(angle)
        im = magn * torch.sin(angle)

        return torch.complex(re, im)

    def get_vis(self):
        kernels = {'K1.re' : torch.real(self.K1.weight), 'K1.im' : torch.imag(self.K1.weight)}

        x, fxmagn = self.f1_abs.draw_complex(draw_type='abs', scale=0, offset=1)
        _, fxphase = self.f1_phi.draw_complex(draw_type='phase', scale=0, offset=1)

        fxmagn /= x.shape[1]

        fx = torch.complex(fxmagn * torch.cos(fxphase), fxmagn * torch.sin(fxphase))

        return kernels, (x, fx)

class MagnitudeFoE(FoEBase):
    def __init__(self, config=None):
        super(MagnitudeFoE, self).__init__(config=config)

        # setup the modules
        if config['dim'] == '2D':
            self.K1 = ComplexPadConv2d(**self.config["K1"])
        elif config['dim'] == '3D':
            self.K1 = ComplexPadConv3d(**self.config['K1'])
        elif config['dim'] == '2Dt':
            self.K1 = ComplexPadConv2Dt(**self.config["K1"])
        else:
            raise RuntimeError(f"FoE regularizer not defined for {config['dim']}!")

        self.f1_abs = TrainableActivation(**self.config["f1_abs"])

    def _activation(self, x):
        magn = self.f1_abs(merlinth.complex_abs(x, eps=1e-6)) / x.shape[1]
        norm = merlinth.complex_norm(x, eps=1e-6)
        fx = magn * norm
        return fx

    def get_vis(self):
        kernels = {'K1.weight.re' : torch.real(self.K1.weight), 'K1.weight.im' : torch.imag(self.K1.weight)}
        x, fxmagn = self.f1.draw_complex(draw_type='abs', scale=0, offset=1)
        x_im = torch.transpose(x, -2, -1)
        x_complex = torch.complex(x, x_im)
        norm = merlinth.complex_norm(x_complex.to(fxmagn.device), eps=1e-12)
        # fxmagn /= x.shape[1]
        fx = fxmagn * norm
        return kernels, (x, fx)

class ComplexFoE(FoEBase):
    def __init__(self, config=None):
        super(ComplexFoE, self).__init__(config=config)

        # setup the modules
        if config['dim'] == '2D':
            self.K1 = ComplexPadConv2d(**self.config["K1"])
        elif config['dim'] == '3D':
            self.K1 = ComplexPadConv3d(**self.config['K1'])
        elif config['dim'] == '2Dt':
            self.K1 = ComplexPadConv2Dt(**self.config["K1"])
        else:
            raise RuntimeError(f"FoE regularizer not defined for {config['dim']}!")

        self.f1 = TrainableActivation(**self.config["f1"])

    def _activation(self, x):
        x_re = self.f1(torch.real(x)) / x.shape[1]
        x_im = self.f1(torch.imag(x)) / x.shape[1]
        return torch.complex(x_re, x_im)

    def get_vis(self):
        kernels = {'K1.weight.re' : torch.real(self.K1.weight), 'K1.weight.im' : torch.imag(self.K1.weight)}
        x, fxre = self.f1.draw_complex(draw_type='real', scale=1)
        _, fxim = self.f1.draw_complex(draw_type='imag', scale=1)
        fx = torch.complex(fxre, fxim)

        return kernels, (x, fx)

class PolarFoETest(unittest.TestCase):
    def test_FoE_polar_2d(self):
        self._test_FoE_polar('2D', 11)

    def test_FoE_polar_3d(self):
        self._test_FoE_polar('3D', (3, 5, 5))

    def test_FoE_polar_2Dt(self):
        self._test_FoE_polar('2Dt', (5, 7, 7))

    def _test_FoE_polar(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        config = {
            'dim': dim,
            'K1': {
                'in_channels' : 1,
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1_abs': {
                'num_channels': nf_in,
                'vmin': 0,
                'vmax': 2,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
            'f1_phi': {
                'num_channels': nf_in,
                'vmin': -np.pi,
                'vmax':  np.pi,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 1,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters' : nf_in})

        model = PolarFoE(config).cuda()

        if dim == '2D':
            x = merlinth.random_normal_complex((nBatch, 1, M, N)).cuda()
        elif dim == '3D' or dim == '2Dt':
            x =  merlinth.random_normal_complex((nBatch, 1, D, M, N)).cuda()
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class MagnitudeFoETest(unittest.TestCase):
    def test_FoE_magnitude_2d(self):
        self._test_FoE_magnitude('2D', 11)

    def test_FoE_magnitude_3d(self):
        self._test_FoE_magnitude('3D', (3, 5, 5))

    def test_FoE_magnitude_2Dt(self):
        self._test_FoE_magnitude('2Dt', (5, 7, 7))

    def _test_FoE_magnitude(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        config = {
            'dim': dim,
            'K1': {
                'in_channels' : 1,
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1_abs': {
                'num_channels': nf_in,
                'vmin': 0,
                'vmax': 2,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters' : nf_in})

        model = MagnitudeFoE(config).cuda()

        if dim == '2D':
            x = merlinth.random_normal_complex((nBatch, 1, M, N)).cuda()
        elif dim == '3D' or dim == '2Dt':
            x = merlinth.random_normal_complex((nBatch, 1, D, M, N)).cuda()
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')        
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class ComplexFoETest(unittest.TestCase):
    def test_FoE_complex_2d(self):
        self._test_FoE_complex('2D', 11)

    def test_FoE_complex_3d(self):
        self._test_FoE_complex('3D', (3, 5, 5))

    def test_FoE_complex_2Dt(self):
        self._test_FoE_complex('2Dt', (5, 7, 7))

    def _test_FoE_complex(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        config = {
            'dim': dim,
            'K1': {
                'in_channels' : 1,
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'num_channels': nf_in,
                'vmin': -1,
                'vmax':  1,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters' : nf_in})

        model = ComplexFoE(config).cuda()

        if dim == '2D':
            x = merlinth.random_normal_complex((nBatch, 1, M, N)).cuda()
        elif dim == '3D' or dim == '2Dt':
            x = merlinth.random_normal_complex((nBatch, 1, D, M, N)).cuda()
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')

        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class Real2chFoETest(unittest.TestCase):
    def test_FoE_real2ch_2d(self):
        self._test_FoE_real2ch('2D', 11)

    def test_FoE_real2ch_3d(self):
        self._test_FoE_real2ch('3D', (3, 5, 5))

    def _test_FoE_real2ch(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        config = {
            'dim': dim,
            'K1': {
                'in_channels' : 2,
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'num_channels' : nf_in,
                'vmin': -1,
                'vmax':  1,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters' : nf_in})

        model = Real2chFoE(config).cuda()

        if dim == '2D':
            x = merlinth.random_normal_complex((nBatch, 1, M, N)).cuda()
        elif dim == '3D' or dim == '2Dt':
            x = merlinth.random_normal_complex((nBatch, 1, D, M, N)).cuda()
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class RealFoETest(unittest.TestCase):
    def test_FoE_real_2d(self):
        self._test_FoE_real('2D', 11)

    def test_FoE_real_3d(self):
        self._test_FoE_real('3D', (3, 5, 5))

    @unittest.expectedFailure
    def test_FoE_real_2dt(self):
        self._test_FoE_real('2Dt', (3, 5, 5))

    def _test_FoE_real(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        config = {
            'dim': dim,
            'K1': {
                'in_channels' : 1,
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'num_channels': nf_in,
                'vmin': -1,
                'vmax':  1,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters' : nf_in})

        model = FoE(config).cuda()

        if dim == '2D':
            x = torch.randn(nBatch, 1, M, N).cuda()
        elif dim == '3D' or dim == '2Dt':
            x = torch.randn(nBatch, 1, D, M, N).cuda()
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

if __name__ == "__main__":
    unittest.test()