
import torch

from .complex_regularizer import *

from .complex_conv2d import *
from optoth.activations import TrainableActivation

from mltoolsth.mytorch.complex import complex_angle, complex_abs, complex_normalization
from mltoolsth.torch_utils import *

import numpy as np
import unittest
from .foe2d import FoE2D

__all__ = ['MagnitudeFoE2D',
           'PolarFoE2D',
           'ComplexFoE2D',
           'Real2chFoE2D',
           'FoERegularizer']

class FoERegularizer(ComplexRegularizer):
    def __init__(self, config=None, file=None):
        super(FoERegularizer, self).__init__()

        # if (config is None and file is None) or \
        #     (not config is None and not file is None):
        #     raise RuntimeError('specify EITHER a config dictionary OR a `.pth`-file!')

        # if not file is None:
        #     if not file.endswith('.pth'):
        #         raise ValueError('file needs to end with `.pth`!')
        #     checkpoint = torch.load(file)
        #     self.config = checkpoint['config']
        #     self.ckpt_state_dict = checkpoint['model']
        #     self.tau = checkpoint['tau']
        # else:
        #     self.ckpt_state_dict = None
        #     self.tau = 1.0
        self.config = config

    def _transformation(self, x):
        return self.K1(x)

    def _transformation_T(self, grad_out):
        return self.K1.backward(grad_out)

    def grad(self, x):
        x = self._transformation(x)
        x = self._activation(x)
        return self._transformation_T(x)

    def get_vis(self):
        raise NotImplementedError

class PolarFoE2D(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(PolarFoE2D, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexPadConv2d(**self.config["K1"])
        self.f1_abs = TrainableActivation(**self.config["f1_abs"])
        self.f1_phi = TrainableActivation(**self.config["f1_phi"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        magn = self.f1_abs(complex_abs(x, eps=1e-6)) #/ x.shape[1]
        angle = self.f1_phi(complex_angle(x, eps=1e-6))

        re = magn * torch.cos(angle)
        im = magn * torch.sin(angle)

        fx = torch.cat([re.unsqueeze_(-1), im.unsqueeze_(-1)], -1)

        return fx

    def get_vis(self):
        kernels = {'K1.re' : self.K1.weight[:,0], 'K1.im' : self.K1.weight[:,1]}

        x, fxmagn = self.f1_abs.draw_complex(draw_type='abs', scale=0, offset=1)
        _, fxphase = self.f1_phi.draw_complex(draw_type='phase', scale=0, offset=1)

        fxmagn /= x.shape[1]
        fxmagn.unsqueeze_(-1)
        fxphase.unsqueeze_(-1)

        fx = torch.cat([fxmagn * torch.cos(fxphase), fxmagn * torch.sin(fxphase)], -1)

        return kernels, (x, fx)

class MagnitudeFoE2D(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(MagnitudeFoE2D, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexPadConv2d(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1_abs"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        magn = self.f1(complex_abs(x, eps=1e-6, keepdim=True)) / x.shape[1]
        norm = complex_normalization(x, eps=1e-6)
        fx = magn * norm
        return fx

    def get_vis(self):
        kernels = {'K1.weight.re' : self.K1.weight[:,0], 'K1.weight.im' : self.K1.weight[:,1]}
        x, fxmagn = self.f1.draw_complex(draw_type='abs', scale=0, offset=1)
        x_im = torch.transpose(x, -2, -1)
        x_complex = torch.cat([torch.unsqueeze(x, -1), x_im.unsqueeze_(-1)], -1)
        norm = complex_normalization(x_complex.to(fxmagn.device), eps=1e-12)
        # fxmagn /= x.shape[1]
        fxmagn.unsqueeze_(-1)
        fx = fxmagn * norm
        return kernels, (x, fx)

class ComplexFoE2D(FoERegularizer):
    """
    Fields of Experts regularizer used in the publication
    Effland, A. et al. "An optimal control approach to early stopping variational methods for image restoration". FoE 2019.
    """
    def __init__(self, config=None, file=None):
        super(ComplexFoE2D, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexPadConv2d(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        x_re = self.f1(x[...,0]) / x.shape[1]
        x_im = self.f1(x[...,1]) / x.shape[1]
        return torch.cat([x_re.unsqueeze_(-1), x_im.unsqueeze_(-1)], -1)

    def get_vis(self):
        kernels = {'K1.weight.re' : self.K1.weight[:,0], 'K1.weight.im' : self.K1.weight[:,1]}
        x, fxre = self.f1.draw_complex(draw_type='real', scale=1)
        _, fxim = self.f1.draw_complex(draw_type='imag', scale=1)
        fx = torch.cat([fxre.unsqueeze_(-1), fxim.unsqueeze_(-1)], -1)

        return kernels, (x, fx)

class Real2chFoE2D(FoE2D):
    def grad(self, x):
        xreal = complex2real(x)
        xreal = super().grad(xreal)
        return real2complex(xreal)

class PolarFoETest(unittest.TestCase):
    def test_FoE_polar(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nw = 31

        config = {
            'dtype': 'complex',
            'K1': {
                'in_channels': 1,
                'out_channels': nf_in,
                'kernel_size': 11,
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

        model = PolarFoE2D(config).cuda()

        x = torch.randn(nBatch, 1, M, N, 2).cuda()
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)
        #model.get_vis()

class MagnitudeFoETest(unittest.TestCase):
    def test_FoE_magnitude(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nw = 31

        config = {
            'dtype': 'complex',
            'K1': {
                'in_channels': 1,
                'out_channels': nf_in,
                'kernel_size': 11,
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

        model = MagnitudeFoE2D(config).cuda()

        x = torch.randn(nBatch, 1, M, N, 2).cuda()
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)
        #model.get_vis()

class ComplexFoETest(unittest.TestCase):
    def test_FoE_complex(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nw = 31
        vabs = 0.75

        config = {
            'dtype': 'complex',
            'K1': {
                'in_channels': 1,
                'out_channels': nf_in,
                'kernel_size': 11,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'num_channels': nf_in,
                'vmin': -vabs,
                'vmax':  vabs,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }

        model = ComplexFoE2D(config).cuda()

        x = torch.randn(nBatch, 1, M, N, 2).cuda()
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)
        #model.get_vis()

class PseudoComplexFoETest(unittest.TestCase):
    def test_FoE_pseudo_complex(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nw = 31
        vabs = 0.75

        config = {
            'dtype': 'complex',
            'K1': {
                'in_channels': 2,
                'out_channels': nf_in,
                'kernel_size': 11,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'num_channels': nf_in,
                'vmin': -vabs,
                'vmax':  vabs,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }

        model = Real2chFoE2D(config).cuda()

        x = torch.randn(nBatch, 1, M, N, 2).cuda()
        Kx = model(x)

        self.assertTrue(Kx.shape == x.shape)
        #model.get_vis() 

if __name__ == "__main__":
    unittest.test()