
import torch

from .complex_regularizer import *

from .complex_conv3d import *
from .complex_foe2d import FoERegularizer
from optoth.activations import TrainableActivation
from mltoolsth.mytorch.complex import complex_angle, complex_abs, complex_normalization

import numpy as np
import unittest

__all__ = ['MagnitudeFoE3d',
           'ComplexFoE3d',
           'MagnitudeFoE2dt',
           'ComplexFoE2dt']

class ComplexFoE3d(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(ComplexFoE3d, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv3d(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1"])

        if not self.ckpt_state_dict is None:
            self.load_state_dict(self.ckpt_state_dict)

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

class MagnitudeFoE3d(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(MagnitudeFoE3d, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv3d(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1"])

        if not self.ckpt_state_dict is None:
            self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        magn = self.f1(complex_abs(x, eps=1e-12, keepdim=True)) / x.shape[1]
        norm = complex_normalization(x, eps=1e-12)
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


class ComplexFoE2dt(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(ComplexFoE2dt, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv2dt(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1"])

        if not self.ckpt_state_dict is None:
            self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        x_re = self.f1(x[...,0]) / x.shape[1]
        x_im = self.f1(x[...,1]) / x.shape[1]
        return torch.cat([x_re.unsqueeze_(-1), x_im.unsqueeze_(-1)], -1)

    def get_vis(self):
        kernels = {'K1.conv_sp.weight' : self.K1.conv_xy.weight[:,:,0],
                   'K1.conv_t.weight' : self.K1.conv_t.weight[...,0]
                   }

        x, fxre = self.f1.draw_complex(draw_type='real', scale=1)
        _, fxim = self.f1.draw_complex(draw_type='imag', scale=1)
        fx = torch.cat([fxre.unsqueeze_(-1), fxim.unsqueeze_(-1)], -1)

        return kernels, (x, fx)

class MagnitudeFoE2dt(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(MagnitudeFoE2dt, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv2dt(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1"])

        if not self.ckpt_state_dict is None:
            self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        magn = self.f1(complex_abs(x, eps=1e-12, keepdim=True)) / x.shape[1]
        norm = complex_normalization(x, eps=1e-12)
        fx = magn * norm
        return fx

    def get_vis(self):
        kernels = {'K1.conv_sp.weight' : self.K1.conv_xy.weight[:,:,0],
                   'K1.conv_t.weight' : self.K1.conv_t.weight[...,0]
                   }
        x, fxmagn = self.f1.draw_complex(draw_type='abs', scale=0, offset=1)
        x_im = torch.transpose(x, -2, -1)
        x_complex = torch.cat([torch.unsqueeze(x, -1), x_im.unsqueeze_(-1)], -1)
        norm = complex_normalization(x_complex.to(fxmagn.device), eps=1e-12)
        # fxmagn /= x.shape[1]
        fxmagn.unsqueeze_(-1)
        fx = fxmagn * norm
        return kernels, (x, fx)

class ComplexFoE3dTest(unittest.TestCase):
    def test_FoE3d_complex(self):
        nBatch = 2
        M = 100
        N = 100
        D = 10
        nf_in = 10
        nw = 31
        vabs = 0.75

        config = {
            'dtype': 'complex',
            'K1': {
                'in_channels': 1,
                'out_channels': nf_in,
                'kernel_size_sp_x': 5,
                'kernel_size_sp_y': 5,
                'kernel_size_t': 3,
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

        model = ComplexFoE3d(config).cuda()

        x = torch.randn(nBatch, 1, D, M, N, 2).cuda()
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)
        model.get_vis()

        model = MagnitudeFoE3d(config).cuda()
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)
        model.get_vis()
        
class ComplexFoE2dtTest(unittest.TestCase):
    def test_FoE2dt_complex(self):
        nBatch = 2
        M = 100
        N = 100
        D = 10
        nf_in = 10
        nw = 31
        vabs = 0.75

        config = {
            'dtype': 'complex',
            'K1': {
                'in_channels': 1,
                'out_channels': nf_in,
                'inter_channels': 20,
                'kernel_size_sp_x': 5,
                'kernel_size_sp_y': 5,
                'kernel_size_t': 3,
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

        model = ComplexFoE2dt(config).cuda()

        x = torch.randn(nBatch, 1, D, M, N, 2).cuda()
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)
        model.get_vis()

        model = MagnitudeFoE2dt(config).cuda()
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)
        model.get_vis()

if __name__ == "__main__":
    unittest.test()