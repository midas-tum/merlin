import tensorflow as tf

from regularizer import *
from complex_conv2d import *
from optotf.activations import TrainableActivation
from complex_layer import *
import unittest
import numpy as np

__all__ = ['MagnitudeFoE2d',
           'PolarFoE2d',
           'ComplexFoE2d',
           'FoERegularizer']


class FoERegularizer(Regularizer):
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

class PolarFoE2d(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(PolarFoE2d, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv2d(**self.config["K1"])
        self.f1_abs = TrainableActivation(**self.config["f1_abs"])
        self.f1_phi = TrainableActivation(**self.config["f1_phi"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        magn = self.f1_abs(complex_abs(x)) #/ x.shape[-1]
        angle = self.f1_phi(complex_angle(x))

        re = magn * tf.math.cos(angle)
        im = magn * tf.math.sin(angle)

        return tf.complex(re, im)

class MagnitudeFoE2d(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(MagnitudeFoE2d, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv2d(**self.config["K1"])
        self.f1_abs = TrainableActivation(**self.config["f1_abs"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        magn = tf.cast(self.f1_abs(complex_abs(x)), tf.complex64) / x.shape[-1]
        return magn * complex_norm(x)

class ComplexFoE2d(FoERegularizer):
    """
    Fields of Experts regularizer used in the publication
    Effland, A. et al. "An optimal control approach to early stopping variational methods for image restoration". FoE 2019.
    """
    def __init__(self, config=None, file=None):
        super(ComplexFoE2d, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv2d(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        x_re = self.f1(tf.math.real(x)) / x.shape[1]
        x_im = self.f1(tf.math.imag(x)) / x.shape[1]
        return tf.complex(x_re, x_im)

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

        model = PolarFoE2d(config)

        x = tf.random.normal((nBatch, M, N, 1))
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

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

        model = MagnitudeFoE2d(config)

        x = tf.random.normal((nBatch, M, N, 1))
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

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

        model = ComplexFoE2d(config)

        x = tf.random.normal((nBatch, M, N, 1))
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

if __name__ == "__main__":
    unittest.test()