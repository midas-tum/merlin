import tensorflow as tf

from regularizer import *
from complex_conv3d import *
from complex_foe2d import FoERegularizer
from optotf.activations import TrainableActivation
from complex_layer import *
import unittest
import numpy as np

__all__ = ['MagnitudeFoE3d',
           'ComplexFoE3d',
        #    'MagnitudeFoE2dt',
        #    'ComplexFoE2dt',
           ]

class MagnitudeFoE3d(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(MagnitudeFoE3d, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv3d(**self.config["K1"])
        self.f1_abs = TrainableActivation(**self.config["f1_abs"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        magn = tf.cast(self.f1_abs(complex_abs(x)), tf.complex64) / x.shape[-1]
        return magn * complex_norm(x)

class ComplexFoE3d(FoERegularizer):
    """
    Fields of Experts regularizer used in the publication
    Effland, A. et al. "An optimal control approach to early stopping variational methods for image restoration". FoE 2019.
    """
    def __init__(self, config=None, file=None):
        super(ComplexFoE3d, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv3d(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        x_re = self.f1(tf.math.real(x)) / x.shape[1]
        x_im = self.f1(tf.math.imag(x)) / x.shape[1]
        return tf.complex(x_re, x_im)

class MagnitudeFoE2dt(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(MagnitudeFoE2dt, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv2dt(**self.config["K1"])
        self.f1_abs = TrainableActivation(**self.config["f1_abs"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        magn = tf.cast(self.f1_abs(complex_abs(x)), tf.complex64) / x.shape[-1]
        return magn * complex_norm(x)

class ComplexFoE2dt(FoERegularizer):
    """
    Fields of Experts regularizer used in the publication
    Effland, A. et al. "An optimal control approach to early stopping variational methods for image restoration". FoE 2019.
    """
    def __init__(self, config=None, file=None):
        super(ComplexFoE2dt, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexConv2dt(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        x_re = self.f1(tf.math.real(x)) / x.shape[1]
        x_im = self.f1(tf.math.imag(x)) / x.shape[1]
        return tf.complex(x_re, x_im)

class MagnitudeFoETest(unittest.TestCase):
    def test_FoE_magnitude(self):
        nBatch = 5
        M = 256
        N = 256
        D = 10
        nf_in = 10
        nw = 31

        config = {
            'dtype': 'complex',
            'K1': {
                'in_channels': 1,
                'out_channels': nf_in,
                'kernel_size': (3, 5, 5),
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

        model = MagnitudeFoE3d(config)

        x = tf.random.normal((nBatch, D, M, N, 1))
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class ComplexFoETest(unittest.TestCase):
    def test_FoE_complex(self):
        nBatch = 5
        M = 256
        N = 256
        D = 10
        nf_in = 10
        nw = 31
        vabs = 0.75

        config = {
            'dtype': 'complex',
            'K1': {
                'in_channels': 1,
                'out_channels': nf_in,
                'kernel_size': (3,5,5),
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

        model = ComplexFoE3d(config)

        x = tf.random.normal((nBatch, D, M, N, 1))
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)


class MagnitudeFoE2dtTest(unittest.TestCase):
    def test_FoE_magnitude(self):
        nBatch = 5
        M = 256
        N = 256
        D = 10
        nf_in = 2
        nf_out = 10
        ksz = (3,5,5)
        nf_inter = np.ceil((nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] + nf_out * ksz[0])).astype(np.int32)
        nw = 31

        config = {
            'dtype': 'complex',
            'K1': {
                'in_channels': nf_in,
                'out_channels': nf_out,
                'inter_channels': nf_inter,
                'kernel_size': ksz,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1_abs': {
                'num_channels': nf_out,
                'vmin': 0,
                'vmax': 2,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }

        model = MagnitudeFoE2dt(config)

        x = tf.random.normal((nBatch, D, M, N, nf_in))
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class ComplexFoE2dtTest(unittest.TestCase):
    def test_FoE_complex(self):
        nBatch = 5
        M = 256
        N = 256
        D = 10
        nf_in = 2
        nf_out = 10
        ksz = (3,5,5)
        nf_inter = np.ceil((nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] + nf_out * ksz[0])).astype(np.int32)
        nw = 31

        config = {
            'dtype': 'complex',
            'K1': {
                'in_channels': nf_in,
                'out_channels': nf_out,
                'inter_channels': nf_inter,
                'kernel_size': ksz,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'num_channels': nf_out,
                'vmin': -1,
                'vmax': 1,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }

        model = ComplexFoE2dt(config)

        x = tf.random.normal((nBatch, D, M, N, nf_in))
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)
if __name__ == "__main__":
    unittest.test()