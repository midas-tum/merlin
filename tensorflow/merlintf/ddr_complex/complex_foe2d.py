import tensorflow as tf

from .regularizer import *
from .complex_padconv import ComplexPadConv2D
from .padconv import PadConv2D
from optotf.activations import TrainableActivationKeras as TrainableActivation
from merlintf.keras_utils import *
import unittest
import numpy as np

__all__ = ['MagnitudeFoE2D',
           'PolarFoE2D',
           'ComplexFoE2D',
           'FoERegularizer',
           'Real2chFoE2D']


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
        x = self._transformation_T(x)
        return x

class FoE2D(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(FoE2D, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = PadConv2D(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        return self.f1(x) / tf.cast(tf.shape(x)[-1], tf.float32)

class Real2chFoE2D(FoE2D):
    def grad(self, x):
        xreal = complex2real(x)
        xreal = super().grad(xreal)
        return real2complex(xreal)

class PolarFoE2D(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(PolarFoE2D, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexPadConv2D(**self.config["K1"])
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

class MagnitudeFoE2D(FoERegularizer):
    def __init__(self, config=None, file=None):
        super(MagnitudeFoE2D, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexPadConv2D(**self.config["K1"])
        self.f1_abs = TrainableActivation(**self.config["f1_abs"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        magn = self.f1_abs(complex_abs(x)) / tf.cast(tf.shape(x)[-1], tf.float32)
        xn = complex_norm(x)
        return complex_scale(xn, magn)

class ComplexFoE2D(FoERegularizer):
    """
    Fields of Experts regularizer used in the publication
    Effland, A. et al. "An optimal control approach to early stopping variational methods for image restoration". FoE 2019.
    """
    def __init__(self, config=None, file=None):
        super(ComplexFoE2D, self).__init__(config=config, file=file)

        # setup the modules
        self.K1 = ComplexPadConv2D(**self.config["K1"])
        self.f1 = TrainableActivation(**self.config["f1"])

        # if not self.ckpt_state_dict is None:
        #     self.load_state_dict(self.ckpt_state_dict)

    def _activation(self, x):
        nf = tf.cast(tf.shape(x)[-1], tf.float32)
        x_re = self.f1(tf.math.real(x)) / nf
        x_im = self.f1(tf.math.imag(x)) / nf
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
                'filters': nf_in,
                'kernel_size': 11,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1_abs': {
                'vmin': 0,
                'vmax': 2,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
            'f1_phi': {
                'vmin': -np.pi,
                'vmax':  np.pi,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 1,
            },
        }

        model = PolarFoE2D(config)

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
                'filters': nf_in,
                'kernel_size': 11,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1_abs': {
                'vmin': 0,
                'vmax': 2,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }

        model = MagnitudeFoE2D(config)

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
                'filters': nf_in,
                'kernel_size': 11,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'vmin': -vabs,
                'vmax':  vabs,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }

        model = ComplexFoE2D(config)

        x = tf.random.normal((nBatch, M, N, 1))
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

if __name__ == "__main__":
    unittest.test()