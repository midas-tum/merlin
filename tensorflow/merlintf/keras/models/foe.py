import tensorflow as tf
import merlintf
from merlintf.keras.layers import ComplexPadConv2D, ComplexPadConv3D
from merlintf.keras.layers import ComplexPadConv2Dt
from merlintf.keras.layers import PadConv1D, PadConv2D, PadConv3D
from optotf.activations import TrainableActivationKeras as TrainableActivation
from tensorflow.python.eager import context

__all__ = ['Regularizer',
           'MagnitudeFoE',
           'PolarFoE',
           'ComplexFoE',
           'FoE',
           'Real2chFoE']

class Regularizer(tf.keras.Model):
    """
    Basic regularization function
    """

    def __init__(self):
        super(Regularizer, self).__init__()

    def call(self, x):
        return self.grad(x)

    def energy(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

class FoEBase(Regularizer):
    def __init__(self, config=None):
        super(FoEBase, self).__init__()
        self.config = config

    def _transformation(self, x):
        Kx = self.K1(x)
        return Kx

    def _transformation_T(self, grad_out):
        return self.K1.backward(grad_out)

    def _activation(self):
        return NotImplementedError

    def grad(self, x):
        input_shape = x.shape
        x = self._transformation(x)
        x = self._activation(x)
        x = self._transformation_T(x)

        if not context.executing_eagerly():
            # Infer the static output shape:
            x.set_shape(input_shape)
        return x

class FoE(FoEBase):
    def __init__(self, config=None):
        super().__init__(config=config)
        if config['dim'] == '1D':
            self.K1 = PadConv1D(**self.config["K1"])
        elif config['dim'] == '2D':
            self.K1 = PadConv2D(**self.config["K1"])
        elif config['dim'] == '3D':
            self.K1 = PadConv3D(**self.config["K1"])
        else:
            raise RuntimeError(f"FoE regularizer not defined for {config['dim']}!")

        self.f1 = TrainableActivation(**self.config["f1"])

    def _activation(self, x):
        return self.f1(x) / tf.cast(tf.shape(x)[-1], x.dtype)

class Real2chFoE(FoE):
    def grad(self, x):
        xreal = merlintf.complex2real(x)
        xreal = super().grad(xreal)
        return merlintf.real2complex(xreal)

class PolarFoE(FoEBase):
    def __init__(self, config=None):
        super(PolarFoE, self).__init__(config=config)

        # setup the modules
        if config['dim'] == '2D':
            self.K1 = ComplexPadConv2D(**self.config["K1"])
        elif config['dim'] == '3D':
            self.K1 = ComplexPadConv3D(**self.config['K1'])
        elif config['dim'] == '2Dt':
            self.K1 = ComplexPadConv2Dt(**self.config["K1"])
        else:
            raise RuntimeError(f"FoE regularizer not defined for {config['dim']}!")
        
        self.f1_abs = TrainableActivation(**self.config["f1_abs"])
        self.f1_phi = TrainableActivation(**self.config["f1_phi"])

    def _activation(self, x):
        magn = self.f1_abs(merlintf.complex_abs(x)) #/ x.shape[-1]
        angle = self.f1_phi(merlintf.complex_angle(x))

        re = magn * tf.math.cos(angle)
        im = magn * tf.math.sin(angle)

        return tf.complex(re, im)

class MagnitudeFoE(FoEBase):
    def __init__(self, config=None):
        super(MagnitudeFoE, self).__init__(config=config)

        # setup the modules
        if config['dim'] == '2D':
            self.K1 = ComplexPadConv2D(**self.config["K1"])
        elif config['dim'] == '3D':
            self.K1 = ComplexPadConv3D(**self.config['K1'])
        elif config['dim'] == '2Dt':
            self.K1 = ComplexPadConv2Dt(**self.config["K1"])
        else:
            raise RuntimeError(f"FoE regularizer not defined for {config['dim']}!")

        self.f1_abs = TrainableActivation(**self.config["f1_abs"])

    def _activation(self, x):
        magn = self.f1_abs(merlintf.complex_abs(x))
        magn /= tf.cast(tf.shape(x)[-1], magn.dtype)
        xn = merlintf.complex_norm(x)
        return merlintf.complex_scale(xn, magn)

class ComplexFoE(FoEBase):
    def __init__(self, config=None):
        super(ComplexFoE, self).__init__(config=config)

        # setup the modules
        if config['dim'] == '2D':
            self.K1 = ComplexPadConv2D(**self.config["K1"])
        elif config['dim'] == '3D':
            self.K1 = ComplexPadConv3D(**self.config['K1'])
        elif config['dim'] == '2Dt':
            self.K1 = ComplexPadConv2Dt(**self.config["K1"])
        else:
            raise RuntimeError(f"FoE regularizer not defined for {config['dim']}!")

        self.f1 = TrainableActivation(**self.config["f1"])

    def _activation(self, x):
        x_re = self.f1(tf.math.real(x))
        x_im = self.f1(tf.math.imag(x))
        nf = tf.cast(tf.shape(x)[-1], x_re.dtype)
        return tf.complex(x_re / nf, x_im  / nf)

