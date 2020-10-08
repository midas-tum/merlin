import tensorflow as tf
import unittest
from mltoolstf.keras_utils.complex import *
import numpy as np

__all__ = ['cReLU',
           'ModReLU',
           'cPReLU',
           'ModPReLU',
           'cStudentT',
           'ModStudentT',
           'cStudentT2',
           'ModStudentT2'
         ]

def get(identifier):
    if identifier is None:
        return Identity()
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError(
            'Could not interpret activation function identifier: {}'.format(
                identifier))

def deserialize(act):
    if act == 'ModReLU':
        return ModReLU()
    elif act == 'cPReLU':
        return cPReLU()
    elif act == 'cReLU':
        return cReLU()
    elif act == 'ModPReLU':
        return ModPReLU()
    elif act == 'hard_sigmoid':
        return HardSigmoid()
    elif act == 'cardioid':
        return Cardioid()
    elif act is None or act == 'identity':
        return Identity()
    else:
        raise ValueError(f"Selected activation '{act}' not implemented in complex activations")

def serialize(act):
    return act.__name__


class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, bias=0.1, trainable=True):
        super().__init__()
        self.bias_init = bias
        self.trainable = trainable

    @property
    def __name__(self):
        return 'hard_sigmoid'

    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.Constant(self.bias_init)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      )
    def call(self, z):
        return tf.cast(tf.keras.activations.hard_sigmoid(complex_abs(z) + self.bias), tf.complex64)
    


class cReLU(tf.keras.layers.Layer):
    def call(self, z):
        actre = tf.keras.activations.relu(tf.math.real(z))
        actim = tf.keras.activations.relu(tf.math.imag(z))
        return tf.complex(actre, actim)
    @property
    def __name__(self):
        return 'cReLU'
        
class Identity(tf.keras.layers.Layer):    
    @property
    def __name__(self):
        return 'identity'

    def call(self, z):
        return z

class ModReLU(tf.keras.layers.Layer):
    def __init__(self, bias=0.0, trainable=True):
        super().__init__()
        self.bias_init = bias
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.Constant(self.bias_init)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      )
    def call(self, z):
        return tf.cast(tf.keras.activations.relu(complex_abs(z) + self.bias), tf.complex64) * complex_norm(z)

    def __str__(self):
        s = f"ModReLU: bias_init={self.bias_init}, trainable={self.trainable}"
        return s

class cPReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.1, trainable=False):
        super().__init__()
        self.alpha_init = alpha
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.Constant(self.alpha_init)
        self.alpha_real = self.add_weight('alpha_real',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      )
        # self.alpha_imag = self.add_weight('alpha_imag',
        #                               shape=(input_shape[-1]),
        #                               initializer=initializer,
        #                               )

    def call(self, z):
        zre = tf.math.real(z)
        zim = tf.math.imag(z)
        actre = tf.maximum(0.0, zre) + self.alpha_real * tf.minimum(0.0, zre)
        actim = tf.maximum(0.0, zim) + self.alpha_real * tf.minimum(0.0, zim)

        return tf.complex(actre, actim)

    def __str__(self):
        s = f"cPReLU: alpha_init={self.alpha_init}, trainable={self.trainable}"
        return s

    @property
    def __name__(self):
        return 'cPReLU'
class ModPReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.1, bias=0, trainable=False):
        super().__init__()
        self.alpha_init = alpha
        self.bias_init = bias
        self.trainable = trainable

    @property
    def __name__(self):
        return 'ModPReLU'

    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.Constant(self.bias_init)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      )
        initializer_alpha = tf.keras.initializers.Constant(self.alpha_init)
        self.alpha = self.add_weight('alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      )
    def call(self, z):
        act = tf.maximum(0.0, complex_abs(z) + self.bias) + self.alpha * tf.minimum(0.0, complex_abs(z) + self.bias)
        return tf.cast(act, tf.complex64) * complex_norm(z)

    def __str__(self):
        s = f"ModPReLU: alpha_init={self.alpha_init}, bias_init={self.bias_init}, trainable={self.trainable}"
        return s

class Cardioid(tf.keras.layers.Layer):
    def __init__(self, bias=2.0, trainable=True):
        super().__init__()
        self.bias_init = bias
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer_bias = tf.keras.initializers.Constant(self.bias_init)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_bias,
                                      trainable=self.trainable,
                                      )
    def call(self, z):
        phase = complex_angle(z)
        cos = tf.cast(tf.math.cos(phase), tf.complex64) 

        return 0.5 * (1 + cos) * z
        
    def __str__(self):
        s = f"Cardioid: bias_init={self.bias_init}, trainable={self.trainable}"
        return s

    @property
    def __name__(self):
        return 'cardioid'

class Cardioid2(tf.keras.layers.Layer):
    def __init__(self, bias=2.0, trainable=True):
        super().__init__()
        self.bias_init = bias
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer_bias = tf.keras.initializers.Constant(self.bias_init)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_bias,
                                      trainable=self.trainable,
                                      )
    def call(self, z):
        phase = complex_angle(z)
        sin = tf.cast(tf.math.sin(phase), tf.complex64) 
        mz = tf.cast(complex_abs(z), tf.complex64)
        cos = tf.cast(tf.math.cos(phase), tf.complex64) 

        fx = 0.5 * (1 + cos) * z

        dfx = 0.5 + 0.5 * cos + 0.25 * 1j * sin
        
        dfxH = - 0.25 * 1j * sin * (z * z) / (mz * mz)

        return fx, dfx, dfxH
        
    def __str__(self):
        s = f"Cardioid2: bias_init={self.bias_init}, trainable={self.trainable}"
        return s

class cStudentT(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0, trainable=False):
        super().__init__()
        self.alpha_init = alpha
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer_alpha = tf.keras.initializers.Constant(self.alpha_init)
        self.alpha = self.add_weight('alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      trainable=self.trainable,
                                      )
    def _calc(self, x):
        d = 1 + self.alpha * x**2
        return tf.math.log(d) / (2 * self.alpha)

    def call(self, z):
        zre = tf.math.real(z)
        zim = tf.math.imag(z)

        actre = self._calc(zre)
        actim = self._calc(zim)
        return tf.complex(actre, actim)

    def __str__(self):
        s = f"cStudentT: alpha_init={self.alpha_init}, trainable={self.trainable}"
        return s

class cStudentT2(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0, trainable=False):
        super().__init__()
        self.alpha_init = alpha
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer_alpha = tf.keras.initializers.Constant(self.alpha_init)
        self.alpha = self.add_weight('alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      trainable=self.trainable
                                      )
    def call(self, z):
        zre = tf.math.real(z)
        zim = tf.math.imag(z)

        def g(x):
            d = 1 + self.alpha * x**2
            return tf.math.log(d) / (2 * self.alpha)

        def h(x):
            d = 1 + self.alpha * x**2
            return x / d

        act = tf.complex(g(zre), g(zim))

        dfzH = tf.cast(0.5 * (h(zre) - h(zim)), tf.complex64)
        dfz  = tf.cast(0.5 * (h(zre) + h(zim)), tf.complex64)
       
        return act, dfz, dfzH

    def __str__(self):
        s = f"cStudentT2: alpha_init={self.alpha_init}, trainable={self.trainable}"
        return s

class ModStudentT(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0, beta=0.1, trainable=False):
        super().__init__()
        self.alpha_init = alpha
        self.beta_init = beta
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer_beta = tf.keras.initializers.Constant(self.beta_init)
        self.beta = self.add_weight('beta',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_beta,
                                      trainable=self.trainable
                                      )
        initializer_alpha = tf.keras.initializers.Constant(self.alpha_init)
        self.alpha = self.add_weight('alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      trainable=self.trainable
                                      )
    def _calc(self, x):
        d = 1 + self.alpha * x**2
        return tf.math.log(d) / (2 * self.alpha)

    def call(self, z):
        act = self._calc(complex_abs(z) + self.beta)
        return tf.cast(act, tf.complex64) * complex_norm(z)

    def __str__(self):
        s = f"ModStudentT: alpha_init={self.alpha_init}, beta_init={self.beta_init}, trainable={self.trainable}"
        return s

class ModStudentT2(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0, beta=0.1, trainable=False):
        super().__init__()
        self.alpha_init = alpha
        self.beta_init = beta
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer_beta = tf.keras.initializers.Constant(self.beta_init)
        self._beta = self.add_weight('beta',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_beta,
                                      trainable=self.trainable
                                      )

        initializer_alpha = tf.keras.initializers.Constant(self.alpha_init)
        self._alpha = self.add_weight('alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      trainable=self.trainable
                                      )

    @property
    def beta(self):
        return tf.cast(self._beta, tf.complex64)

    @property
    def alpha(self):
        return tf.cast(self._alpha, tf.complex64)

    def call(self, z):
        mz = tf.cast(complex_abs(z), tf.complex64)
        nz = complex_norm(z)

        d = 1 + self.alpha * (mz + self.beta)**2
        
        act = tf.math.log(d) / (2 * self.alpha) * nz

        dfx = (mz + self.beta) / (2 * d) + tf.math.log(d)  / (4 * self.alpha * mz)
        dfxH = (z * z) / (2 * mz ** 2) * ((mz + self.beta)/d - tf.math.log(d) / (2 * self.alpha * mz))

        return act, dfx, dfxH

    def __str__(self):
        s = f"ModStudentT2: alpha_init={self.alpha_init}, beta_init={self.beta_init}, trainable={self.trainable}"
        return s

class TestActivation(unittest.TestCase):   
    def _test(self, act, args, shape):
        model = act(**args)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)
        print(model)
    
    def test_cReLU(self):
        self._test(cReLU, {}, [5, 32])

    def test_cPReLU(self):
        self._test(cPReLU, {'alpha':0.1, 'trainable':True}, [5, 32])

    def test_ModReLU(self):
        self._test(ModReLU, {'bias':0.1, 'trainable':True}, [5, 32])

    def test_Cardioid(self):
        self._test(Cardioid, {'bias':0.1, 'trainable':True}, [5, 32])

    def test_ModPReLU(self):
        self._test(ModPReLU, {'alpha':0.1, 'bias':0.01, 'trainable':True}, [5, 32])

    def test_cStudentT(self):
        self._test(cStudentT, {'alpha':0.1, 'trainable':True}, [5, 32])

    def test_ModStudentT(self):
        self._test(ModStudentT, {'alpha':0.1, 'beta':0.01, 'trainable':True}, [5, 32])

class TestActivation2(unittest.TestCase):   
    def _test(self, act, args, shape):
        model = act(**args)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))

        with tf.GradientTape() as g:
            g.watch(x)
            fx, dfx, dfxH = model(x)
            loss = 0.5 * tf.reduce_sum(tf.math.conj(fx) * fx)

        grad_x = g.gradient(loss, x)
        x_autograd = grad_x.numpy()

        zH = fx
        z = tf.math.conj(zH)
        fprimex = z * dfxH + zH * tf.math.conj(dfx)
        x_bwd = fprimex.numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_ModStudentT2(self):
        self._test(ModStudentT2, {'alpha':0.1, 'beta':0.01, 'trainable':True}, [5, 32])

    def test_cStudentT2(self):
        self._test(cStudentT2, {'alpha':0.1, 'trainable':True}, [5, 32])

    def test_CardioidT2(self):
        self._test(Cardioid2, {'bias':0.1, 'trainable':True}, [5, 32])

if __name__ == "__main__":
    unittest.test()