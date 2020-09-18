import tensorflow as tf
import unittest
from complex_layer import *

# __all__ = ['cReLU',
#            'cPReLU',
#            'ModReLU',
#            'ModPReLU',
#         #    'ComplexStudentT2',
#         #    'ComplexStudentT',
#         #    'ComplexTrainablePolarActivation',
#         #    'ComplexTrainablePolarActivationBias',
#         #    'ComplexTrainableMagnitudeActivation',
#         #    'ComplexTrainableMagnitudeActivationBias'
#         ]

class cReLU(tf.keras.layers.Layer):
    def call(self, z):
        actre = tf.keras.activations.relu(tf.math.real(z))
        actim = tf.keras.activations.relu(tf.math.imag(z))
        return tf.complex(actre, actim)

class ModReLU(tf.keras.layers.Layer):
    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.Constant(0)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      )
    def call(self, z):
        return tf.cast(tf.keras.activations.relu(complex_abs(z) + self.bias), tf.complex64) * complex_norm(z)

class cPReLU(tf.keras.layers.Layer):
    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.Constant(0.1)
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

class ModPReLU(tf.keras.layers.Layer):
    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.Constant(0)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      )
        initializer_alpha = tf.keras.initializers.Constant(0.1)
        self.alpha = self.add_weight('alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      )
    def call(self, z):
        act = tf.maximum(0.0, complex_abs(z) + self.bias) + self.alpha * tf.minimum(0.0, complex_abs(z) + self.bias)
        return tf.cast(act, tf.complex64) * complex_norm(z)


class cStudentT(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
    # def build(self, input_shape):
    #     super().build(input_shape)
    #     initializer = tf.keras.initializers.Constant(0)
    #     self.bias = self.add_weight('bias',
    #                                   shape=(input_shape[-1]),
    #                                   initializer=initializer,
    #                                   )
    #     initializer_alpha = tf.keras.initializers.Constant(0.1)
    #     self.alpha = self.add_weight('alpha',
    #                                   shape=(input_shape[-1]),
    #                                   initializer=initializer_alpha,
    #                                   )
    def _calc(self, x):
        d = 1 + self.alpha * x**2
        return tf.math.log(d) / (2 * self.alpha)

    def call(self, z):
        zre = tf.math.real(z)
        zim = tf.math.imag(z)

        actre = self._calc(zre)
        actim = self._calc(zim)
        return tf.complex(actre, actim)

class ModStudentT(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.Constant(0)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      )
        # initializer_alpha = tf.keras.initializers.Constant(0.1)
        # self.alpha = self.add_weight('alpha',
        #                               shape=(input_shape[-1]),
        #                               initializer=initializer_alpha,
        #                               )
    def _calc(self, x):
        d = 1 + self.alpha * x**2
        return tf.math.log(d) / (2 * self.alpha)

    def call(self, z):
        act = self._calc(complex_abs(z) + self.bias)
        return tf.cast(act, tf.complex64) * complex_norm(z)

class cStudentT2(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
    # def build(self, input_shape):
    #     super().build(input_shape)
    #     initializer_alpha = tf.keras.initializers.Constant(0.1)
    #     self.alpha = self.add_weight('alpha',
    #                                   shape=(input_shape[-1]),
    #                                   initializer=initializer_alpha,
    #                                   )
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

    #     actre = self._calc(zre)
    #     actim = self._calc(zim)
    #     return tf.complex(actre[0], actim[0]), tf.complex(actre[1], actim[1])
    # def call(self, z):
    #     with tf.GradientTape() as g:
    #         g.watch(z)
    #         zre = tf.math.real(z)
    #         zim = tf.math.imag(z)
    #         actre = tf.math.log(1 + self.alpha * zre ** 2) / ( 2 * self.alpha)
    #         actim = tf.math.log(1 + self.alpha * zim ** 2) / ( 2 * self.alpha)

    #         act = tf.complex(actre, actim)

    #     act_prime = g.gradient(act, z)
    #     return act, act_prime

class Idendity2(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
    # def build(self, input_shape):
    #     super().build(input_shape)
    #     initializer_alpha = tf.keras.initializers.Constant(0.1)
    #     self.alpha = self.add_weight('alpha',
    #                                   shape=(input_shape[-1]),
    #                                   initializer=initializer_alpha,
    #                                   )
    def call(self, z):
        return z, tf.zeros_like(z)

class ModStudentT2(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.Constant(0.1)
        self._bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      )
        # initializer_alpha = tf.keras.initializers.Constant(0.1)
        # self.alpha = self.add_weight('alpha',
        #                               shape=(input_shape[-1]),
        #                               initializer=initializer_alpha,
        #                               )

    @property
    def bias(self):
        return tf.cast(self._bias, tf.complex64)

    def call(self, z):
        mz = tf.cast(complex_abs(z), tf.complex64)
        nz = complex_norm(z)

        d = 1 + self.alpha * (mz + self.bias)**2
        
        act = tf.math.log(d) / (2 * self.alpha) * nz

        dx = (mz + self.bias) / (2 * d) + tf.math.log(d)  / (4 * self.alpha * mz)
        dxH = (z * z) / (2 * mz ** 2) * ((mz + self.bias)/d - tf.math.log(d) / (2 * self.alpha * mz))

        return act, dx, dxH

class TestActivation(unittest.TestCase):   
    def _test(self, act, shape):
        model = act()
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)
    
    def test_cReLU(self):
        self._test(cReLU, [5, 32])

    def test_cPReLU(self):
        self._test(cPReLU, [5, 32])

    def test_ModReLU(self):
        self._test(ModReLU, [5, 32])

    def test_ModPReLU(self):
        self._test(ModPReLU, [5, 32])

# class TestStudentTActivation(unittest.TestCase):   
#     def _test(self, act, shape):
#         model = act()
#         x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
#         act, act_prime = model(x)

#     def test_ModStudentT2(self):
#         self._test(ModStudentT2, [5, 32])

#     def test_cStudentT2(self):
#         self._test(cStudentT2, [5, 32])

    
if __name__ == "__main__":
    unittest.test()