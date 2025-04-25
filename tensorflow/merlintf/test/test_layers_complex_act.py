
import unittest
import numpy as np
import tensorflow as tf
from merlintf.keras.layers.complex_act import (
    cReLU,
    ModReLU,
    cPReLU,
    ModPReLU,
    cStudentT,
    ModStudentT,
    cStudentT2,
    ModStudentT2,
    Cardioid,
    Cardioid2,
)
import tensorflow.keras.backend as K
#K.set_floatx('float64')

class TestActivation(unittest.TestCase):   
    def _test(self, act, args, shape):
        model = act(**args)
        x = tf.complex(tf.random.normal(shape, dtype=K.floatx()), tf.random.normal(shape, dtype=K.floatx()))
        Kx = model(x)
        print(model)
        print(model.get_config())
    
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
        x = tf.complex(tf.random.normal(shape, dtype=K.floatx()), tf.random.normal(shape, dtype=K.floatx()))

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

    def test_Cardioid2(self):
        self._test(Cardioid2, {'bias':0.1, 'trainable':True}, [5, 32])

if __name__ == "__main__":
    unittest.main()