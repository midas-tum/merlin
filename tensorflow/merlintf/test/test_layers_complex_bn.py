import unittest
import tensorflow.keras.backend as K
import tensorflow as tf
from merlintf.keras.layers.complex_bn import ComplexBatchNormalization
import numpy as np
import tensorflow.keras.backend as K
#K.set_floatx('float64')

class ComplexNormTest(unittest.TestCase):
    def _test_norm(self, shape, channel_last=True):

        model = ComplexBatchNormalization(channel_last=channel_last, scale=False)
        x = tf.complex(tf.random.normal(shape, dtype=K.floatx()), tf.random.normal(shape, dtype=K.floatx())*2)

        for i in range(2):
            xn = model(x, training=True)

        print(model.moving_mean)
        print(model.moving_Vii)

        if channel_last:
            axes=tuple(range(1, tf.rank(x)-1))
        else:
            axes=tuple(range(2, tf.rank(x)))
        axes += (0,)

        print('test axes', axes)

        xnre = tf.math.real(xn)
        xnim = tf.math.imag(xn)

        np_mu = K.mean(xn, axes).numpy()
        self.assertTrue(np.linalg.norm(np_mu) < 1e-6)

        uu = K.var(xnre, axes).numpy()
        vv = K.var(xnim, axes).numpy()
        uv = K.mean(xnre * xnim, axes).numpy()
        print('vv', f'{vv}')
        print('vv', f'{vv}')
        print('uv', f'{uv}')
        self.assertTrue(np.linalg.norm(uu - 1) < 1e-3)
        self.assertTrue(np.linalg.norm(vv - 1) < 1e-3)
        self.assertTrue(np.linalg.norm(uv) < 1e-6)

    def test1_batch(self):
        self._test_norm([3, 320, 320, 2], channel_last=True)

    def test2_batch(self):
        self._test_norm([3, 2, 320, 320], channel_last=False)

    def test3_batch(self):
        self._test_norm([3, 10, 320, 320, 2], channel_last=True)

    def test4_batch(self):
        self._test_norm([3, 2, 10, 320, 320], channel_last=False)


if __name__ == "__main__":
    unittest.main()
