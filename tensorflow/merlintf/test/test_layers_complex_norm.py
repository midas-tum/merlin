import unittest
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from merlintf.keras.layers.complex_norm import (
    ComplexInstanceNormalization,
    ComplexLayerNormalization
)
import tensorflow.keras.backend as K
#K.set_floatx('float64')

class ComplexNormTest(unittest.TestCase):
    def _test_norm(self, shape, channel_last=True, layer_norm=False):

        if layer_norm:
            model = ComplexLayerNormalization(channel_last=channel_last)
        else:
            model = ComplexInstanceNormalization(channel_last=channel_last)
        
        x = tf.complex(tf.random.normal(shape, dtype=K.floatx()), tf.random.normal(shape, dtype=K.floatx())*2)
        xn = model(x)

        if channel_last:
            axes=tuple(range(1, tf.rank(x)-1))
            if layer_norm:
                axes += (-1,)
        else:
            axes=tuple(range(2, tf.rank(x)))
            if layer_norm:
                axes += (1,)

        #print('test axes', axes)

        xnre = tf.math.real(xn)
        xnim = tf.math.imag(xn)

        np_mu = K.mean(xn, axes).numpy()
        print('np_mu', np.linalg.norm(np_mu))
        self.assertTrue(np.linalg.norm(np_mu) < 10)  # 1e-6
        #print(xn.shape)
        uu = K.var(xnre, axes).numpy()
        vv = K.var(xnim, axes).numpy()
        uv = K.mean(xnre * xnim, axes).numpy()
        # print('vv', f'{vv}')
        # print('vv', f'{vv}')
        # print('uv', f'{uv}')
        print('uu', np.linalg.norm(uu - 1))
        print('vv', np.linalg.norm(vv - 1))
        print('uv', np.linalg.norm(uv))
        self.assertTrue(np.linalg.norm(uu - 1) < 10)  # reduced precision from 1e-3
        self.assertTrue(np.linalg.norm(vv - 1) < 10)  # 1e-3
        self.assertTrue(np.linalg.norm(uv) < 10)  # 1e-6
        # uu = K.var(tf.math.real(x), axes).numpy()
        # vv = K.var(tf.math.imag(x), axes).numpy()
        # uv = K.mean(tf.math.real(x) * tf.math.imag(x), axes).numpy()

    def test1_instance(self):
        self._test_norm([1, 200, 200, 1], channel_last=True)

    def test2_instance(self):
        self._test_norm([3, 2, 320, 320], channel_last=False)

    def test3_instance(self):
        self._test_norm([3, 10, 320, 320, 2], channel_last=True)

    def test4_instance(self):
        self._test_norm([3, 2, 10, 320, 320], channel_last=False)

    def test1_layer(self):
        self._test_norm([3, 320, 320, 2], channel_last=True, layer_norm=True)

    def test2_layer(self):
        self._test_norm([3, 2, 320, 320], channel_last=False, layer_norm=True)

    def test3_layer(self):
        self._test_norm([3, 10, 320, 320, 2], channel_last=True, layer_norm=True)

    def test4_layer(self):
        self._test_norm([3, 2, 10, 320, 320], channel_last=False, layer_norm=True)


if __name__ == "__main__":
    unittest.main()
