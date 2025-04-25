import unittest
import tensorflow as tf
import numpy as np
import merlintf
from merlintf.keras.layers.complex_avgpool import (
    MagnitudeAveragePool1D,
    MagnitudeAveragePool2D,
    MagnitudeAveragePool2Dt,
    MagnitudeAveragePool3D,
    MagnitudeAveragePool3Dt
)
import tensorflow.keras.backend as K
#K.set_floatx('float32')

class TestMagnitudePool(unittest.TestCase):
    def test4d(self):
        self._test((1, 4, 6, 6, 6, 2), (2, 2, 2, 2), (2, 2, 2, 2), 'valid')
        self._test((1, 4, 6, 6, 6, 2), (2, 2, 2, 2), (2, 2, 2, 2), 'same')
        self._test((1, 5, 7, 7, 7, 2), (2, 2, 2, 2), (2, 2, 2, 2), 'valid')
        self._test((1, 5, 7, 7, 7, 2), (2, 2, 2, 2), (2, 2, 2, 2), 'same')

    def test3d(self):
        self._test((1, 4, 6, 6, 2), (2, 2, 2), (2, 2, 2), 'valid')
        self._test((1, 4, 6, 6, 2), (2, 2, 2), (2, 2, 2), 'same')
        self._test((1, 5, 7, 7, 2), (2, 2, 2), (2, 2, 2), 'valid')
        self._test((1, 5, 7, 7, 2), (2, 2, 2), (2, 2, 2), 'same')

        self._verify_shape((1, 4, 6, 6, 2), (2, 2, 2), (2, 2, 2), 'valid')
        self._verify_shape((1, 4, 6, 6, 2), (2, 2, 2), (2, 2, 2), 'same')
        self._verify_shape((1, 5, 7, 7, 2), (2, 2, 2), (2, 2, 2), 'valid')
        self._verify_shape((1, 5, 7, 7, 2), (2, 2, 2), (2, 2, 2), 'same')

    def test2d(self):
        self._test((1, 4, 6, 2), (2, 2), (2, 2), 'valid')
        self._test((1, 4, 6, 2), (2, 2), (2, 2), 'same')
        self._test((1, 5, 7, 2), (2, 2), (2, 2), 'valid')
        self._test((1, 5, 7, 2), (2, 2), (2, 2), 'same')

        self._verify_shape((1, 4, 6, 2), (2, 2), (2, 2), 'valid')
        self._verify_shape((1, 4, 6, 2), (2, 2), (2, 2), 'same')
        self._verify_shape((1, 5, 7, 2), (2, 2), (2, 2), 'valid')
        self._verify_shape((1, 5, 7, 2), (2, 2), (2, 2), 'same')

    def test1d(self):
        self._test((1, 4, 2), (2,), (2, ), 'valid')
        self._test((1, 4, 2), (2,), (2, ), 'same')
        self._test((1, 5, 2), (2,), (2, ), 'valid')
        self._test((1, 5, 2), (2,), (2,), 'same')

        self._verify_shape((1, 4, 2), (2,), (2,), 'valid')
        self._verify_shape((1, 4, 2), (2,), (2,), 'same')
        self._verify_shape((1, 5, 2), (2,), (2,), 'valid')
        self._verify_shape((1, 5, 2), (2,), (2,), 'same')

    def _padding_shape(self, input_spatial_shape, spatial_filter_shape, strides, dilations_rate, padding_mode):
        if padding_mode.lower() == 'valid':
            return np.ceil((input_spatial_shape - (spatial_filter_shape - 1) * dilations_rate) / strides)
        elif padding_mode.lower() == 'same':
            return np.ceil(input_spatial_shape / strides)
        else:
            raise Exception('padding_mode can be only valid or same!')
        
    def _verify_shape(self, shape, pool_size, strides, padding_mode):
        x = merlintf.random_normal_complex(shape, dtype=tf.float32)

        if len(shape) == 3:  # 1d
            op = MagnitudeAveragePool1D(pool_size, strides, padding_mode)
            op_backend = tf.keras.layers.AveragePooling1D(pool_size, strides, padding_mode)
        elif len(shape) == 4:  # 2d
            op = MagnitudeAveragePool2D(pool_size, strides, padding_mode)
            op_backend = tf.keras.layers.AveragePooling2D(pool_size, strides, padding_mode)
        elif len(shape) == 5:  # 3d
            op = MagnitudeAveragePool3D(pool_size, strides, padding_mode)
            op_backend = tf.keras.layers.AveragePooling3D(pool_size, strides, padding_mode)
        elif len(shape) == 6:  # 4d
            op = MagnitudeAveragePool3Dt(pool_size, strides, padding_mode)

        out = op(x)
        out_backend = op_backend(merlintf.complex_abs(x))

        self.assertTrue(np.sum(np.abs(np.array(out.shape) - np.array(out_backend.shape))) == 0)

    def _test(self, shape, pool_size, strides, padding_mode, dilations_rate=(1, 1, 1, 1)):
        # test tf.nn.average_pool_with_argaverage
        x = merlintf.random_normal_complex(shape, dtype=tf.float32)

        if len(shape) == 3:  # 1d
            op = MagnitudeAveragePool1D(pool_size, strides, padding_mode)
        elif len(shape) == 4:  # 2d
            op = MagnitudeAveragePool2D(pool_size, strides, padding_mode)
        elif len(shape) == 5:  # 3d
            op = MagnitudeAveragePool3D(pool_size, strides, padding_mode)
        elif len(shape) == 6:  # 4d
            op = MagnitudeAveragePool3Dt(pool_size, strides, padding_mode)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            out_complex = op(x)
            gradients = tape.gradient(tf.math.reduce_sum(out_complex), x)

        # (N, T, H, W, D, C)
        expected_shape = [shape[0]]
        for i in range(len(shape) - 2):
            expected_shape.append(self._padding_shape(shape[i + 1], pool_size[i], strides[i], dilations_rate[i], padding_mode))
        expected_shape.append(shape[-1])

        self.assertTrue(np.abs(np.array(expected_shape) - np.array(out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array(x.shape) - np.array(gradients.shape)).all() < 1e-8)


if __name__ == "__main__":
    unittest.main()