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

class TestMagnitudePool(unittest.TestCase):
    def test4d(self):
        self._test((1, 4, 6, 6, 6, 2), (2, 2, 2, 2), (2, 2, 2, 2), 'valid')
        self._test((1, 4, 6, 6, 6, 2), (2, 2, 2, 2), (2, 2, 2, 2), 'same')
        self._test((1, 5, 7, 7, 7, 2), (2, 2, 2, 2), (2, 2, 2, 2), 'valid')

    def test3d(self):
        self._test((1, 4, 6, 6, 2), (2, 2, 2), (2, 2, 2), 'valid')
        self._test((1, 4, 6, 6, 2), (2, 2, 2), (2, 2, 2), 'same')
        self._test((1, 5, 7, 7, 2), (2, 2, 2), (2, 2, 2), 'valid')

    def test2d(self):
        self._test((1, 4, 6, 2), (2, 2), (2, 2), 'valid')
        self._test((1, 4, 6, 2), (2, 2), (2, 2), 'same')
        self._test((1, 5, 7, 2), (2, 2), (2, 2), 'valid')

    def test1d(self):
        self._test((1, 4, 2), (2,), (2, ), 'valid')
        self._test((1, 4, 2), (2,), (2, ), 'same')
        self._test((1, 5, 2), (2,), (2, ), 'valid')

    def _padding_shape(self, input_spatial_shape, spatial_filter_shape, strides, dilations_rate, padding_mode):
        if padding_mode.lower() == 'valid':
            return np.ceil((input_spatial_shape - (spatial_filter_shape - 1) * dilations_rate) / strides)
        elif padding_mode.lower() == 'same':
            return np.ceil(input_spatial_shape / strides)
        else:
            raise Exception('padding_mode can be only valid or same!')

    def _test(self, shape, pool_size, strides, padding_mode):
        # test tf.nn.average_pool_with_argaverage
        x = merlintf.random_normal_complex(shape)

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
        for i in len(shape) - 2:
            expected_shape.append(
                self.padding_shape(shape[i + 1], pool_size[i], strides[i], dilations_rate[i], padding_mode))
        expected_shape.append(shape[-1])

        self.assertTrue(np.abs(np.array(expected_shape) - np.array(out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array(x.shape) - np.array(gradients.shape)).all() < 1e-8)


if __name__ == "__main__":
    unittest.main()