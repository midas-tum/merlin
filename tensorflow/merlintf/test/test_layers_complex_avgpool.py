import unittest
import tensorflow as tf
import numpy as np
import merlintf
from merlintf.keras.layers.complex_avgpool import (
    MagnitudeAveragePool,
    MagnitudeAveragePool2D,
    MagnitudeAveragePool2Dt,
    MagnitudeAveragePool3D,
    MagnitudeAveragePool3Dt
)

class TestMagnitudePool(unittest.TestCase):
    def _test(self, shape, pool_size=2, strides=2):
        # test tf.nn.average_pool_with_argaverage
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeAveragePool(pool_size, strides, optox=False)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_2dt(self, shape, pool_size=(2, 2, 2), strides=(2, 2, 2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeAveragePool2Dt(pool_size, strides)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_2d(self, shape, pool_size=(2, 2), strides=(2, 2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeAveragePool2D(pool_size, strides, optox=True)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_3d(self, shape, pool_size=(2, 2, 2), strides=(2, 2, 2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeAveragePool3D(pool_size, strides, optox=True)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_3dt(self, shape, pool_size=(2, 2, 2, 2), strides=(2, 2, 2, 2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeAveragePool3Dt(pool_size, strides, optox=True)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_2d_accuracy(self, shape, pool_size=(2, 2), strides=(2, 2)):
        print('_______')
        print('test_2d_accuracy')
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        # averagepooling 2D in optotf
        pool = MagnitudeAveragePool2D(pool_size, strides, optox=True)
        y = pool(x)
        # averagepooling 2D in tf.nn.avg_pool2d
        x_abs = tf.math.abs(x)
        x_abs = tf.nn.avg_pool2d(x_abs, pool_size, strides, padding='SAME')

        print('tf.math.abs(y) - x_abs:', tf.math.abs(y) - x_abs)
        shape = x_abs.shape
        test_id = [np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[2]),
                   np.random.randint(0, shape[3])]
        self.assertTrue((tf.math.abs(y)[test_id[0], test_id[1], test_id[2], test_id[3]] - x_abs[
            test_id[0], test_id[1], test_id[2], test_id[3]]) == 0.0)

    def _test_3d_accuracy(self, shape, pool_size=(2, 2, 2), strides=(2, 2, 2)):
        print('_______')
        print('test_3d_accuracy...')
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        # averagepooling 3D  in optotf
        pool = MagnitudeAveragePool3D(pool_size, strides, optox=True)
        y = pool(x)

        x_abs = tf.math.abs(x)
        x_abs = tf.nn.avg_pool3d(x_abs, pool_size, strides, padding='SAME')

        print('tf.math.abs(y) - x_abs', tf.math.abs(y) - x_abs)
        shape = x_abs.shape
        test_id = [np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[2]),
                   np.random.randint(0, shape[3]), np.random.randint(0, shape[4])]
        self.assertTrue((tf.math.abs(y)[test_id[0], test_id[1], test_id[2], test_id[3], test_id[4]] - x_abs[
            test_id[0], test_id[1], test_id[2], test_id[3], test_id[4]]) == 0.0)

    def test_average_pool(self):
        self._test([2, 2, 2, 1])
        self._test([2, 2, 2, 1], (2, 2))

    def test_2dt(self):
        # Averagepooling 2dt
        self._test_2dt([2, 4, 2, 2, 1])

    def test_3dt(self):
        # Averagepooling 2dt
        self._test_3dt([2, 4, 2, 2, 2, 1])

    def test_2d(self):
        # Averagepooling 2d
        self._test_2d([2, 2, 2, 1])
        self._test_2d([2, 2, 2, 1], (2, 2))
        # input shape: [batch, height, width, channel]
        self._test_2d_accuracy([1, 8, 12, 3], pool_size=(3, 2))

    def test_2(self):
        # Averagepooling 3d
        self._test_3d([2, 16, 8, 4, 1])
        self._test_3d([2, 16, 8, 4, 1], (4, 2, 2))
        # input shape: [batch, height, width, depth, channel]
        self._test_3d_accuracy([2, 8, 6, 8, 2], pool_size=(3, 2, 2))


if __name__ == "__main__":
    unittest.main()