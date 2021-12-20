import tensorflow as tf
import numpy as np
import optotf.averagepooling
import merlintf
import unittest
import six


def get(identifier):
    return MagnitudeAveragePooling(identifier)


def MagnitudeAveragePooling(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'MagnitudeAveragePool' + str(identifier).upper().replace('T', 't')
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret average pooling function identifier: {}'.format(identifier))


def deserialize(op):
    if op == 'MagnitudeAveragePool1D' or op == 'MagnitudeAveragePooling1D':
        return MagnitudeAveragePool1D
    elif op == 'MagnitudeAveragePool2D' or op == 'MagnitudeAveragePooling2D':
        return MagnitudeAveragePool2D
    elif op == 'MagnitudeAveragePool2Dt' or op == 'MagnitudeAveragePooling2Dt':
        return MagnitudeAveragePool2Dt
    elif op == 'MagnitudeAveragePool3D' or op == 'MagnitudeAveragePooling3D':
        return MagnitudeAveragePool3D
    elif op == 'MagnitudeAveragePool3Dt' or op == 'MagnitudeAveragePooling3Dt':
        return MagnitudeAveragePool3Dt
    else:
        raise ValueError(f"Selected operation '{op}' not implemented in complex convolutional")


class MagnitudeAveragePool(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True):
        super(MagnitudeAveragePool, self).__init__()
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        self.padding = padding
        self.alpha = 1  # magnitude ratio in real part
        self.beta = 1  # magnitude ratio in imag part
        self.optox = optox  # True: execute Optox pooling; False: use TF pooling (not supported for all cases)


    def call(self, x):  # default to TF
        xabs = merlintf.complex_abs(x)
        x_pool = tf.nn.avg_pool(
            xabs, self.pool_size, self.strides, self.padding)
        return x_pool

class MagnitudeAveragePool1D(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True):
        super(MagnitudeAveragePool1D, self).__init__(pool_size, strides, padding, optox)


class MagnitudeAveragePool2D(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True):
        super(MagnitudeAveragePool2D, self).__init__(pool_size, strides, padding, optox)

    def call(self, x):
        if self.optox:
            if merlintf.iscomplextf(x):
                x_pool = optotf.averagepooling.averagepooling2d(x, pool_size=self.pool_size, strides=self.strides,
                                                                      alpha=self.alpha, beta=self.beta,
                                                                      mode=self.padding)
                return x_pool
            else:
                x_pool = tf.nn.avg_pool2d(x, self.pool_size, self.strides, self.padding)
                return x_pool
        else:
            return super().call(x)


class MagnitudeAveragePool3D(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True):
        super(MagnitudeAveragePool3D, self).__init__(pool_size, strides, padding, optox)

    def call(self, x):
        if self.optox:
            if merlintf.iscomplextf(x):
                x_pool = optotf.averagepooling.averagepooling3d(x, pool_size=self.pool_size,
                                                                    strides=self.strides,
                                                                    alpha=self.alpha, beta=self.beta,
                                                                    mode=self.padding)

                return x_pool
            else:
                x_pool = tf.nn.avg_pool3d(x, ksize=self.pool_size, strides=self.strides, padding=self.padding)
                return x_pool
        else:
            return super().call(x)


class MagnitudeAveragePool2Dt(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True):
        super(MagnitudeAveragePool2Dt, self).__init__(pool_size, strides, padding, optox)

    def call(self, x):

            orig_shape = x.shape
            batched_shape = [x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
            x = tf.reshape(x, batched_shape)

            if merlintf.iscomplextf(x):
                x_pool = optotf.averagepooling.averagepooling2d(x, pool_size=self.pool_size, strides=self.strides,
                                                                      alpha=self.alpha, beta=self.beta,
                                                                      mode=self.padding)

            else:
                x_pool = tf.nn.avg_pool2d(x, ksize=self.pool_size, strides=self.strides,
                                                                  padding=self.padding)

            pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
            x_pool = tf.reshape(x_pool, pooled_shape)

            return x_pool


class MagnitudeAveragePool3Dt(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True):
        super(MagnitudeAveragePool3Dt, self).__init__(pool_size, strides, padding, optox)

    def call(self, x):  # only Optox supported
        orig_shape = x.shape
        rank = tf.rank(x)
        batched_shape = 0
        if rank == 6:
            batched_shape = [x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]]
        elif rank == 5:
            batched_shape = [x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
        x = tf.reshape(x, batched_shape)

        if merlintf.iscomplextf(x):
            x_pool = optotf.averagepooling.averagepooling3d(x, pool_size=self.pool_size, strides=self.strides,
                                                                  alpha=self.alpha,
                                                                  beta=self.beta, mode=self.padding)
        else:
            # same as above
            x_pool = optotf.averagepooling.averagepooling3d(x, pool_size=self.pool_size, strides=self.strides,
                                                                  alpha=self.alpha,
                                                                  beta=self.beta, mode=self.padding)

        pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
        x_pool = tf.reshape(x_pool, pooled_shape)
        return x_pool


class TestMagnitudePool(unittest.TestCase):
    def _test(self, shape, pool_size=2, strides=2):
        # test tf.nn.average_pool_with_argaverage
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeAveragePool(pool_size, strides, optox=False)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_2dt(self, shape, pool_size=2, strides=2):
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
        x_abs = tf.nn.average_pool3d(x_abs, pool_size, strides, padding='SAME')

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
    # unittest.test()
    unittest.main()
