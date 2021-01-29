import tensorflow as tf
import merlintf
import unittest
import six

def get(identifier):
    return MagnitudeMaxPooling(identifier)

def MagnitudeMaxPooling(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'MagnitudeMaxPool' + str(identifier).upper().replace('T','t')
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret max pooling function identifier: {}'.format(identifier))

def deserialize(op):
    if op == 'MagnitudeMaxPool1D' or op == 'MagnitudeMaxPooling1D':
        return MagnitudeMaxPool1D
    elif op == 'MagnitudeMaxPool2D' or op == 'MagnitudeMaxPooling2D':
        return MagnitudeMaxPool2D
    elif op == 'MagnitudeMaxPool2Dt' or op == 'MagnitudeMaxPooling2Dt':
        return MagnitudeMaxPool2Dt
    elif op == 'MagnitudeMaxPool3D' or op == 'MagnitudeMaxPooling3D':
        return MagnitudeMaxPool3D
    else:
        raise ValueError(f"Selected operation '{conv}' not implemented in complex convolutional")

class MagnitudeMaxPool(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, padding='SAME'):
        super(MagnitudeMaxPool, self).__init__()
        #assert isinstance(ksize, int)
        #assert isinstance(strides, int)
        self.ksize = ksize
        self.strides = strides
        self.padding =  padding

    def call(self, x, ):
        xabs = merlintf.complex_abs(x)
        _, idx = tf.nn.max_pool_with_argmax(
    xabs, self.ksize, self.strides, self.padding, include_batch_in_index=True)
        x_pool = tf.reshape(tf.gather(tf.reshape(x,shape= [-1,]),idx), shape=idx.shape)

        return x_pool

class MagnitudeMaxPool1D(MagnitudeMaxPool):
    def __init__(self, ksize, strides, padding='SAME'):
        super(MagnitudeMaxPool1D, self).__init__(ksize, strides, padding)

class MagnitudeMaxPool2D(MagnitudeMaxPool):
    def __init__(self, ksize, strides, padding='SAME'):
        super(MagnitudeMaxPool2D, self).__init__(ksize, strides, padding)

class MagnitudeMaxPool3D(MagnitudeMaxPool):
    def __init__(self, ksize, strides, padding='SAME'):
        super(MagnitudeMaxPool3D, self).__init__(ksize, strides, padding)

class MagnitudeMaxPool2Dt(MagnitudeMaxPool):
    def __init__(self, ksize, strides, padding='SAME'):
        super(MagnitudeMaxPool2Dt, self).__init__(ksize, strides, padding)

    def call(self, x, ):
        orig_shape = x.shape
        rank = tf.rank(x)
        batched_shape = [x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
        x = tf.reshape(x, batched_shape)

        xabs = merlintf.complex_abs(x)
        _, idx = tf.nn.max_pool_with_argmax(
    xabs, self.ksize, self.strides, self.padding, include_batch_in_index=True)
        x_pool = tf.reshape(tf.gather(tf.reshape(x,shape= [-1,]),idx), shape=idx.shape)

        pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
        x_pool = tf.reshape(x_pool, pooled_shape)
        return x_pool

class TestMagnitudePool(unittest.TestCase):
    def _test(self, shape, ksize=2, strides=2):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool(ksize, strides)

        y = pool(x)
        magn = merlintf.complex_abs(y)

    #def test1d(self):
    #    self._test([2, 2, 1])

    def test2d(self):
        self._test([2, 2, 2, 1])
        self._test([2, 2, 2, 1], (2, 2))

    def test2dt(self):
        self._test([2, 4, 2, 2, 1])

    #def test3d(self):
    #    self._test([2, 16, 8, 4, 1])
    #    self._test([2, 16, 8, 4, 1], (4, 2, 2))

if __name__ == "__main__":
    unittest.test()
