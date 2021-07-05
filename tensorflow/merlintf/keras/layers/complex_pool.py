import tensorflow as tf

import optotf.maxpooling
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
        raise ValueError(f"Selected operation '{op}' not implemented in complex convolutional")

class MagnitudeMaxPool(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding='SAME'):
        super(MagnitudeMaxPool, self).__init__()
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        self.padding =  padding
        self.alpha = 1 # magnitude ratio in real part
        self.beta = 1  # magnitude ratio in imag part

    def call(self, x):
        xabs = merlintf.complex_abs(x)
        _, idx = tf.nn.max_pool_with_argmax(
    xabs, self.pool_size, self.strides, self.padding, include_batch_in_index=True)
        x_pool = tf.reshape(tf.gather(tf.reshape(x,shape= [-1,]),idx), shape=idx.shape)

        return x_pool

class MagnitudeMaxPool1D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME'):
        super(MagnitudeMaxPool1D, self).__init__(pool_size, strides, padding)

class MagnitudeMaxPool2D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME'):
        super(MagnitudeMaxPool2D, self).__init__(pool_size, strides, padding)

class MagnitudeMaxPool3D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME'):
        super(MagnitudeMaxPool3D, self).__init__(pool_size, strides, padding)




class MagnitudeMaxPool2Dt(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME'):
        super(MagnitudeMaxPool2Dt, self).__init__(pool_size, strides, padding)

    def call(self, x):
        orig_shape = x.shape
        rank = tf.rank(x)
        batched_shape = [x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
        x = tf.reshape(x, batched_shape)

        xabs = merlintf.complex_abs(x)
        _, idx = tf.nn.max_pool_with_argmax(
    xabs, self.pool_size, self.strides, self.padding, include_batch_in_index=True)
        x_pool = tf.reshape(tf.gather(tf.reshape(x,shape= [-1,]),idx), shape=idx.shape)

        pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
        x_pool = tf.reshape(x_pool, pooled_shape)
        return x_pool


class MagnitudeMaxPool2D_1(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding='SAME'):
        super(MagnitudeMaxPool2D_1, self).__init__()
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        self.padding =  padding
        self.alpha = 1 # magnitude ratio in real part
        self.beta = 1  # magnitude ratio in imag part

    def call(self, x):
        if merlintf.iscomplextf(x):
            x_pool, _ = optotf.maxpooling.maxpooling2d(x, pooling=self.pool_size, stride=self.strides, alpha=self.alpha, beta=self.beta,mode=self.padding)
        else:
            x_pool = tf.nn.max_pool(x,ksize=self.pool_size,strides=self.strides,padding=self.padding)
        return x_pool



class MagnitudeMaxPool2Dt_1(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME'):
        super(MagnitudeMaxPool2Dt_1, self).__init__(pool_size, strides, padding)

    def call(self, x):
        orig_shape = x.shape
        rank = tf.rank(x)
        batched_shape = [x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
        x = tf.reshape(x, batched_shape)


        if merlintf.iscomplextf(x):
            x_pool,_ = optotf.maxpooling.maxpooling2d(x, pooling=self.pool_size, stride=self.strides, alpha=self.alpha, beta=self.beta, mode=self.padding)
        else:
            x_pool = tf.nn.max_pool(x, ksize=self.pool_size, strides=self.strides, padding=self.padding)


        pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
        x_pool = tf.reshape(x_pool, pooled_shape)
        return x_pool


class MagnitudeMaxPool3D_1(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding='SAME'):
        super(MagnitudeMaxPool3D_1, self).__init__()
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        self.padding =  padding
        self.alpha = 1 # magnitude ratio in real part
        self.beta = 1  # magnitude ratio in imag part

    def call(self, x):
        if merlintf.iscomplextf(x):
            x_pool, _ = optotf.maxpooling.maxpooling3d(x, pooling=self.pool_size, stride=self.strides,  alpha=self.alpha, beta=self.beta, mode=self.padding)
        else:
            x_pool = tf.nn.max_pool3d(x,ksize=self.pool_size,strides=self.strides,padding=self.padding)
        return x_pool



class MagnitudeMaxPool3Dt_1(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME'):
        super(MagnitudeMaxPool3Dt_1, self).__init__(pool_size, strides, padding)

    def call(self, x):
        orig_shape = x.shape
        rank = tf.rank(x)
        if rank==6:
            batched_shape = [x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]]
        elif rank == 5:
            batched_shape = [x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]]

        if merlintf.iscomplextf(x):
            x_pool, _  = optotf.maxpooling.maxpooling3d(x, ksize=self.pool_size, strides=self.strides, alpha=self.alpha, beta=self.beta, padding=self.padding)
        else:
            x_pool = tf.nn.max_pool(x, ksize=self.pool_size, strides=self.strides, padding=self.padding)

        pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
        x_pool = tf.reshape(x_pool, pooled_shape)
        return x_pool





class TestMagnitudePool(unittest.TestCase):
    def _test(self, shape, pool_size=2, strides=2):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool(pool_size, strides)

        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_t(self, shape, pool_size=2, strides=2):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool2Dt(pool_size, strides)

        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_1(self, shape, pool_size=(2,2), strides=(2,2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool2D_1(pool_size, strides)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_2(self, shape, pool_size=(2,2,2), strides=(2,2,2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool3D_1(pool_size, strides)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    

    #def test1d(self):
    #    self._test([2, 2, 1])

    def test2d(self):
        self._test([2, 2, 2, 1])
        self._test([2, 2, 2, 1], (2, 2))

    def test2dt(self):
        self._test_t([2, 4, 2, 2, 1])

    def test_1(self):
        # Maxpooling 2d
        self._test_1([2, 2, 2, 1])
        self._test_1([2, 2, 2, 1], (2, 2))
    def test_2(self):
        # Maxpooling 3d
        self._test_2([2, 16, 8, 4, 1])
        self._test_2([2, 16, 8, 4, 1], (4, 2, 2))


if __name__ == "__main__":
    #unittest.test()
    unittest.main()
