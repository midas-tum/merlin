import tensorflow as tf
import merlintf
import unittest

class MagnitudeMaxPool(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, padding='SAME'):
        super().__init__()
        assert isinstance(ksize, int)
        assert isinstance(strides, int)
        self.ksize = ksize
        self.strides = strides
        self.padding =  padding

    def call(self, x, ):
        orig_shape = x.shape
        rank = tf.rank(x)
        if rank == 5:
            batched_shape = [x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
            x = tf.reshape(x, batched_shape)

        xabs = merlintf.complex_abs(x)
        _, idx = tf.nn.max_pool_with_argmax(
    xabs, self.ksize, self.strides, self.padding, include_batch_in_index=True)
        x_pool = tf.reshape(tf.gather(tf.reshape(x,shape= [-1,]),idx), shape=idx.shape)

        if rank == 5:
            pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
            x_pool = tf.reshape(x_pool, pooled_shape)
        return x_pool

class TestMagnitudePool(unittest.TestCase):
    def _test(self, shape):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool(2, 2)

        y = pool(x)
        magn = merlintf.complex_abs(y)

    def test2d(self):
        self._test([2, 2, 2, 1])

    def test2dt(self):
        self._test([2, 4, 2, 2, 1])

if __name__ == "__main__":
    unittest.test()
