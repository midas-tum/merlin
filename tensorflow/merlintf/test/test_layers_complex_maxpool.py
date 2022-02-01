import unittest
import numpy as np
import tensorflow as tf
import merlintf
from merlintf.keras.layers.complex_maxpool import (
    MagnitudeMaxPool,
    MagnitudeMaxPool2D,
    MagnitudeMaxPool2Dt,
    MagnitudeMaxPool3D,
    MagnitudeMaxPool3Dt
)

class TestMagnitudePool(unittest.TestCase):
    def _test(self, shape, pool_size=2, strides=2):
        # test tf.nn.max_pool_with_argmax
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool(pool_size, strides, optox=False)
        y = pool(x)
        magn = merlintf.complex_abs(y)
        pool_size = pool_size if (isinstance(pool_size, list) or isinstance(pool_size, tuple)) else [pool_size] * 2
        self.assertEqual(magn.shape.as_list(), [shape[0], shape[1] // pool_size[0], shape[2] // pool_size[1], shape[3]])

    def _test_2dt(self, shape, pool_size=(1, 2, 2), strides=2):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool2Dt(pool_size, strides, optox=True)
        y = pool(x)
        magn = merlintf.complex_abs(y)
        pool_size = pool_size if (isinstance(pool_size, list) or isinstance(pool_size, tuple)) else [pool_size] * 3
        self.assertEqual(magn.shape.as_list(), [shape[0], shape[1] // pool_size[0], shape[2] // pool_size[1], shape[3] // pool_size[2], shape[4]])

    def _test_2d(self, shape, pool_size=(2, 2), strides=(2, 2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool2D(pool_size, strides, optox=True)
        y = pool(x)
        magn = merlintf.complex_abs(y)
        pool_size = pool_size if (isinstance(pool_size, list) or isinstance(pool_size, tuple)) else [pool_size] * 2
        self.assertEqual(magn.shape.as_list(), [shape[0], shape[1] // pool_size[0], shape[2] // pool_size[1], shape[3]])

    def _test_3d(self, shape, pool_size=(2, 2, 2), strides=(2, 2, 2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool3D(pool_size, strides, optox=True)
        y = pool(x)
        magn = merlintf.complex_abs(y)
        pool_size = pool_size if (isinstance(pool_size, list) or isinstance(pool_size, tuple)) else [pool_size] * 3
        self.assertEqual(magn.shape.as_list(), [
        shape[0], shape[1] // pool_size[0], shape[2] // pool_size[1], shape[3] // pool_size[2], shape[4]])

    def _test_3dt(self, shape, pool_size=(2, 2, 2, 2), strides=(2, 2, 2, 2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool3Dt(pool_size, strides, optox=True)
        y = pool(x)
        magn = merlintf.complex_abs(y)
        pool_size = pool_size if (isinstance(pool_size, list) or isinstance(pool_size, tuple)) else [pool_size] * 4
        self.assertEqual(magn.shape.as_list(), [
        shape[0], shape[1] // pool_size[0], shape[2] // pool_size[1], shape[3] // pool_size[2], shape[4] // pool_size[3], shape[5]])

    def _index_transfer(self, index_input, index_1, include_batch_in_index=False):
        # transfer from optox argmax index-> tensorflow argmax index

        rank = tf.rank(index_input)
        if rank == 4:
            batch, height, width, channel = index_input.shape
            batch_o, height_o, width_o, channel_o = index_1.shape
            out_index = np.zeros((batch_o, height_o, width_o, channel_o))
            if include_batch_in_index:
                for b in range(batch_o):
                    for h in range(height_o):
                        for w in range(width_o):
                            for c in range(channel_o):  # int idx = h1*width_in+w1;
                                w1 = index_1[b, h, w, c] % width
                                h1 = (index_1[b, h, w, c] - w1) // width
                                out_index[b, h, w, c] = ((b * height + h1) * width + w1) * channel_o + c

            else:
                for b in range(batch_o):
                    for h in range(height_o):
                        for w in range(width_o):
                            for c in range(channel_o):
                                w1 = index_1[b, h, w, c] % width
                                h1 = (index_1[b, h, w, c] - w1) // width
                                out_index[b, h, w, c] = (h1 * width + w1) * channel_o + c
        elif rank == 5:
            batch, height, width, depth, channel = index_input.shape
            batch_o, height_o, width_o, depth_o, channel_o = index_1.shape
            out_index = np.zeros((batch_o, height_o, width_o, depth_o, channel_o))
            if include_batch_in_index:
                for b in range(batch_o):
                    for h in range(height_o):
                        for w in range(width_o):
                            for d in range(depth_o):
                                for c in range(channel_o):
                                    d1 = index_1[b, h, w, d, c] % depth
                                    temp1 = (index_1[b, h, w, d, c] - d1) // depth
                                    w1 = temp1 % width
                                    h1 = (temp1 - w1) // width
                                    out_index[b, h, w, d, c] = (((
                                            (b * height + h1) * width + w1)) * depth + d) * channel_o + c

            else:
                for b in range(batch_o):
                    for h in range(height_o):
                        for d in range(depth_o):
                            for w in range(width_o):
                                for c in range(channel_o):
                                    d1 = index_1[b, h, w, d, c] % depth
                                    temp1 = (index_1[b, h, w, d, c] - d1) // depth
                                    w1 = temp1 % width
                                    h1 = (temp1 - w1) // width
                                    out_index[b, h, w, d, c] = (((
                                            (height + h1) * width + w1)) * depth + d) * channel_o + c

        return out_index.astype(int)

    def _test_2d_accuracy(self, shape, pool_size=(2, 2), strides=(2, 2)):
        print('_______')
        print('test_2d_accuracy')
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        # maxpooling 2D with index in optotf
        pool = MagnitudeMaxPool2D(pool_size, strides, optox=True, argmax_index=True)
        y, out_ind_optox = pool(x)
        out_ind_optox = self._index_transfer(x, out_ind_optox, include_batch_in_index=True)
        print('out indice in optox: ', out_ind_optox)

        # maxpooling 2D with index in in tf.nn.max_pool_with_argmax
        x_abs = tf.math.abs(x)
        x_abs, out_ind_nn = tf.nn.max_pool_with_argmax(x_abs, pool_size, strides, padding='SAME',
                                                       include_batch_in_index=True)
        print('out indice in tf.nn: ', out_ind_nn)

        # assert out_ind_optox=out_ind_nn
        print('out_ind_optox - out_ind_nn:', out_ind_optox - out_ind_nn)
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
        # maxpooling 3D with index in optotf
        pool = MagnitudeMaxPool3D(pool_size, strides, optox=True, argmax_index=True)
        y, out_ind_optox = pool(x)
        out_ind_optox = self._index_transfer(x, out_ind_optox, include_batch_in_index=True)
        print('out indice in optox: ', out_ind_optox)

        # No 3D index in tf.nn.max_pool_with_argmax
        x_abs = tf.math.abs(x)
        x_abs = tf.nn.max_pool3d(x_abs, pool_size, strides, padding='SAME')

        print('tf.math.abs(y) - x_abs',tf.math.abs(y) - x_abs)
        shape = x_abs.shape
        test_id = [np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[2]),
                   np.random.randint(0, shape[3]), np.random.randint(0, shape[4])]
        self.assertTrue((tf.math.abs(y)[test_id[0], test_id[1], test_id[2], test_id[3], test_id[4]] - x_abs[
            test_id[0], test_id[1], test_id[2], test_id[3], test_id[4]]) == 0.0)

    def test_max_pool_with_argmax(self):
        self._test([2, 2, 2, 1])
        self._test([2, 2, 2, 1], (2, 2))

    def test_2dt(self):
        # Maxpooling 2dt
        self._test_2dt([2, 4, 2, 2, 1], (1, 2, 2))

    def test_3dt(self):
        # Maxpooling 2dt
        self._test_3dt([2, 4, 2, 2, 2, 1])

    def test_2d(self):
        # Maxpooling 2d
        self._test_2d([2, 2, 2, 1])
        self._test_2d([2, 2, 2, 1], (2, 2))
        # input shape: [batch, height, width, channel]
        self._test_2d_accuracy([1, 8, 12, 3], pool_size=(3, 2))

    def test_3d(self):
        # Maxpooling 3d
        self._test_3d([2, 16, 8, 4, 1])
        self._test_3d([2, 16, 8, 4, 1], (4, 2, 2))
        # input shape: [batch, height, width, depth, channel]
        self._test_3d_accuracy([2, 8, 6, 8, 2], pool_size=(3, 2, 2))


if __name__ == "__main__":
    # unittest.main()
    unittest.main()
