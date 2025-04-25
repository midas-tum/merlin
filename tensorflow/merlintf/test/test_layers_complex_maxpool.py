import unittest
import numpy as np
import tensorflow as tf
import merlintf
from merlintf.keras.layers.complex_maxpool import (
    MagnitudeMaxPool1D,
    MagnitudeMaxPool2D,
    MagnitudeMaxPool2Dt,
    MagnitudeMaxPool3D,
    MagnitudeMaxPool3Dt
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
        self._test((1, 4, 2), (2,), (2,), 'valid')
        self._test((1, 4, 2), (2,), (2,), 'same')
        self._test((1, 5, 2), (2,), (2,), 'valid')
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
            op = MagnitudeMaxPool1D(pool_size, strides, padding_mode)
            op_backend = tf.keras.layers.MaxPooling1D(pool_size, strides, padding_mode)
        elif len(shape) == 4:  # 2d
            op = MagnitudeMaxPool2D(pool_size, strides, padding_mode)
            op_backend = tf.keras.layers.MaxPooling2D(pool_size, strides, padding_mode)
        elif len(shape) == 5:  # 3d
            op = MagnitudeMaxPool3D(pool_size, strides, padding_mode)
            op_backend = tf.keras.layers.MaxPooling3D(pool_size, strides, padding_mode)
        elif len(shape) == 6:  # 4d
            op = MagnitudeMaxPool3Dt(pool_size, strides, padding_mode)

        out = op(x)
        out_backend = op_backend(merlintf.complex_abs(x))

        self.assertTrue(np.sum(np.abs(np.array(out.shape) - np.array(out_backend.shape))) == 0)

    def _test(self, shape, pool_size, strides, padding_mode, dilations_rate=(1, 1, 1, 1)):
        # test tf.nn.average_pool_with_argaverage
        x = merlintf.random_normal_complex(shape, dtype=tf.float32)

        if len(shape) == 3:  # 1d
            op = MagnitudeMaxPool1D(pool_size, strides, padding_mode)
        elif len(shape) == 4:  # 2d
            op = MagnitudeMaxPool2D(pool_size, strides, padding_mode)
        elif len(shape) == 5:  # 3d
            op = MagnitudeMaxPool3D(pool_size, strides, padding_mode)
        elif len(shape) == 6:  # 4d
            op = MagnitudeMaxPool3Dt(pool_size, strides, padding_mode)

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

        print('tf.math.abs(y) - x_abs', tf.math.abs(y) - x_abs)
        shape = x_abs.shape
        test_id = [np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[2]),
                   np.random.randint(0, shape[3]), np.random.randint(0, shape[4])]
        self.assertTrue((tf.math.abs(y)[test_id[0], test_id[1], test_id[2], test_id[3], test_id[4]] - x_abs[
            test_id[0], test_id[1], test_id[2], test_id[3], test_id[4]]) == 0.0)


if __name__ == "__main__":
    # unittest.main()
    unittest.main()
