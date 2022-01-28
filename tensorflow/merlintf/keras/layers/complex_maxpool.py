import sys
import tensorflow as tf
import numpy as np
try:
    import optotf.maxpooling
except:
    print('optotf could not be imported')
import merlintf
import unittest
import six


def get(identifier):
    return MagnitudeMaxPooling(identifier)


def MagnitudeMaxPooling(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'MagnitudeMaxPool' + str(identifier).upper().replace('T', 't')
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
    elif op == 'MagnitudeMaxPool3Dt' or op == 'MagnitudeMaxPooling3Dt':
        return MagnitudeMaxPool3Dt
    else:
        raise ValueError(f"Selected operation '{op}' not implemented in complex convolutional")


class MagnitudeMaxPool(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True, argmax_index=False):
        super(MagnitudeMaxPool, self).__init__()
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        self.padding = padding
        self.alpha = 1  # magnitude ratio in real part
        self.beta = 1  # magnitude ratio in imag part
        self.optox = optox and (True if 'optotf.maxpooling' in sys.modules else False)  # True: execute Optox pooling; False: use TF pooling (not supported for all cases)
        self.argmax_index = argmax_index

    def call(self, x, **kwargs):  # default to TF
        xabs = merlintf.complex_abs(x)
        _, idx = tf.nn.max_pool_with_argmax(
            xabs, self.pool_size, self.strides, self.padding, include_batch_in_index=True)
        x_pool = tf.reshape(tf.gather(tf.reshape(x, shape=[-1, ]), idx), shape=idx.shape)

        return x_pool


class MagnitudeMaxPool1D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True):
        super(MagnitudeMaxPool1D, self).__init__(pool_size, strides, padding, optox)


class MagnitudeMaxPool2D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True, argmax_index=False):
        super(MagnitudeMaxPool2D, self).__init__(pool_size, strides, padding, optox, argmax_index)

    def call(self, x, **kwargs):
        if self.optox:
            if merlintf.iscomplextf(x):

                x_pool, x_pool_index = optotf.maxpooling.maxpooling2d(x, pool_size=self.pool_size, strides=self.strides,
                                                                      alpha=self.alpha, beta=self.beta,
                                                                      mode=self.padding)
                if self.argmax_index:
                    return x_pool, x_pool_index
                else:
                    return x_pool
            else:
                x_pool, x_pool_index = tf.nn.max_pool_with_argmax(x, ksize=self.pool_size, strides=self.strides,
                                                                  padding=self.padding)
                if self.argmax_index:
                    return x_pool, x_pool_index
                else:
                    return x_pool
        else:
            return super().call(x, **kwargs)


class MagnitudeMaxPool3D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True, argmax_index=False):
        super(MagnitudeMaxPool3D, self).__init__(pool_size, strides, padding, optox, argmax_index)

    def call(self, x, **kwargs):
        if self.optox:
            if merlintf.iscomplextf(x):
                x_pool, x_pool_index = optotf.maxpooling.maxpooling3d(x, pool_size=self.pool_size, strides=self.strides,
                                                                      alpha=self.alpha, beta=self.beta,
                                                                      mode=self.padding)

                if self.argmax_index:
                    return x_pool, x_pool_index
                else:
                    return x_pool
            else:
                if self.argmax_index:
                    x_pool, x_pool_index = tf.nn.max_pool3d(x, ksize=self.pool_size, strides=self.strides,
                                                            padding=self.padding)
                    return x_pool, x_pool_index
                else:
                    x_pool = tf.nn.max_pool3d(x, ksize=self.pool_size, strides=self.strides, padding=self.padding)
                    return x_pool
        else:
            if self.argmax_index or merlintf.iscomplextf(x):
                xabs = merlintf.complex_abs(x)
                xangle = merlintf.complex_angle(x)
                x_pool = tf.nn.max_pool3d(xabs, ksize=self.pool_size, strides=self.strides,
                                                        padding=self.padding)
                xangle_pool = tf.nn.max_pool3d(xangle, ksize=self.pool_size, strides=self.strides,
                                                        padding=self.padding)
                return merlintf.magpha2complex(tf.concat([x_pool, xangle_pool], -1))
            else:
                x_pool = tf.nn.max_pool3d(x, ksize=self.pool_size, strides=self.strides, padding=self.padding)
                return x_pool


class MagnitudeMaxPool2Dt(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True, argmax_index=False):
        super(MagnitudeMaxPool2Dt, self).__init__(pool_size, strides, padding, optox, argmax_index)

    def call(self, x, **kwargs):
        if self.optox:
            orig_shape = x.shape
            batched_shape = [x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
            x = tf.reshape(x, batched_shape)

            if merlintf.iscomplextf(x):
                x_pool, x_pool_index = optotf.maxpooling.maxpooling2d(x, pool_size=self.pool_size, strides=self.strides,
                                                                      alpha=self.alpha, beta=self.beta,
                                                                      mode=self.padding)

            else:
                x_pool, x_pool_index = tf.nn.max_pool_with_argmax(x, ksize=self.pool_size, strides=self.strides,
                                                                  padding=self.padding)

            pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
            x_pool = tf.reshape(x_pool, pooled_shape)
            if self.argmax_index:
                return x_pool, x_pool_index
            else:
                return x_pool
        else:
            orig_shape = x.shape
            batched_shape = [x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
            x = tf.reshape(x, batched_shape)

            xabs = merlintf.complex_abs(x)
            _, idx = tf.nn.max_pool_with_argmax(
                xabs, self.pool_size, self.strides, self.padding, include_batch_in_index=True)
            x_pool = tf.reshape(tf.gather(tf.reshape(x, shape=[-1, ]), idx), shape=idx.shape)

            pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
            x_pool = tf.reshape(x_pool, pooled_shape)
            if self.argmax_index:
                return x_pool, idx
            else:
                return x_pool


class MagnitudeMaxPool3Dt(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True, argmax_index=False):
        super(MagnitudeMaxPool3Dt, self).__init__(pool_size, strides, padding, optox, argmax_index)

    def call(self, x, **kwargs):  # only Optox supported
        orig_shape = x.shape
        rank = tf.rank(x)
        batched_shape = 0
        if rank == 6:
            batched_shape = [x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]]
        elif rank == 5:
            batched_shape = [x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
        x = tf.reshape(x, batched_shape)

        if merlintf.iscomplextf(x):
            x_pool, x_pool_index = optotf.maxpooling.maxpooling3d(x, pool_size=self.pool_size, strides=self.strides,
                                                                  alpha=self.alpha,
                                                                  beta=self.beta, mode=self.padding)
        else:
            # same as above
            x_pool, x_pool_index = optotf.maxpooling.maxpooling3d(x, pool_size=self.pool_size, strides=self.strides,
                                                                  alpha=self.alpha,
                                                                  beta=self.beta, mode=self.padding)

        pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
        x_pool = tf.reshape(x_pool, pooled_shape)

        if self.argmax_index:
            return x_pool, x_pool_index
        else:
            return x_pool


class TestMagnitudePool(unittest.TestCase):
    def _test(self, shape, pool_size=2, strides=2):
        # test tf.nn.max_pool_with_argmax
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool(pool_size, strides, optox=False)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_2dt(self, shape, pool_size=2, strides=2):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool2Dt(pool_size, strides, optox=False)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_2d(self, shape, pool_size=(2, 2), strides=(2, 2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool2D(pool_size, strides, optox=True)
        y = pool(x)
        magn = merlintf.complex_abs(y)

    def _test_3d(self, shape, pool_size=(2, 2, 2), strides=(2, 2, 2)):
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        pool = MagnitudeMaxPool3D(pool_size, strides, optox=True)
        y = pool(x)
        magn = merlintf.complex_abs(y)

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
        self._test_2dt([2, 4, 2, 2, 1])
    def test_3dt(self):
        # Maxpooling 2dt
        self._test_3dt([2, 4, 2, 2, 2, 1])

    def test_2d(self):
        # Maxpooling 2d
        self._test_2d([2, 2, 2, 1])
        self._test_2d([2, 2, 2, 1], (2, 2))
        # input shape: [batch, height, width, channel]
        self._test_2d_accuracy([1, 8, 12, 3], pool_size=(3, 2))

    def test_2(self):
        # Maxpooling 3d
        self._test_3d([2, 16, 8, 4, 1])
        self._test_3d([2, 16, 8, 4, 1], (4, 2, 2))
        # input shape: [batch, height, width, depth, channel]
        self._test_3d_accuracy([2, 8, 6, 8, 2], pool_size=(3, 2, 2))


if __name__ == "__main__":
    # unittest.test()
    unittest.main()
