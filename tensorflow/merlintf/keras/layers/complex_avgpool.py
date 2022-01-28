import sys
import tensorflow as tf
try:
    import optotf.averagepooling
except:
    print('optotf could not be imported')
import merlintf
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
        self.optox = optox and (True if 'optotf.averagepooling' in sys.modules else False)  # True: execute Optox pooling; False: use TF pooling (not supported for all cases)

    def call(self, x, **kwargs):  # default to TF
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

    def call(self, x, **kwargs):
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
            return super().call(x, **kwargs)


class MagnitudeAveragePool3D(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True):
        super(MagnitudeAveragePool3D, self).__init__(pool_size, strides, padding, optox)

    def call(self, x, **kwargs):
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
            return super().call(x, **kwargs)


class MagnitudeAveragePool2Dt(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True):
        super(MagnitudeAveragePool2Dt, self).__init__(pool_size, strides, padding, optox)

    def call(self, x, **kwargs):
        if self.optox:
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
        else:
            return super().call(x, **kwargs)


class MagnitudeAveragePool3Dt(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', optox=True):
        super(MagnitudeAveragePool3Dt, self).__init__(pool_size, strides, padding, optox)

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
