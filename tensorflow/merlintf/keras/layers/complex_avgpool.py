import sys
import tensorflow as tf
try:
    import optotf.keras.averagepooling
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
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=None, optox=True):
        super(MagnitudeAveragePool, self).__init__()
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        self.padding = padding
        self.dilations_rate = dilations_rate
        self.alpha = 1  # magnitude ratio in real part
        self.beta = 1  # magnitude ratio in imag part
        self.optox = optox and (True if 'optotf.keras.averagepooling' in sys.modules else False)  # True: execute Optox pooling; False: use TF pooling (not supported for all cases)

    def call(self, x, **kwargs):  # default to TF
        xabs = merlintf.complex_abs(x)
        x_pool = tf.nn.avg_pool(
            xabs, self.pool_size, self.strides, self.padding)
        return x_pool


class MagnitudeAveragePool1D(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=1, optox=True):
        super(MagnitudeAveragePool1D, self).__init__(pool_size, strides, padding, dilations_rate, optox)


class MagnitudeAveragePool2D(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1), optox=True):
        super(MagnitudeAveragePool2D, self).__init__(pool_size, strides, padding, dilations_rate, optox)
        self.op = optotf.keras.averagepooling.Averagepooling2d(pool_size=self.pool_size, strides=self.strides,
                                                               alpha=self.alpha, beta=self.beta,
                                                               channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                               dilations_rate=self.dilations_rate, mode=self.padding)
        self.grad = optotf.keras.averagepooling.Averagepooling2d_grad_backward(pool_size=self.pool_size, strides=self.strides,
                                                               alpha=self.alpha, beta=self.beta,
                                                               channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                               dilations_rate=self.dilations_rate, mode=self.padding)

    def call(self, x, **kwargs):
        if self.optox:
            @tf.custom_gradient
            def optox_pool(x):
                def grad(dy):
                    return self.grad(x, dy)
                return self.op(x), grad

            return optox_pool(x)
        else:
            return super().call(x, **kwargs)


class MagnitudeAveragePool3D(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1), optox=True):
        super(MagnitudeAveragePool3D, self).__init__(pool_size, strides, padding, dilations_rate, optox)
        self.op = optotf.keras.averagepooling.Averagepooling3d(pool_size=self.pool_size, strides=self.strides,
                                                               alpha=self.alpha, beta=self.beta,
                                                               channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                               dilations_rate=self.dilations_rate, mode=self.padding)
        self.grad = optotf.keras.averagepooling.Averagepooling3d_grad_backward(pool_size=self.pool_size,
                                                                               strides=self.strides,
                                                                               alpha=self.alpha, beta=self.beta,
                                                                               channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                                               dilations_rate=self.dilations_rate,
                                                                               mode=self.padding)

    def call(self, x, **kwargs):
        if self.optox:
            @tf.custom_gradient
            def optox_pool(x):
                def grad(dy):
                    return self.grad(x, dy)

                return self.op(x), grad

            return optox_pool(x)
        else:
            return super().call(x, **kwargs)


class MagnitudeAveragePool2Dt(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1), optox=True):
        super(MagnitudeAveragePool2Dt, self).__init__(pool_size, strides, padding, dilations_rate, optox)
        self.op = optotf.keras.averagepooling.Averagepooling3d(pool_size=self.pool_size, strides=self.strides,
                                                               alpha=self.alpha, beta=self.beta,
                                                               channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                               dilations_rate=self.dilations_rate, mode=self.padding)
        self.grad = optotf.keras.averagepooling.Averagepooling3d_grad_backward(pool_size=self.pool_size,
                                                                               strides=self.strides,
                                                                               alpha=self.alpha, beta=self.beta,
                                                                               channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                                               dilations_rate=self.dilations_rate,
                                                                               mode=self.padding)

    def call(self, x, **kwargs):
        if self.optox:
            @tf.custom_gradient
            def optox_pool(x):
                def grad(dy):
                    return self.grad(x, dy)

                return self.op(x), grad

            return optox_pool(x)
        else:
            return super().call(x, **kwargs)


class MagnitudeAveragePool3Dt(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1, 1), optox=True):
        super(MagnitudeAveragePool3Dt, self).__init__(pool_size, strides, padding, dilations_rate, optox)
        self.op = optotf.keras.averagepooling.Averagepooling4d(pool_size=self.pool_size, strides=self.strides,
                                                               alpha=self.alpha, beta=self.beta,
                                                               channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                               dilations_rate=self.dilations_rate, mode=self.padding)
        self.grad = optotf.keras.averagepooling.Averagepooling4d_grad_backward(pool_size=self.pool_size,
                                                                               strides=self.strides,
                                                                               alpha=self.alpha, beta=self.beta,
                                                                               channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                                               dilations_rate=self.dilations_rate,
                                                                               mode=self.padding)

    def call(self, x, **kwargs):  # only Optox supported
        @tf.custom_gradient
        def optox_pool(x):
            def grad(dy):
                return self.grad(x, dy)

            x_pool = self.op(x)
            return x_pool, grad

        return optox_pool(x)

# Aliases
MagnitudeAveragePool4D = MagnitudeAveragePool3Dt