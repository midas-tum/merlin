import sys
import tensorflow as tf
try:
    import optotf.keras.maxpooling
except:
    print('optotf could not be imported')
import merlintf
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
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=None, optox=True, argmax_index=False):
        super(MagnitudeMaxPool, self).__init__()
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        self.padding = padding
        self.dilations_rate = dilations_rate
        self.alpha = 1  # magnitude ratio in real part
        self.beta = 1  # magnitude ratio in imag part
        self.optox = optox and (True if 'optotf.keras.maxpooling' in sys.modules else False)  # True: execute Optox pooling; False: use TF pooling (not supported for all cases)
        self.argmax_index = argmax_index

    def call(self, x, **kwargs):  # default to TF
        xabs = merlintf.complex_abs(x)
        _, idx = tf.nn.max_pool_with_argmax(
            xabs, self.pool_size, self.strides, self.padding, include_batch_in_index=True)
        x_pool = tf.reshape(tf.gather(tf.reshape(x, shape=[-1, ]), idx), shape=idx.shape)

        return x_pool


class MagnitudeMaxPool1D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=1, optox=True):
        super(MagnitudeMaxPool1D, self).__init__(pool_size, strides, padding, dilations_rate, optox)


class MagnitudeMaxPool2D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1), optox=True, argmax_index=False):
        super(MagnitudeMaxPool2D, self).__init__(pool_size, strides, padding, dilations_rate, optox, argmax_index)
        self.op = optotf.keras.maxpooling.Maxpooling2d(pool_size=self.pool_size, strides=self.strides,
                                                       alpha=self.alpha, beta=self.beta,
                                                       channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                       mode=self.padding, dilations_rate=self.dilations_rate, argmax=self.argmax_index)
        self.grad = optotf.keras.maxpooling.Maxpooling2d_grad_backward(pool_size=self.pool_size, strides=self.strides,
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
            return super(MagnitudeMaxPool2D, self).call(x, **kwargs)


class MagnitudeMaxPool3D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1), optox=True, argmax_index=False):
        super(MagnitudeMaxPool3D, self).__init__(pool_size, strides, padding, dilations_rate, optox, argmax_index)
        self.op = optotf.keras.maxpooling.Maxpooling3d(pool_size=self.pool_size, strides=self.strides,
                                                       alpha=self.alpha, beta=self.beta,
                                                       channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                       mode=self.padding, dilations_rate=self.dilations_rate, argmax=self.argmax_index)
        self.grad = optotf.keras.maxpooling.Maxpooling3d_grad_backward(pool_size=self.pool_size, strides=self.strides,
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
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1), optox=True, argmax_index=False):
        super(MagnitudeMaxPool2Dt, self).__init__(pool_size, strides, padding, dilations_rate, optox, argmax_index)
        self.op = optotf.keras.maxpooling.Maxpooling3d(pool_size=self.pool_size, strides=self.strides,
                                                       alpha=self.alpha, beta=self.beta,
                                                       channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                       mode=self.padding, dilations_rate=self.dilations_rate, argmax=argmax_index)
        self.grad = optotf.keras.maxpooling.Maxpooling3d_grad_backward(pool_size=self.pool_size, strides=self.strides,
                                                                       alpha=self.alpha, beta=self.beta,
                                                                       channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                                       dilations_rate=self.dilations_rate, mode=self.padding)
    def call(self, x, **kwargs):
        if self.optox:
            @tf.custom_gradient
            def optox_pool(x):
                def grad(dy):
                    return self.grad(x, dy)
                x_pool = self.op(x)
                return x_pool, grad
            return optox_pool(x)

        else:
            if tf.keras.backend.image_data_format() == 'channels_last':
                x = tf.transpose(x, [0, 4, 1, 2, 3])
            orig_shape = x.shape
            batched_shape = [x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
            x = tf.reshape(x, batched_shape)

            xabs = merlintf.complex_abs(x)
            _, idx = tf.nn.max_pool_with_argmax(
                xabs, self.pool_size, self.strides, self.padding, include_batch_in_index=True)
            x_pool = tf.reshape(tf.gather(tf.reshape(x, shape=[-1, ]), idx), shape=idx.shape)

            pooled_shape = [orig_shape[0], orig_shape[1], x_pool.shape[1], x_pool.shape[2], orig_shape[-1]]
            x_pool = tf.reshape(x_pool, pooled_shape)
            if tf.keras.backend.image_data_format() == 'channels_last':
                x_pool = tf.transpose(x_pool, [0, 2, 3, 4, 1])

            return x_pool


class MagnitudeMaxPool3Dt(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1, 1), optox=True, argmax_index=True):
        super(MagnitudeMaxPool3Dt, self).__init__(pool_size, strides, padding, dilations_rate, optox, argmax_index)
        self.op = optotf.keras.maxpooling.Maxpooling4d(pool_size=self.pool_size, strides=self.strides,
                                                       alpha=self.alpha, beta=self.beta,
                                                       channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                       mode=self.padding, dilations_rate=self.dilations_rate, argmax=self.argmax_index)
        self.grad = optotf.keras.maxpooling.Maxpooling4d_grad_backward(pool_size=self.pool_size, strides=self.strides,
                                                                       alpha=self.alpha, beta=self.beta,
                                                                       channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                                                                       dilations_rate=self.dilations_rate, mode=self.padding)

    def call(self, x, **kwargs):  # only Optox supported
        @tf.custom_gradient
        def optox_pool(x):
            def grad(dy):
                return self.grad(x, dy)
            x_pool = self.op(x)
            return x_pool, grad

        return optox_pool(x)


# Aliases
MagnitudeMaxPool4D = MagnitudeMaxPool3Dt