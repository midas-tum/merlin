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
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=None, optox=True, layer_name='MagnitudeAvgPool', alpha=1, beta=1, **kwargs):
        super(MagnitudeAveragePool, self).__init__()
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        self.padding = padding
        self.dilations_rate = dilations_rate
        self.alpha = alpha  # magnitude ratio in real part
        self.beta = beta  # magnitude ratio in imag part
        self.layer_name = layer_name
        self.ceil_mode = True  # TF default
        self.optox = optox and (True if 'optotf.averagepooling' in sys.modules else False)  # True: execute Optox pooling; False: use TF pooling (not supported for all cases)

    def call(self, x, **kwargs):  # default to TF
        if self.optox and merlintf.iscomplex(x):
            out = self.op(x, pool_size=self.pool_size,
                          strides=self.strides,
                          alpha=self.alpha, beta=self.beta, name=self.layer_name,
                          dilations_rate=self.dilations_rate,
                          channel_first=tf.keras.backend.image_data_format() == 'channels_first',
                          mode=str.upper(self.padding), ceil_mode=self.ceil_mode)
            if x is not list:
                return out
            else:
                return tf.math.real(out), tf.math.imag(out)
        else:
            xabs = merlintf.complex_abs(x)
            pad = str.upper(self.padding)
            if len(xabs.shape) == 6:
                orig_shape = tf.shape(xabs)
                batched_shape = [orig_shape[0] * orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4], orig_shape[5]]
                xabs_b = tf.reshape(xabs, batched_shape)
                ps = self.pool_size[1:] if len(self.pool_size) == 4 else self.pool_size
                st = self.strides[1:] if len(self.strides) == 4 else self.strides
                x_pool_b = tf.nn.avg_pool(xabs_b, ps, st, pad, data_format='NDHWC')
                pooled_shape = [orig_shape[0], orig_shape[1], tf.shape(x_pool_b)[1], tf.shape(x_pool_b)[2], tf.shape(x_pool_b)[3], orig_shape[5]]
                return tf.reshape(x_pool_b, pooled_shape)
            elif len(xabs.shape) == 5:
                return tf.nn.avg_pool(xabs, self.pool_size, self.strides, pad, data_format='NDHWC')
            else:
                return tf.nn.avg_pool(xabs, self.pool_size, self.strides, pad)


class MagnitudeAveragePool1D(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, ), optox=True, layer_name='MagnitudeAvgPool1D', alpha=1, beta=1, **kwargs):
        super(MagnitudeAveragePool1D, self).__init__(pool_size, strides, padding, dilations_rate, optox, layer_name, alpha, beta, **kwargs)
        if self.optox:
            self.op = optotf.averagepooling.averagepooling1d


class MagnitudeAveragePool2D(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1), optox=True, layer_name='MagnitudeAvgPool2D', alpha=1, beta=1, **kwargs):
        super(MagnitudeAveragePool2D, self).__init__(pool_size, strides, padding, dilations_rate, optox, layer_name, alpha, beta, **kwargs)
        if self.optox:
            self.op = optotf.averagepooling.averagepooling2d


class MagnitudeAveragePool3D(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1), optox=True, layer_name='MagnitudeAvgPool3D', alpha=1, beta=1, **kwargs):
        super(MagnitudeAveragePool3D, self).__init__(pool_size, strides, padding, dilations_rate, optox, layer_name, alpha, beta, **kwargs)
        if self.optox:
            self.op = optotf.averagepooling.averagepooling3d


class MagnitudeAveragePool2Dt(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1), optox=True, layer_name='MagnitudeAvgPool2Dt', alpha=1, beta=1, **kwargs):
        super(MagnitudeAveragePool2Dt, self).__init__(pool_size, strides, padding, dilations_rate, optox, layer_name, alpha, beta, **kwargs)
        if self.optox:
            self.op = optotf.averagepooling.averagepooling3d


class MagnitudeAveragePool3Dt(MagnitudeAveragePool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1, 1), optox=True, layer_name='MagnitudeAvgPool3Dt', alpha=1, beta=1, **kwargs):
        super(MagnitudeAveragePool3Dt, self).__init__(pool_size, strides, padding, dilations_rate, optox, layer_name, alpha, beta, **kwargs)
        if self.optox:
            self.op = optotf.averagepooling.averagepooling4d


# Aliases
MagnitudeAveragePool4D = MagnitudeAveragePool3Dt