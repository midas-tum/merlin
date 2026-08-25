import sys
import tensorflow as tf
try:
    import optotf.maxpooling
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
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=None, optox=True, argmax_index=False, layer_name='MagnitudeMaxPool', alpha=1, beta=1, **kwargs):
        super(MagnitudeMaxPool, self).__init__()
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        self.padding = padding
        self.dilations_rate = dilations_rate
        self.alpha = alpha  # magnitude ratio in real part
        self.beta = beta  # magnitude ratio in imag part
        self.ceil_mode = True  # TF default
        self.optox = optox and (True if 'optotf.maxpooling' in sys.modules else False)  # True: execute Optox pooling; False: use TF pooling (not supported for all cases)
        self.argmax_index = argmax_index
        self.layer_name = layer_name

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
                x_pool_b = tf.nn.max_pool(xabs_b, ps, st, pad, data_format='NDHWC')
                pooled_shape = [orig_shape[0], orig_shape[1], tf.shape(x_pool_b)[1], tf.shape(x_pool_b)[2], tf.shape(x_pool_b)[3], orig_shape[5]]
                return tf.reshape(x_pool_b, pooled_shape)
            elif len(xabs.shape) == 5:
                return tf.nn.max_pool(xabs, self.pool_size, self.strides, pad, data_format='NDHWC')
            else:
                return tf.nn.max_pool(xabs, self.pool_size, self.strides, pad)



class MagnitudeMaxPool1D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, ), optox=True, argmax_index=False, layer_name='MagnitudeMaxPool1D', alpha=1, beta=1, **kwargs):
        super(MagnitudeMaxPool1D, self).__init__(pool_size, strides, padding, dilations_rate, optox, argmax_index, layer_name, alpha, beta, **kwargs)
        if self.optox:
            self.op = optotf.maxpooling.maxpooling1d


class MagnitudeMaxPool2D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1), optox=True, argmax_index=False, layer_name='MagnitudeMaxPool2D', alpha=1, beta=1, **kwargs):
        super(MagnitudeMaxPool2D, self).__init__(pool_size, strides, padding, dilations_rate, optox, argmax_index, layer_name, alpha, beta, **kwargs)
        if self.optox:
            self.op = optotf.maxpooling.maxpooling2d


class MagnitudeMaxPool3D(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1), optox=True, argmax_index=False, layer_name='MagnitudeMaxPool3D', alpha=1, beta=1, **kwargs):
        super(MagnitudeMaxPool3D, self).__init__(pool_size, strides, padding, dilations_rate, optox, argmax_index, layer_name, alpha, beta, **kwargs)
        if self.optox:
            self.op = optotf.maxpooling.maxpooling3d


class MagnitudeMaxPool2Dt(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1), optox=True, argmax_index=False, layer_name='MagnitudeMaxPool2Dt', alpha=1, beta=1, **kwargs):
        super(MagnitudeMaxPool2Dt, self).__init__(pool_size, strides, padding, dilations_rate, optox, argmax_index, layer_name, alpha, beta, **kwargs)
        if self.optox:
            self.op = optotf.maxpooling.maxpooling3d


class MagnitudeMaxPool3Dt(MagnitudeMaxPool):
    def __init__(self, pool_size, strides=None, padding='SAME', dilations_rate=(1, 1, 1, 1), optox=True, argmax_index=False, layer_name='MagnitudeMaxPool3Dt', alpha=1, beta=1, **kwargs):
        super(MagnitudeMaxPool3Dt, self).__init__(pool_size, strides, padding, dilations_rate, optox, argmax_index, layer_name, alpha, beta, **kwargs)
        if self.optox:
            self.op = optotf.maxpooling.maxpooling4d


# Aliases
MagnitudeMaxPool4D = MagnitudeMaxPool3Dt