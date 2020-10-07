import tensorflow as tf
import optotf.pad2d

import numpy as np
import unittest

from .complex_init import complex_initializer
from .complex_conv import (
    complex_conv2d,
    complex_conv2d_transpose,
    complex_conv2d_real_weight,
    complex_conv2d_real_weight_transpose
)
from .complex_pad import (
    complex_pad2d,
    complex_pad2d_transpose
)

__all__ = ['ComplexPadConv2d',
           'ComplexPadConvScale2d',
           'ComplexPadConvScaleTranspose2d',
           'ComplexPadConvRealWeight2d']

class ComplexPadConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, bias=False, 
                 zero_mean=True, bound_norm=True, pad=True):
        super(ComplexPadConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        def get_param(x):
            if isinstance(x, int):
                return (x, x)
            elif isinstance(x, tuple):
                assert len(x) == 2
                return x
            else:
                raise ValueError('expects tuple or int')
        
        self.kernel_size = get_param(kernel_size)
        self.stride = get_param(stride)
        self.dilation = get_param(dilation)
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.pad = pad
        self.weight_shape =  self.kernel_size + (in_channels, out_channels, 2,)
        self.bias_shape = (out_channels, 2)
        self.use_bias = bias

    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.GlorotNormal() # <--- TODO change to complex normalization!
        self._weight = self.add_weight('weight',
                                      shape=self.weight_shape,
                                      initializer=initializer,
                                      dtype=tf.float32
                                      )
        self._bias   = self.add_weight('bias',
                                      shape=self.bias_shape,
                                      initializer=tf.keras.initializers.Zeros(),
                                      dtype=tf.float32) if self.use_bias else None
        
        # define the weight constraints
        if self.zero_mean or self.bound_norm:
            self._weight.reduction_dim = (0, 1, 2, 3)
            reduction_dim_mean = (0, 1, 2,)

            def l2_proj(weight, surface=False):
                tmp = weight
                # reduce the mean
                if self.zero_mean:
                    tmp = tmp - tf.reduce_mean(tmp, reduction_dim_mean, True)
                # normalize by the l2-norm
                if self.bound_norm:
                    norm = tf.math.sqrt(tf.reduce_sum(tmp ** 2, self._weight.reduction_dim, True))
                    if surface:
                        tmp = tmp / tf.math.maximum(norm, tf.ones_like(norm)*1e-9)
                    else:
                        tmp = tmp / tf.math.maximum(norm, tf.ones_like(norm))
                return tmp

            self._weight.proj = l2_proj
            self._weight.assign(l2_proj(self._weight, True))

    @property
    def weight(self):
        return tf.complex(self._weight[...,0], self._weight[...,1])

    @property
    def bias(self):
        return tf.complex(self._bias[...,0], self._bias[...,1])
        
    def call(self, x):
        # then pad
        pad = [w//2 for w in self.weight.shape[:2]]

        if self.pad and any(pad):
            x = complex_pad2d(x, (pad[1],pad[1],pad[0],pad[0]), mode='symmetric')

        # compute the convolution
        x = complex_conv2d(x, self.weight, padding="VALID", strides=(1,) + self.stride + (1,), dilations=(1,) + self.dilation + (1,))

        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        return x

    def backward(self, x, output_shape=None):
        # zero pad
        pad = [w//2 for w in self.weight.shape[:2]]
        ksz = [w for w in self.weight.shape[:2]]

        # determine the output padding
        if not output_shape is None:
            output_shape = tf.unstack(output_shape)
            output_shape[-1] = self.in_channels
            output_padding = [output_shape[i+1] - ((x.shape[i+1]-1)*self.stride[i]+1) for i in range(2)]
        else:
            output_shape = [tf.shape(x)[0], 1, 1, self.in_channels]
            output_padding = [0,0]

        # construct output shape
        output_shape = [(tf.shape(x)[i] - 1)*self.stride[i-1] + self.dilation[i-1] * (ksz[i-1] - 1) + output_padding[i-1] + 1 if (i > 0 and i < 3) else output_shape[i] for i in range(4) ]
        output_shape = tf.stack(output_shape)

        # zero pad input
        pad_k = [w//2 for w in self.kernel_size]
        tf_pad = [[0,0,],] + \
                 [[pad_k[i] + output_padding[i]//2, pad_k[i] + output_padding[i]//2 + np.mod(output_padding[i],2)] for i in range(2)] + \
                 [[0,0,],]
        x = tf.pad(x, tf_pad)


        # remove bias
        if self.use_bias:
            x = tf.nn.bias_add(x, -self.bias)

        # compute the transpose convolution
        x = complex_conv2d_transpose(x, self.weight, output_shape, padding='SAME', strides=self.stride, dilations=self.dilation)

        # transpose padding
        if self.pad and any(pad):
            x = complex_pad2d_transpose(x, (pad[1],pad[1],pad[0],pad[0]), mode='symmetric')
        return x

    def extra_repr(self):
        s = "({out_channels}, {in_channels}, {kernel_size}),"
        if self.stride != 1:
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if not self.bias is None:
            s += ", bias=True"
        if self.zero_mean:
            s += ", zero_mean={zero_mean}"
        if self.bound_norm:
            s += ", bound_norm={bound_norm}"
        return s.format(**self.__dict__)


class ComplexPadConvScale2d(ComplexPadConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(ComplexPadConvScale2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, dilation=1, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)
        # create the convolution kernel
        if self.stride[0] > 1 :
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            np_k = np.reshape(np_k, (5, 5, 1, 1))
            self.blur = tf.Variable(initial_value=tf.convert_to_tensor(np_k), trainable=False)

    @property
    def weight(self):
        weight = super().weight
        if self.stride[0] > 1 :
            weight = tf.reshape(weight, self.kernel_size + (self.out_channels*self.in_channels, 1))
            weight = tf.transpose(weight, (2, 0, 1, 3))
            for i in range(self.stride[0]//2):
                weight = tf.pad(weight, [[0,0], [5,5], [5,5], [0,0]], 'CONSTANT')
                weight = complex_conv2d_real_weight(weight, self.blur, padding="SAME", strides=self.stride, dilations=self.dilation)
            weight = tf.transpose(weight, (1, 2, 0, 3))
            weight = tf.reshape(weight, (self.kernel_size[0]+2*self.stride[0], self.kernel_size[1]+2*self.stride[1], self.in_channels, self.out_channels))
        return weight


class ComplexPadConvScaleTranspose2d(ComplexPadConvScale2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(ComplexPadConvScaleTranspose2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

    def call(self, x, output_shape=None):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().call(x)


class ComplexPadConvRealWeight2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, bias=False, 
                 zero_mean=True, bound_norm=True, pad=True):
        super(ComplexPadConvRealWeight2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        def get_param(x):
            if isinstance(x, int):
                return (x, x)
            elif isinstance(x, tuple):
                assert len(x) == 2
                return x
            else:
                raise ValueError('expects tuple or int')
        
        self.kernel_size = get_param(kernel_size)
        self.stride = get_param(stride)
        self.dilation = get_param(dilation)
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.pad = pad
        self.weight_shape =  self.kernel_size + (in_channels, out_channels,)
        self.bias_shape = (out_channels,)
        self.use_bias = bias

    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.GlorotNormal()
        self._weight = self.add_weight('weight',
                                      shape=self.weight_shape,
                                      initializer=initializer,
                                      dtype=tf.float32
                                      )
        self._bias   = self.add_weight('bias',
                                      shape=self.bias_shape,
                                      initializer=tf.keras.initializers.Zeros(),
                                      dtype=tf.float32) if self.use_bias else None
        
        # define the weight constraints
        if self.zero_mean or self.bound_norm:
            self._weight.reduction_dim = (0, 1, 2,)
            reduction_dim_mean = (0, 1, 2,)

            def l2_proj(weight, surface=False):
                tmp = weight
                # reduce the mean
                if self.zero_mean:
                    tmp = tmp - tf.reduce_mean(tmp, reduction_dim_mean, True)
                # normalize by the l2-norm
                if self.bound_norm:
                    norm = tf.math.sqrt(tf.reduce_sum(tmp ** 2, self._weight.reduction_dim, True))
                    if surface:
                        tmp = tmp / tf.math.maximum(norm, tf.ones_like(norm)*1e-9)
                    else:
                        tmp = tmp / tf.math.maximum(norm, tf.ones_like(norm))
                return tmp

            self._weight.proj = l2_proj
            self._weight.assign(l2_proj(self._weight, True))

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias
        
    def call(self, x):
        # then pad
        pad = [w//2 for w in self.weight.shape[:2]]

        if self.pad and any(pad):
            x = complex_pad2d(x, (pad[1],pad[1],pad[0],pad[0]), mode='symmetric')

        # compute the convolution
        x = complex_conv2d_real_weight(x, self.weight, padding="VALID", strides=(1,) + self.stride + (1,), dilations=(1,) + self.dilation + (1,))

        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        return x

    def backward(self, x, output_shape=None):
        # zero pad
        pad = [w//2 for w in self.weight.shape[:2]]
        ksz = [w for w in self.weight.shape[:2]]

        # determine the output padding
        if not output_shape is None:
            output_shape = tf.unstack(output_shape)
            output_shape[-1] = self.in_channels
            output_padding = [output_shape[i+1] - ((x.shape[i+1]-1)*self.stride[i]+1) for i in range(2)]
        else:
            output_shape = [tf.shape(x)[0], 1, 1, self.in_channels]
            output_padding = [0,0]

        # construct output shape
        output_shape = [(tf.shape(x)[i] - 1)*self.stride[i-1] + self.dilation[i-1] * (ksz[i-1] - 1) + output_padding[i-1] + 1 if (i > 0 and i < 3) else output_shape[i] for i in range(4) ]
        output_shape = tf.stack(output_shape)

        # zero pad input
        pad_k = [w//2 for w in self.kernel_size]
        tf_pad = [[0,0,],] + \
                 [[pad_k[i] + output_padding[i]//2, pad_k[i] + output_padding[i]//2 + np.mod(output_padding[i],2)] for i in range(2)] + \
                 [[0,0,],]
        x = tf.pad(x, tf_pad)


        # remove bias
        if self.use_bias:
            x = tf.nn.bias_add(x, -self.bias)

        # compute the transpose convolution
        x = complex_conv2d_real_weight_transpose(x, self.weight, output_shape, padding='SAME', strides=self.stride, dilations=self.dilation)

        # transpose padding
        if self.pad and any(pad):
            x = complex_pad2d_transpose(x, (pad[1],pad[1],pad[0],pad[0]), mode='symmetric')
        return x

    def extra_repr(self):
        s = "({out_channels}, {in_channels}, {kernel_size}),"
        if self.stride != 1:
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if not self.bias is None:
            s += ", bias=True"
        if self.zero_mean:
            s += ", zero_mean={zero_mean}"
        if self.bound_norm:
            s += ", bound_norm={bound_norm}"
        return s.format(**self.__dict__)

class ComplexPadConv2dTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, M, N, nf_in]
        
        model = ComplexPadConv2d(nf_in, nf_out, kernel_size=3)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)

        with tf.GradientTape() as g:
            g.watch(x)
            Kx = model(x)
            loss = 0.5 * tf.reduce_sum(tf.math.conj(Kx) * Kx)
        grad_x = g.gradient(loss, x)
        x_autograd = grad_x.numpy()

        KHKx = model.backward(Kx, output_shape=x.shape)
        x_bwd = KHKx.numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_adjoint(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 1
        nf_out = 32
        shape = [nBatch, M, N, nf_in]

        model = ComplexPadConv2d(nf_in, nf_out, kernel_size=3)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)

        y =  tf.complex(tf.random.normal(Kx.shape), tf.random.normal(Kx.shape))
        KHy = model.backward(y, x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)

    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConv2d(nf_in, nf_out, kernel_size=3)
        model.build(())
        np_weight = model.weights[0].numpy()
        reduction_dim = model.weights[0].reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))

        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

class ComplexPadConvRealWeight2dTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, M, N, nf_in]
        
        model = ComplexPadConvRealWeight2d(nf_in, nf_out, kernel_size=3)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)

        with tf.GradientTape() as g:
            g.watch(x)
            Kx = model(x)
            loss = 0.5 * tf.reduce_sum(tf.math.conj(Kx) * Kx)
        grad_x = g.gradient(loss, x)
        x_autograd = grad_x.numpy()

        KHKx = model.backward(Kx, output_shape=x.shape)
        x_bwd = KHKx.numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_adjoint(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 1
        nf_out = 32
        shape = [nBatch, M, N, nf_in]

        model = ComplexPadConvRealWeight2d(nf_in, nf_out, kernel_size=3)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)

        y =  tf.complex(tf.random.normal(Kx.shape), tf.random.normal(Kx.shape))
        KHy = model.backward(y, x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)

    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConvRealWeight2d(nf_in, nf_out, kernel_size=3)
        model.build(())
        np_weight = model.weights[0].numpy()
        reduction_dim = model.weights[0].reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))

        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

class ComplexPadConvScale2dTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, M, N, nf_in]

        model = ComplexPadConvScale2d(nf_in, nf_out, kernel_size=3, stride=2)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        
        with tf.GradientTape() as g:
            g.watch(x)
            Kx = model(x)
            loss = 0.5 * tf.reduce_sum(tf.math.conj(Kx) * Kx)
        grad_x = g.gradient(loss, x)
        x_autograd = grad_x.numpy()

        KHKx = model.backward(Kx, output_shape=x.shape)
        x_bwd = KHKx.numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_adjoint(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, M, N, nf_in]

        model = ComplexPadConvScale2d(nf_in, nf_out, kernel_size=3, stride=2)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)

        y =  tf.complex(tf.random.normal(Kx.shape), tf.random.normal(Kx.shape))
        KHy = model.backward(y, output_shape=x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)

if __name__ == "__main__":
    unittest.test()
