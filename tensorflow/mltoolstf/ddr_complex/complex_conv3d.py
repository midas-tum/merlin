import tensorflow as tf
import optotf.pad3d

import numpy as np
import unittest

from complex_init import complex_initializer
from complex_conv import complex_conv3d, complex_conv3d_transpose, complex_conv3d_real_weight
from complex_pad import complex_pad3d, complex_pad3d_transpose

__all__ = ['ComplexConv3d',
           'ComplexConvScale3d',
           'ComplexConvScaleTranspose3d',
           'ComplexConv2dt']

class ComplexConv3d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, bias=False, 
                 zero_mean=True, bound_norm=True, pad=True):
        super(ComplexConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        def get_param(x):
            if isinstance(x, int):
                return (x, x, x)
            elif isinstance(x, tuple):
                assert len(x) == 3
                return x
            else:
                raise ValueError('expects tuple or int')
        
        self.kernel_size = get_param(kernel_size)
        self.stride = get_param(stride)
        self.dilation = get_param(dilation)
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.pad = pad
        self.weight_shape =  self.kernel_size + (in_channels, out_channels)
        self.bias_shape = (out_channels,)
        self.use_bias = bias

    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.GlorotNormal
        self.weight = self.add_weight('weight',
                                      shape=self.weight_shape,
                                      initializer=complex_initializer(initializer),
                                      dtype=tf.complex64
                                      )
        self.bias   = self.add_weight('bias',
                                      shape=self.bias_shape,
                                      initializer=complex_initializer(tf.keras.initializers.Zeros),
                                      dtype=tf.complex64) if self.use_bias else None

        # define the weight constraints
        if self.zero_mean or self.bound_norm:
            self.weight.reduction_dim = (0, 1, 2, 3)
            reduction_dim_mean = (0, 1, 2, 3)

            def l2_proj(weight, surface=False):
                tmp = weight
                # reduce the mean
                if self.zero_mean:
                    tmp = tmp - tf.reduce_mean(tmp, reduction_dim_mean, True)
                # normalize by the l2-norm
                if self.bound_norm:
                    norm = tf.cast(
                        tf.math.sqrt(tf.reduce_sum(tf.math.conj(tmp)*tmp, self.weight.reduction_dim, True)), tf.float32)
                    if surface:
                        tmp = tmp / tf.cast(tf.math.maximum(norm, tf.ones_like(norm)*1e-9), tf.complex64)
                    else:
                        tmp = tmp / tf.cast(tf.math.maximum(norm, tf.ones_like(norm)), tf.complex64)
                return tmp

            self.weight.proj = l2_proj
            self.weight.assign(l2_proj(self.weight, True))

    def get_weight(self):
        return self.weight

    def call(self, x):
        weight = self.get_weight()

        # then pad
        pad = [w//2 for w in weight.shape[:3]]

        if self.pad and any(pad):
            x = complex_pad3d(x, (pad[2],pad[2],pad[1],pad[1],pad[0],pad[0]), mode='symmetric')

        # compute the ComplexConvolution
        x = complex_conv3d(x, weight, padding="VALID", strides=(1,) + self.stride + (1,), dilations=(1,) + self.dilation + (1,))

        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        return x

    def backward(self, x, output_shape=None):
        weight = self.get_weight()

        # zero pad
        pad = [w//2 for w in weight.shape[:3]]
        ksz = [w for w in weight.shape[:3]]

        # determine the output padding
        if not output_shape is None:
            output_shape = list(output_shape)
            output_shape[-1] = self.in_channels
            output_padding = [output_shape[i+1] - ((x.shape[i+1]-1)*self.stride[i]+1) for i in range(3)]
        else:
            output_shape = [x.shape[0], 1, 1, 1, self.in_channels]
            output_padding = [0,0,0]

        
        # construct output shape

        output_shape = [(x.shape[i] - 1)*self.stride[i-1] + self.dilation[i-1] * (ksz[i-1] - 1) + output_padding[i-1] + 1 if (i > 0 and i < 4) else output_shape[i] for i in range(5) ]

        # zero pad input
        pad_k = [w//2 for w in self.kernel_size]
        tf_pad = [[0,0,],] + \
                 [[pad_k[i] + output_padding[i]//2, pad_k[i] + output_padding[i]//2 + np.mod(output_padding[i],2)] for i in range(3)] + \
                 [[0,0,],]
        x = tf.pad(x, tf_pad)

        # remove bias
        if self.use_bias:
            x = tf.nn.bias_add(x, -self.bias)

        # compute the transpose ComplexConvolution
        x = complex_conv3d_transpose(x, weight, output_shape, padding='SAME', strides=(1,) + self.stride + (1,), dilations=(1,) + self.dilation + (1,))

        # transpose padding
        if self.pad and any(pad):
            x = complex_pad3d_transpose(x, (pad[2],pad[2],pad[1],pad[1],pad[0],pad[0]), mode='symmetric')
        return x

class ComplexConvScale3d(ComplexConv3d):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=(1,2,2), bias=False, zero_mean=False, bound_norm=False):
        super(ComplexConvScale3d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, dilation=1, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)
        assert self.stride[0] == 1
        # create the ComplexConvolution kernel
        if self.stride[1] > 1 :
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            np_k = np.reshape(np_k, (1, 5, 5, 1, 1))
            self.blur = tf.Variable(initial_value=tf.convert_to_tensor(np_k), trainable=False)

    def get_weight(self):
        weight = super().get_weight()
        if self.stride[1] > 1 :
            weight = tf.reshape(weight, self.kernel_size + (self.out_channels*self.in_channels, 1))
            weight = tf.transpose(weight, (3, 0, 1, 2, 4))
            for i in range(self.stride[1]//2):
                weight = tf.pad(weight, [[0,0], [0,0], [5,5], [5,5], [0,0]], 'CONSTANT')
                weight = complex_conv3d_real_weight(weight, self.blur, padding="SAME", strides=(1,) + self.stride + (1,), dilations=(1,) + self.dilation + (1,))
            weight = tf.transpose(weight, (1, 2, 3, 0, 4))
            weight = tf.reshape(weight, (self.kernel_size[0], self.kernel_size[1]+2*self.stride[1], self.kernel_size[2]+2*self.stride[2], self.in_channels, self.out_channels))
        return weight


class ComplexConvScaleTranspose3d(ComplexConvScale3d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(1,2,2), bias=False, zero_mean=False, bound_norm=False):
        super(ComplexConvScaleTranspose3d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, dilation=1, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

    def call(self, x, output_shape=None):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().call(x)


class ComplexConv2dt(tf.keras.layers.Layer):
    def __init__(self, in_channels, inter_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, bias=False, zero_mean=True, bound_norm=True):
        super(ComplexConv2dt, self).__init__()

        if stride > 2:
            conv_module = ComplexConvScale3d
        else:
            conv_module = ComplexConv3d

        self.conv_xy = conv_module(in_channels,
                    inter_channels,
                    kernel_size=(1, *kernel_size[1:]),
                    stride=stride,
                    bias=bias,
                    zero_mean=zero_mean,
                    bound_norm=bound_norm)

        self.conv_t = ComplexConv3d(inter_channels,
                 out_channels,
                 kernel_size=(kernel_size[0], 1, 1),
                 bias=bias,
                 zero_mean=False,
                 bound_norm=bound_norm)

    def call(self, x):
        x_sp = self.conv_xy(x)
        x_t = self.conv_t(x_sp)
        return x_t  

    def backward(self, x, output_shape=None):
        xT_t = self.conv_t.backward(x, output_shape)
        xT_sp = self.conv_xy.backward(xT_t, output_shape)
        return xT_sp

class ComplexConv2dtTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 128
        N = 128
        D = 24
        nf_in = 2
        nf_out = 32
        shape = [nBatch, D, M, N, nf_in]
        ksz = (3,5,5)
        nf_inter = np.ceil((nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] + nf_out * ksz[0])).astype(np.int32)
        model = ComplexConv2dt(nf_in, nf_out, nf_inter, kernel_size=ksz)
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
        M = 128
        N = 128
        D = 24
        nf_in = 2
        nf_out = 32
        shape = [nBatch, D, M, N, nf_in]

        ksz = (3,5,5)
        nf_inter = np.ceil((nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] + nf_out * ksz[0])).astype(np.int32)

        model = ComplexConv2dt(nf_in, nf_out, nf_inter, kernel_size=ksz)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)

        y = tf.complex(tf.random.normal(Kx.shape), tf.random.normal(Kx.shape))
        KHy = model.backward(y, x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)
class ComplexConv3dTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 128
        N = 128
        D = 24
        nf_in = 10
        nf_out = 32
        shape = [nBatch, D, M, N, nf_in]

        model = ComplexConv3d(nf_in, nf_out, kernel_size=(3,5,5))
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
        M = 128
        N = 128
        D = 24
        nf_in = 10
        nf_out = 32
        shape = [nBatch, D, M, N, nf_in]

        model = ComplexConv3d(nf_in, nf_out, kernel_size=(3,5,5))
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)

        y = tf.complex(tf.random.normal(Kx.shape), tf.random.normal(Kx.shape))
        KHy = model.backward(y, x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)

    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = ComplexConv3d(nf_in, nf_out, kernel_size=(3,5,5))
        model.build(())
        np_weight = model.weights[0].numpy()
        reduction_dim = model.weights[0].reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))

        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

class ComplexConvScale3dTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 256
        N = 256
        D = 24
        nf_in = 10
        nf_out = 32
        shape = [nBatch, D, M, N, nf_in]

        model = ComplexConvScale3d(nf_in, nf_out, kernel_size=3, stride=(1,2,2))
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
        M = 128
        N = 128
        D = 24
        nf_in = 10
        nf_out = 32
        shape = [nBatch, D, M, N, nf_in]
        
        model = ComplexConvScale3d(nf_in, nf_out, kernel_size=3, stride=(1,2,2))
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)

        y = tf.complex(tf.random.normal(Kx.shape), tf.random.normal(Kx.shape))
        KHy = model.backward(y, output_shape=x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)

if __name__ == "__main__":
    unittest.test()
