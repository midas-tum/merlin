import tensorflow as tf
import optotf.pad2d

import numpy as np
import unittest

from complex_init import complex_initializer

__all__ = ['ComplexConv2d',
           'ComplexConvScale2d',
           'ComplexConvScaleTranspose2d']

class ComplexConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, bias=False, 
                 zero_mean=True, bound_norm=True, pad=True):
        super(ComplexConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.pad = pad
        self.weight_shape =  (kernel_size, kernel_size, in_channels, out_channels)
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
            self.weight.reduction_dim = (0, 1, 2,)
            reduction_dim_mean = (0, 1, 2,)

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

    def complex_pad2d(self, x, pad, mode='symmetric'):
        xp_re = optotf.pad2d.pad2d(tf.math.real(x), pad, mode=mode)
        xp_im = optotf.pad2d.pad2d(tf.math.imag(x), pad, mode=mode)

        return tf.complex(xp_re, xp_im)

    def complex_pad2d_transpose(self, x, pad, mode='symmetric'):
        xp_re = optotf.pad2d.pad2d_transpose(tf.math.real(x), pad, mode=mode)
        xp_im = optotf.pad2d.pad2d_transpose(tf.math.imag(x), pad, mode=mode)

        return tf.complex(xp_re, xp_im)

    def complex_conv2d(self, x, weight, padding="VALID", strides=1, dilations=1):
        xre = tf.math.real(x)
        xim = tf.math.imag(x)

        wre = tf.math.real(weight)
        wim = tf.math.imag(weight)

        conv_rr = tf.nn.conv2d(xre, wre, padding=padding, strides=strides, dilations=dilations)
        conv_ii = tf.nn.conv2d(xim, wim, padding=padding, strides=strides, dilations=dilations)
        conv_ri = tf.nn.conv2d(xre, wim, padding=padding, strides=strides, dilations=dilations)
        conv_ir = tf.nn.conv2d(xim, wre, padding=padding, strides=strides, dilations=dilations)

        conv_re = conv_rr - conv_ii
        conv_im = conv_ir + conv_ri

        return tf.complex(conv_re, conv_im)

    def complex_conv2d_real_weight(self, x, weight, padding="VALID", strides=1, dilations=1):
        xre = tf.math.real(x)
        xim = tf.math.imag(x)

        conv_rr = tf.nn.conv2d(xre, weight, padding=padding, strides=strides, dilations=dilations)
        conv_ir = tf.nn.conv2d(xim, weight, padding=padding, strides=strides, dilations=dilations)

        conv_re = conv_rr
        conv_im = conv_ir

        return tf.complex(conv_re, conv_im)

    def complex_conv2d_transpose(self, x, weight, output_shape, padding="VALID", strides=1, dilations=1):
        xre = tf.math.real(x)
        xim = tf.math.imag(x)

        wre = tf.math.real(weight)
        wim = tf.math.imag(weight)

        convT_rr = tf.nn.conv2d_transpose(xre, wre, output_shape, padding=padding, strides=strides, dilations=dilations)
        convT_ii = tf.nn.conv2d_transpose(xim, wim, output_shape, padding=padding, strides=strides, dilations=dilations)
        convT_ri = tf.nn.conv2d_transpose(xre, wim, output_shape, padding=padding, strides=strides, dilations=dilations)
        convT_ir = tf.nn.conv2d_transpose(xim, wre, output_shape, padding=padding, strides=strides, dilations=dilations)

        convT_re = convT_rr + convT_ii
        convT_im = convT_ir - convT_ri

        return tf.complex(convT_re, convT_im)

    def get_weight(self):
        return self.weight

    def call(self, x):
        weight = self.get_weight()
        # then pad
        pad = weight.shape[0]//2

        if self.pad and pad > 0:
            x = self.complex_pad2d(x, (pad,pad,pad,pad), mode='symmetric')

        # compute the convolution
        x = self.complex_conv2d(x, weight, padding="VALID", strides=self.stride, dilations=self.dilation)

        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        return x

    def backward(self, x, output_shape=None):
        weight = self.get_weight()

        # zero pad
        pad = weight.shape[0]//2
        ksz = weight.shape[0]

        # determine the output padding
        if not output_shape is None:
            output_shape = list(output_shape)
            output_padding = (
                output_shape[1] - ((x.shape[1]-1)*self.stride+1),
                output_shape[2] - ((x.shape[2]-1)*self.stride+1)
            )
        else:
            output_shape = [x.shape[0], 1, 1, self.in_channels]
            output_padding = [0,0]

        # construct output shape
        output_shape[1] = (x.shape[1] - 1)*self.stride + self.dilation * (ksz - 1) + output_padding[0] + 1
        output_shape[2] = (x.shape[2] - 1)*self.stride + self.dilation * (ksz - 1) + output_padding[1] + 1

        # zero pad input
        pad_k = self.kernel_size // 2
        x = tf.pad(x, [[0,0], 
                        [pad_k + output_padding[0]//2, pad_k + output_padding[0]//2 + np.mod(output_padding[0],2)], 
                        [pad_k + output_padding[1]//2, pad_k + output_padding[1]//2 + np.mod(output_padding[1],2)], 
                        [0,0]])

        # remove bias
        if self.use_bias:
            x = tf.nn.bias_add(x, -self.bias)

        # compute the transpose convolution
        x = self.complex_conv2d_transpose(x, weight, output_shape, padding='SAME', strides=self.stride, dilations=self.dilation)

        # transpose padding
        if self.pad and pad > 0:
            x = self.complex_pad2d_transpose(x, (pad,pad,pad,pad), mode='symmetric')
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


class ComplexConvScale2d(ComplexConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(ComplexConvScale2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, dilation=1, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)
        # create the convolution kernel
        if self.stride > 1:
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            np_k = np.reshape(np_k, (5, 5, 1, 1))
            self.blur = tf.Variable(initial_value=tf.convert_to_tensor(np_k), trainable=False)

    def get_weight(self):
        weight = super().get_weight()
        if self.stride > 1:
            weight = tf.reshape(weight, (self.kernel_size, self.kernel_size, self.out_channels*self.in_channels, 1))
            weight = tf.transpose(weight, (2, 0, 1, 3))
            for i in range(self.stride//2):
                weight = tf.pad(weight, [[0,0], [5,5], [5,5], [0,0]], 'CONSTANT')
                weight = self.complex_conv2d_real_weight(weight, self.blur, padding="SAME", strides=self.stride, dilations=self.dilation)
            weight = tf.transpose(weight, (1, 2, 0, 3))
            weight = tf.reshape(weight, (self.kernel_size+2*self.stride, self.kernel_size+2*self.stride, self.in_channels, self.out_channels))
        return weight


class ComplexConvScaleTranspose2d(ComplexConvScale2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(ComplexConvScaleTranspose2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

    def call(self, x, output_shape=None):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().call(x)


class Conv2dTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, M, N, nf_in]
        
        model = ComplexConv2d(nf_in, nf_out, kernel_size=3)
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

        model = ComplexConv2d(nf_in, nf_out, kernel_size=3)
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
        
        model = ComplexConv2d(nf_in, nf_out, kernel_size=3)
        model.build(())
        np_weight = model.weights[0].numpy()
        reduction_dim = model.weights[0].reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))

        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

class ConvScale2dTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, M, N, nf_in]

        model = ComplexConvScale2d(nf_in, nf_out, kernel_size=3, stride=2)
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

        model = ComplexConvScale2d(nf_in, nf_out, kernel_size=3, stride=2)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)

        y =  tf.complex(tf.random.normal(Kx.shape), tf.random.normal(Kx.shape))
        KHy = model.backward(y, output_shape=x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)

if __name__ == "__main__":
    unittest.test()
