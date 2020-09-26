import tensorflow as tf
import optotf.pad2d

import numpy as np
import unittest

__all__ = ['PadConv2d',
           'PadConvScale2d',
           'PadConvScaleTranspose2d']

class PadConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, bias=False, 
                 zero_mean=True, bound_norm=True, pad=True):
        super(PadConv2d, self).__init__()

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
        initializer = tf.keras.initializers.GlorotNormal()
        self.weight = self.add_weight('weight', shape=self.weight_shape, initializer=initializer)
        self.bias   = self.add_weight('bias', shape=self.bias_shape, initializer=tf.keras.initializers.Zeros()) if self.use_bias else None
        
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
                    norm = tf.math.sqrt(tf.reduce_sum(tmp**2, self.weight.reduction_dim, True))
                    if surface:
                        tmp = tmp / tf.math.maximum(norm, tf.ones_like(norm)*1e-9)
                    else:
                        tmp = tmp / tf.math.maximum(norm, tf.ones_like(norm))
                return tmp

            self.weight.proj = l2_proj
            self.weight.assign(l2_proj(self.weight, True))

    def get_weight(self):
        return self.weight

    def call(self, x):
        weight = self.get_weight()
        # then pad
        pad = weight.shape[0]//2

        if self.pad and pad > 0:
            x = optotf.pad2d.pad2d(x, (pad,pad,pad,pad), mode='symmetric')

        # compute the convolution
        x = tf.nn.conv2d(x, weight, padding="VALID", strides=self.stride, dilations=self.dilation)

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
            output_padding = [0, 0]
            output_shape = [x.shape[0], 1, 1, self.in_channels]

        # construct output shape
        output_shape[1] = (x.shape[1] - 1)*self.stride + self.dilation * (ksz - 1) + output_padding[0] + 1
        output_shape[2] = (x.shape[2] - 1)*self.stride + self.dilation * (ksz - 1) + output_padding[1] + 1

        # zero pad input
        pad_k = self.kernel_size // 2
        tf_pad = [[0,0], 
                        [pad_k + output_padding[0]//2, pad_k + output_padding[0]//2 + np.mod(output_padding[0],2)], 
                        [pad_k + output_padding[1]//2, pad_k + output_padding[1]//2 + np.mod(output_padding[1],2)], 
                        [0,0]]
        x = tf.pad(x, tf_pad)

        # remove bias
        if self.use_bias:
            x = tf.nn.bias_add(x, -self.bias)

        # compute the transpose convolution
        x = tf.nn.conv2d_transpose(x, weight, output_shape, padding='SAME', strides=self.stride, dilations=self.dilation)

        # transpose padding
        if self.pad and pad > 0:
            x = optotf.pad2d.pad2d_transpose(x, (pad,pad,pad,pad), mode='symmetric')
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


class PadConvScale2d(PadConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(PadConvScale2d, self).__init__(
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
                weight = tf.nn.conv2d(weight, self.blur, padding="SAME", strides=self.stride, dilations=self.dilation)
            weight = tf.transpose(weight, (1, 2, 0, 3))
            weight = tf.reshape(weight, (self.kernel_size+2*self.stride, self.kernel_size+2*self.stride, self.in_channels, self.out_channels))
        return weight


class PadConvScaleTranspose2d(PadConvScale2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(PadConvScaleTranspose2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

    def call(self, x, output_shape=None):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().call(x)


class PadConv2dTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32

        model = PadConv2d(nf_in, nf_out, kernel_size=3)
        x = tf.random.normal([nBatch, M, N, nf_in])
        Kx = model(x)

        with tf.GradientTape() as g:
            g.watch(x)
            Kx = model(x)
            loss = 0.5 * tf.reduce_sum(Kx ** 2)
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

        model = PadConv2d(nf_in, nf_out, kernel_size=3)
        x = tf.random.normal([nBatch, M, N, nf_in])
        Kx = model(x)

        y = tf.random.normal(Kx.shape)
        KHy = model.backward(y, x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)

    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = PadConv2d(nf_in, nf_out, kernel_size=3)
        model.build(())
        np_weight = model.weights[0].numpy()
        reduction_dim = model.weights[0].reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))

        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

class PadConvScale2dTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32

        model = PadConvScale2d(nf_in, nf_out, kernel_size=3, stride=2)
        x = tf.random.normal([nBatch, M, N, nf_in])
        
        with tf.GradientTape() as g:
            g.watch(x)
            Kx = model(x)
            loss = 0.5 * tf.reduce_sum(Kx ** 2)
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

        model = PadConvScale2d(nf_in, nf_out, kernel_size=3, stride=2)
        x = tf.random.normal([nBatch, M, N, nf_in])
        Kx = model(x)

        y = tf.random.normal(Kx.shape)
        KHy = model.backward(y, output_shape=x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)

if __name__ == "__main__":
    unittest.test()
