import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from merlintf.keras.layers.convolutional.complex_padconv import ComplexPadConv3D, ComplexPadConvScale3D

import numpy as np
import unittest

class ComplexPadConv3Dt(tf.keras.layers.Layer):
    def __init__(self,
               filters,
               intermediate_filters,
               kernel_size,
               strides=(1, 1, 1, 1),
               padding='symmetric',
               data_format=None,
               dilation_rate=(1, 1, 1, 1),
               groups=1,
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               zero_mean=True,
               bound_norm=True,
               pad=True,
               **kwargs):
        super(ComplexPadConv3Dt, self).__init__()

        self.intermediate_filters = intermediate_filters

        if strides[1] > 2:
            conv_module = ComplexPadConvScale3D
        else:
            conv_module = ComplexPadConv3D

        self.conv_xyz = conv_module(
        filters=intermediate_filters,
        kernel_size=(kernel_size[1], kernel_size[2], kernel_size[3]),
        strides=(strides[1], strides[2], strides[3]),
        padding=padding,
        data_format=data_format,
        dilation_rate=(dilation_rate[1], dilation_rate[2], dilation_rate[3]),
        groups=groups,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        zero_mean=zero_mean,
        bound_norm=bound_norm,
        pad=pad,
        **kwargs)

        self.conv_t = ComplexPadConv3D(
        filters=filters,
        kernel_size=(kernel_size[0], 1, 1),
        strides=(strides[0], 1, 1),
        padding=padding,
        data_format=data_format,
        dilation_rate=(dilation_rate[0], 1, 1),
        groups=groups,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        zero_mean=False,
        bound_norm=bound_norm,
        pad=pad,
        **kwargs)

    def call(self, x):
        if self._data_format == 'channels_first':
            x_sp = tf.stack([self.conv_xyz(tf.squeeze(x[:, :, i, :, :, :], axis=[2])) for i in range(0, self.input_shape[2])], axis=2)
            x_t = tf.stack([self.conv_t(tf.squeeze(x_sp[:, :, :, :, :, i], axis=[5])) for i in range(0, self.input_shape[5])], axis=5)
        else:
            x_sp = tf.stack([self.conv_xyz(tf.squeeze(x[:, i, :, :, :, :], axis=[1])) for i in range(0, self.input_shape[1])], axis=1)
            x_t = tf.stack([self.conv_t(tf.squeeze(x_sp[:, :, :, :, i, :], axis=[4])) for i in range(0, self.input_shape[4])],  axis=4)
        return x_t

    def backward(self, x, output_shape=None):
        if self._data_format == 'channels_first':
            xT_t = tf.stack([self.conv_t.backward(tf.squeeze(x[:, :, :, :, :, i], axis=[5]), output_shape) for i in range(0, self.input_shape[5])], axis=5)
            xT_sp = tf.stack([self.conv_xyz.backward(tf.squeeze(xT_t[:, :, i, :, :, :], axis=[2]), output_shape) for i in range(0, self.input_shape[2])], axis=2)
        else:
            xT_t = tf.stack([self.conv_t.backward(tf.squeeze(x[:, :, :, :, i, :], axis=[4]), output_shape) for i in range(0, self.input_shape[4])], axis=4)
            xT_sp = tf.stack([self.conv_xyz.backward(tf.squeeze(xT_t[:, i, :, :, :, :], axis=[1]), output_shape) for i in range(0, self.input_shape[1])], axis=1)
        return xT_sp

class ComplexPadConv3dtTest(unittest.TestCase):
    def test_grad(self):
        nBatch = 5
        M = 128
        N = 128
        D = 24
        T = 16
        nf_in = 2
        nf_out = 32
        shape = [nBatch, T, M, N, D, nf_in]
        ksz = (3,5,5,5)
        nf_inter = np.ceil((nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] * ksz[3] + nf_out * ksz[0])).astype(np.int32)
        model = ComplexPadConv3Dt(nf_out, nf_inter, kernel_size=ksz)
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
        T = 16
        nf_in = 2
        nf_out = 32
        shape = [nBatch, T, M, N, D, nf_in]
        ksz = (3,5,5,5)
        nf_inter = np.ceil((nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] * ksz[3] + nf_out * ksz[0])).astype(np.int32)

        model = ComplexPadConv3Dt(nf_out, nf_inter, kernel_size=ksz)
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape))
        Kx = model(x)

        y = tf.complex(tf.random.normal(Kx.shape), tf.random.normal(Kx.shape))
        KHy = model.backward(y, x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)