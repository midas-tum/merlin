import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

import numpy as np
import optotf.pad

from merlintf.keras.layers.complex_init import complex_initializer
from merlintf.keras.layers.convolutional.complex_conv import (
    complex_conv2d_real_weight_transpose,
    complex_conv3d_real_weight_transpose,
    complex_conv2d_real_weight,
    complex_conv3d_real_weight,
)

from merlintf.keras.layers.convolutional.complex_convolutional_realkernel import ComplexConvRealWeight

__all__ = ['ComplexPadConvRealWeight2D',
           'ComplexPadConvRealWeight3D',
           ]

"""
Experimental pad conv class for Variational Networks
Dilations and Stridings are NOT fully supported!
"""

class ComplexPadConvRealWeight(ComplexConvRealWeight):
    def __init__(self,
            rank,
            filters,
            kernel_size,
            strides=1,
            padding='symmetric',
            data_format=None,
            dilation_rate=1,
            groups=1,
            use_bias=False,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            conv_op=None,
            zero_mean=True,
            bound_norm=True,
            pad=True,
            **kwargs):
        super().__init__(
               rank,
               filters,
               kernel_size,
               strides,
               "valid",
               data_format,
               dilation_rate,
               groups,
               None,
               use_bias,
               kernel_initializer,
               bias_initializer,
               kernel_regularizer,
               bias_regularizer,
               activity_regularizer,
               kernel_constraint,
               bias_constraint,
               trainable,
               name,
               conv_op,
               **kwargs)
        
        self.optox_padding = padding
        self.pad = pad
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        assert not self._channels_first

    def build(self, input_shape):
        super().build(input_shape)
        # define the weight constraints
        if self.zero_mean or self.bound_norm:
            self.kernel.reduction_dim = tuple([d for d in range(tf.rank(self.kernel))])
            reduction_dim_mean = self.kernel.reduction_dim

            def l2_proj(weight, surface=False):
                tmp = weight
                # reduce the mean
                if self.zero_mean:
                    tmp = tmp - tf.reduce_mean(tmp, reduction_dim_mean, True)
                # normalize by the l2-norm
                if self.bound_norm:
                    norm = tf.math.sqrt(tf.reduce_sum(tmp ** 2, self.kernel.reduction_dim, True))
                    if surface:
                        tmp = tmp / tf.math.maximum(norm, tf.ones_like(norm)*1e-9)
                    else:
                        tmp = tmp / tf.math.maximum(norm, tf.ones_like(norm))
                return tmp

            self.kernel.proj = l2_proj
            self.kernel.assign(l2_proj(self.kernel, True))

    def _compute_optox_padding(self):
        pad = []
        for w in self.kernel.shape[:self.rank][::-1]:
            pad += [w//2, w//2]
        return pad

    def call(self, inputs):
        # first pad
        pad = self._compute_optox_padding()
        if self.pad and any(pad):
            inputs = optotf.pad._pad(self.rank, inputs, pad, self.optox_padding)

        outputs = self._complex_convolution_op(inputs, self.kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs

    def _complex_conv_transpose_op(self, x, weight, output_shape):
        if self.rank == 2:
            conv_fun = complex_conv2d_real_weight_transpose
        elif self.rank == 3:
            conv_fun = complex_conv3d_real_weight_transpose
        return conv_fun(x, weight, output_shape, padding='SAME', strides=(1,) + self.strides + (1,), dilations=(1,) + self.dilation_rate + (1,))

    def backward(self, x, output_shape=None):
        pad = self._compute_optox_padding()
        ksz = [w for w in self.kernel.shape[:self.rank]]

        # determine the output padding
        if not output_shape is None:
            output_shape = tf.unstack(output_shape)
            output_shape[-1] = tf.shape(self.kernel)[-2]
            output_padding = [output_shape[i+1] - ((tf.shape(x)[i+1]-1)*self.strides[i]+1) for i in range(self.rank)]
        else:
            output_shape = [tf.shape(x)[0],] + [1 for i in range(self.rank)] + [tf.shape(self.kernel)[-2],]
            output_padding = [0 for i in range(self.rank)]

        # construct output shape
        output_shape = [(tf.shape(x)[i] - 1)*self.strides[i-1] + self.dilation_rate[i-1] * (ksz[i-1] - 1) + output_padding[i-1] + 1 if (i > 0 and i < self.rank + 1) else output_shape[i] for i in range(self.rank + 2) ]
        output_shape = tf.stack(output_shape)
        # zero pad input
        pad_k = [w//2 for w in self.kernel_size]
        tf_pad = [[0,0,],] + \
                 [[pad_k[i] + output_padding[i]//2, pad_k[i] + output_padding[i]//2 + np.mod(output_padding[i],2)] for i in range(self.rank)] + \
                 [[0,0,],]
        x = tf.pad(x, tf_pad)
       
        # remove bias
        if self.use_bias:
            x = tf.nn.bias_add(x, -1 * self.bias)

        # compute the transpose convolution
        x = self._complex_conv_transpose_op(x, self.kernel, output_shape)

        # transpose padding
        if self.pad and any(pad):
            x = optotf.pad._pad_transpose(self.rank, x, pad, mode=self.optox_padding)
        return x

class ComplexPadConvRealWeight2D(ComplexPadConvRealWeight):
  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='symmetric',
               data_format=None,
               dilation_rate=(1, 1),
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
    super(ComplexPadConvRealWeight2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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
        pad=pad,
        **kwargs)


class ComplexPadConvRealWeight3D(ComplexPadConvRealWeight):
  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='symmetric',
               data_format=None,
               dilation_rate=(1, 1, 1),
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
    super(ComplexPadConvRealWeight3D, self).__init__(
        rank=3,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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

