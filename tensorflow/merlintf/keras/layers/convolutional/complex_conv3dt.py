import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import conv_utils
from merlintf.keras.layers import complex_act as activations
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
import unittest
import six
import functools
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from merlintf.keras.layers.convolutional.complex_convolutional import ComplexConv, ComplexConv2D, ComplexConv3DTranspose, ComplexConv3D
from merlintf.keras.utils import validate_input_dimension

def calculate_intermediate_filters(filters, kernel_size, channel_in):
    return np.ceil((filters * channel_in * np.prod(kernel_size)) /
            (channel_in * kernel_size[1] * kernel_size[2] * kernel_size[3] + filters * kernel_size[0])).astype(np.int32)

class Conv3Dt(tf.keras.layers.Layer):
    def __init__(self,
                 filters,  # out
                 kernel_size,
                 strides=(1, 1, 1, 1),
                 padding='same',
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
                 shapes=None,
                 axis_conv_t=2,
                 zero_mean=True,
                 bound_norm=True,
                 pad=True,
                 intermediate_filters=None,
                 **kwargs):
        super(Conv3Dt, self).__init__()

        if intermediate_filters == None:
            self.intermediate_filters = filters
        else:
            self.intermediate_filters = intermediate_filters
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.shape = shapes
        self.axis_conv_t = axis_conv_t
        self.padding = padding
        self.kernel_size = validate_input_dimension('3Dt', kernel_size)
        self.strides = validate_input_dimension('3Dt', strides)
        self.dilation_rate = validate_input_dimension('3Dt', dilation_rate)

        self.conv_xyz = ComplexConv3D(
            filters=intermediate_filters,
            kernel_size=(self.kernel_size[1], self.kernel_size[2], self.kernel_size[3]),
            strides=(self.strides[1], self.strides[2], self.strides[3]),
            padding=self.padding,
            data_format=data_format,
            dilation_rate=(self.dilation_rate[1], self.dilation_rate[2], self.dilation_rate[3]),
            groups=groups,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

        conv_t_filters = filters

        self.conv_t = ComplexConv3D(
            filters=conv_t_filters,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding=self.padding,
            data_format=data_format,
            dilation_rate=(self.dilation_rate[0], 1, 1),
            groups=groups,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

    def calculate_output_shape(self, input_spatial_shape):
        # calculate output shape x,y,z, but not channels
        output_spatial_shape = input_spatial_shape.copy()
        if self.padding.upper() == "SAME":
            if self.data_format == 'channels_first':
                for i in range(3):  # calculate x,y,z
                    output_spatial_shape[i + 2] = int(np.ceil(input_spatial_shape[i + 2] / self.strides[i + 1]))
            else:  # channel last
                for i in range(3):
                    output_spatial_shape[i + 1] = int(np.ceil(input_spatial_shape[i + 1] / self.strides[i + 1]))

        elif self.padding.upper() == "VALID":

            if self.data_format == 'channels_first':
                for i in range(3):
                    output_spatial_shape[i + 2] = int(np.ceil((input_spatial_shape[i + 2]
                                                               - (self.kernel_size[i + 1] - 1)
                                                               * self.dilation_rate[i + 1]) / self.strides[i + 1]))
            else:  # channel last
                for i in range(3):
                    output_spatial_shape[i + 1] = int(np.ceil((input_spatial_shape[i + 1]
                                                               - (self.kernel_size[i + 1] - 1)
                                                               * self.dilation_rate[i + 1]) / self.strides[i + 1]))

        return output_spatial_shape  # 5 dimensional shape

    def batch_concat_conv(self, inputs, func, axis=0):
        shape_in = inputs.shape
        x_list = tf.split(inputs, shape_in[axis], axis=axis)
        x = tf.concat(x_list, axis=0)
        x = tf.squeeze(x, axis=axis)
        x = func(x)
        x_list = tf.split(x, shape_in[axis], axis=0)
        return tf.stack(x_list, axis=axis)

    def build(self, input_shape):

        self.shape = input_shape

        input_shape = list(input_shape)
        shape_xyz = input_shape.copy()

        if self.data_format == 'channels_first':
            # [batch, channel, time, x, y, z]
            shape_xyz.pop(2)  # xyz pop time dimension
            shape_t = self.calculate_output_shape(shape_xyz)
            shape_t[1] = self.intermediate_filters  # channels of input of conv_t = channels of output of shape_xyz

        else:  # channels last
            # [batch, time, x, y, z, channel]
            shape_xyz.pop(1)  # xyz pop time dimension
            shape_t = self.calculate_output_shape(shape_xyz)
            shape_t[-1] = self.intermediate_filters  # channels of input of conv_t = channels of output of shape_xyz

        self.conv_xyz.build(shape_xyz)
        self.conv_t.build(shape_t)

    def call(self, x):
        if self.data_format == 'channels_first':  # [batch, chs, time, x, y, z]
            x_sp = self.batch_concat_conv(x, self.conv_xyz, axis=2)
            x_t = self.batch_concat_conv(x_sp, self.conv_t, axis=self.axis_conv_t)

        else:  # channels last #[batch, time, x, y, z, chs]
            x_sp = self.batch_concat_conv(x, self.conv_xyz, axis=1)
            x_t = self.batch_concat_conv(x_sp, self.conv_t, axis=self.axis_conv_t)

        return x_t


class Conv3DtTranspose(tf.keras.layers.Layer):
    def __init__(self,
                 filters,  # out
                 kernel_size,
                 strides=(1, 1, 1, 1),
                 padding='same',
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
                 shapes=None,
                 axis_conv_t=2,
                 zero_mean=True,
                 bound_norm=True,
                 pad=True,
                 intermediate_filters=None,
                 **kwargs):
        super(Conv3DtTranspose, self).__init__()

        if intermediate_filters == None:
            self.intermediate_filters = filters
        else:
            self.intermediate_filters = intermediate_filters
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.shape = shapes
        self.axis_conv_t = axis_conv_t

        self.kernel_size = validate_input_dimension('3Dt', kernel_size)
        self.strides = validate_input_dimension('3Dt', strides)
        self.dilation_rate = validate_input_dimension('3Dt', dilation_rate)

        self.conv_xyz_filters = filters
        self.conv_xyz = ComplexConv3DTranspose(
            filters=self.conv_xyz_filters,
            kernel_size=(self.kernel_size[1], self.kernel_size[2], self.kernel_size[3]),
            strides=(self.strides[1], self.strides[2], self.strides[3]),
            padding=padding,
            data_format=data_format,
            dilation_rate=(self.dilation_rate[1], self.dilation_rate[2], self.dilation_rate[3]),
            groups=groups,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

        self.conv_t = ComplexConv3DTranspose(
            filters=self.intermediate_filters,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding=padding,
            data_format=data_format,
            dilation_rate=(self.dilation_rate[0], 1, 1),
            groups=groups,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

    def batch_concat_conv(self, inputs,func, axis=0):
        shape_in = inputs.shape
        x_list = tf.split(inputs, shape_in[axis], axis=axis)
        x = tf.concat(x_list, axis=0)
        x = tf.squeeze(x, axis=axis)
        x = func(x)
        x_list = tf.split(x, shape_in[axis], axis=0)
        return tf.stack(x_list, axis=axis)

    def build(self, input_shape):
        self.shape = input_shape
        input_shape = list(input_shape)
        shape_xyz = input_shape.copy()
        shape_t = input_shape.copy()

        if self.data_format == 'channels_first':  # [batch, channels, time, x, y, z]
            shape_t.pop(self.axis_conv_t)  # pop selected axis
            shape_xyz.pop(2)  # [batch, channels, x,y,z]
            shape_xyz[1] = self.intermediate_filters  # second channel num = intermediate_filters

        else:  # channels last # [batch, time, x, y, z, channels]
            shape_t.pop(self.axis_conv_t)
            shape_xyz.pop(1)  # [batch, x,y,z, channels]
            shape_xyz[-1] = self.intermediate_filters  # last channel num = intermediate_filters from conv_t output

        self.conv_xyz.build(shape_xyz)
        self.conv_t.build(shape_t)

    def call(self, x):
        if self.data_format == 'channels_first':  # [batch, chs, time, x, y, z]
            x_t = self.batch_concat_conv(x, self.conv_t, axis=self.axis_conv_t)
            x_sp = self.batch_concat_conv(x_t, self.conv_xyz, axis=2)

        else:
            x_t = self.batch_concat_conv(x, self.conv_t, axis=self.axis_conv_t)
            x_sp = self.batch_concat_conv(x_t, self.conv_xyz, axis=1)

        return x_sp


class ComplexConv3dtTest(unittest.TestCase):
    def test_Conv3dt(self):
        self._test_Conv3dt()
        self._test_Conv3dt(stride=(2, 2, 2, 2))
        self._test_Conv3dt(channel_last=False)

    def test_Conv3dtTranspose(self):
        self._test_Conv3dt(is_transpose=True)
        self._test_Conv3dt(is_transpose=True, stride=(2, 2, 2, 2))
        self._test_Conv3dt(is_transpose=True, channel_last=False)

    def _test_Conv3dt(self, dim_in=[8, 32, 32, 12], nBatch=2, nf_in=3, nf_out=18, ksz=(3, 5, 5, 5), stride=(1, 1, 1, 1),
                      channel_last=True, axis_conv_t=2, is_transpose=False):
        if channel_last:
            shape = [nBatch] + dim_in + [nf_in]
            expected_shape = [nBatch] + list(np.asarry(dim_in)/np.asarray(stride)) + [nf_out]
        else:
            shape = [nBatch] + [nf_in] + dim_in
            expected_shape = [nBatch] + [nf_out] + list(np.asarry(dim_in)/np.asarray(stride))

        ksz = validate_input_dimension('3Dt', ksz)
        nf_inter = calculate_intermediate_filters(nf_out, ksz, nf_in)

        if is_transpose:
            model = Conv3DtTranspose(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter)
        else:
            model = Conv3Dt(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter)

        x_real = tf.cast(tf.random.normal(shape), dtype=tf.float32)
        x_imag = tf.cast(tf.random.normal(shape), dtype=tf.float32)
        x = tf.complex(x_real, x_imag)
        Kx = model(x)

        self.assertTrue(Kx.shape == expected_shape)

if __name__ == "__main__":
    unittest.main()
