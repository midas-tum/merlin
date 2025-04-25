import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import Conv3D, Conv3DTranspose
import numpy as np
from merlintf.keras.utils import validate_input_dimension

#TODO remove
def calculate_intermediate_filters_3D(filters, kernel_size, channel_in):
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
                 axis_conv_t=2,  # axis to "loop over" for the temporal convolution, best fully sampled direction x or slice
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

        self.conv_xyz = Conv3D(
            filters=self.intermediate_filters,
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

        self.conv_t = Conv3D(
            filters=filters,
            kernel_size=(self.kernel_size[0], 1, 1),
            strides=(self.strides[0], 1, 1),
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
                 axis_conv_t=2,  # axis to "loop over" for the temporal convolution, best fully sampled direction x or slice
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
        self.conv_xyz = Conv3DTranspose(
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

        self.conv_t = Conv3DTranspose(
            filters=self.intermediate_filters,
            kernel_size=(self.kernel_size[0], 1, 1),
            strides=(self.strides[0], 1, 1),
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

# aliases
Convolution3Dt = Conv3Dt
Convolution3DtTranspose = Conv3DtTranspose
Deconvolution3Dt = Deconv3Dt = Conv3DtTranspose
