import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import conv_utils
import numpy as np
from merlintf.keras.layers.convolutional.complex_convolutional import ComplexConv2DTranspose, ComplexConv2D, ComplexConv3DTranspose, ComplexConv3D
from merlintf.keras.utils import validate_input_dimension

def calculate_intermediate_filters_2D(filters, kernel_size, channel_in):
    return np.ceil((filters * channel_in * np.prod(kernel_size)) / (channel_in * kernel_size[1] * kernel_size[2]
                                                                    + filters * kernel_size[0])).astype(np.int32)


class ComplexConv2Dt(tf.keras.layers.Layer):
    def __init__(self,
                 filters,  # out
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='same',
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
                 shapes=None,
                 axis_conv_t=2,  # axis to "loop over" for the temporal convolution, best fully sampled direction x or slice
                 zero_mean=True,
                 bound_norm=True,
                 pad=True,
                 intermediate_filters=None,
                 use_3D_convs=True,  # True: use 3D conv layers, False: use 2D conv layers and stack along batch dim
                 **kwargs):
        super(ComplexConv2Dt, self).__init__()

        if intermediate_filters == None:
            self.intermediate_filters = filters
        else:
            self.intermediate_filters = intermediate_filters
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.shape = shapes
        self.axis_conv_t = axis_conv_t
        self.padding = padding
        self.kernel_size = validate_input_dimension('2Dt', kernel_size)
        self.strides = validate_input_dimension('2Dt', strides)
        self.dilation_rate = validate_input_dimension('2Dt', dilation_rate)
        self.use_3D_convs = use_3D_convs

        if use_3D_convs:
            self.conv_xy = ComplexConv3D(
                filters=self.intermediate_filters,
                kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
                strides=(1, self.strides[1], self.strides[2]),
                padding=padding,
                data_format=data_format,
                dilation_rate=(1, self.dilation_rate[1], self.dilation_rate[2]),
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

            self.conv_t = ComplexConv3D(
                filters=filters,
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
        else:
            self.conv_xy = ComplexConv2D(
                filters=self.intermediate_filters,
                kernel_size=(self.kernel_size[1], self.kernel_size[2]),
                strides=(self.strides[1], self.strides[2]),
                padding=padding,
                data_format=data_format,
                dilation_rate=(self.dilation_rate[1], self.dilation_rate[2]),
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

            self.conv_t = ComplexConv2D(
                filters=filters,
                kernel_size=(self.kernel_size[0], 1),
                strides=(self.strides[0], 1),
                padding=padding,
                data_format=data_format,
                dilation_rate=(self.dilation_rate[0], 1),
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
        # calculate output shape x, y, but not channels
        output_spatial_shape = input_spatial_shape.copy()
        if self.padding.upper() == "SAME":
            if self.data_format == 'channels_first':
                # input_spatial_shape=[batch, channels, x,y]
                for i in range(2):  # calculate x,y
                    output_spatial_shape[i + 2] = int(np.ceil(input_spatial_shape[i + 2] / self.strides[i + 1]))
            else:  # channel last
                for i in range(2):  # calculate x,y
                    output_spatial_shape[i + 1] = int(np.ceil(input_spatial_shape[i + 1] / self.strides[i + 1]))

        elif self.padding.upper() == "VALID":
            if self.data_format == 'channels_first':
                for i in range(2):  # calculate x,y
                    output_spatial_shape[i + 2] = int(np.ceil((input_spatial_shape[i + 2]
                                                               - (self.kernel_size[i + 1] - 1)
                                                               * self.dilation_rate[i + 1]) / self.strides[i + 1]))
            else:  # channel last
                for i in range(2):  # calculate x,y
                    output_spatial_shape[i + 1] = int(np.ceil((input_spatial_shape[i + 1]
                                                               - (self.kernel_size[i + 1] - 1)
                                                               * self.dilation_rate[i + 1]) / self.strides[i + 1]))

        return output_spatial_shape  # 4 dimensional shape

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
        shape_xy = input_shape.copy()

        if self.data_format == 'channels_first':
            # [batch, channel, time, x, y]
            if not self.use_3D_convs:
                shape_xy.pop(2)  # xy pop time dimension
            shape_t = self.calculate_output_shape(shape_xy)
            shape_t[1] = self.intermediate_filters  # channels of input of conv_t = channels of output of shape_xy

        else:  # channels last
            # [batch, time, x, y, channel]
            if not self.use_3D_convs:
                shape_xy.pop(1)  # xy pop time dimension
            shape_t = self.calculate_output_shape(shape_xy)
            shape_t[-1] = self.intermediate_filters  # channels of input of conv_t = channels of output of shape_xy

        self.conv_xy.build(shape_xy)
        self.conv_t.build(shape_t)

    def call(self, x):
        if self.use_3D_convs:
            return self.conv_t(self.conv_xy(x))
        else:
            if self.data_format == 'channels_first':  # [batch, chs, time, x, y]
                x_sp = self.batch_concat_conv(x, self.conv_xy, axis=2)
                x_t = self.batch_concat_conv(x_sp, self.conv_t, axis=self.axis_conv_t)

            else:  # channels last [batch, time, x, y, chs]
                x_sp = self.batch_concat_conv(x, self.conv_xy, axis=1)
                x_t = self.batch_concat_conv(x_sp, self.conv_t, axis=self.axis_conv_t)
            return x_t


class ComplexConv2DtTranspose(tf.keras.layers.Layer):
    def __init__(self,
                 filters,  # out
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='same',
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
                 shapes=None,
                 axis_conv_t=2,  # axis to "loop over" for the temporal convolution, best fully sampled direction x or slice
                 zero_mean=True,
                 bound_norm=True,
                 pad=True,
                 intermediate_filters=None,
                 use_3D_convs=True,  # True: use 3D conv layers, False: use 2D conv layers and stack along batch dim
                 **kwargs):
        super(ComplexConv2DtTranspose, self).__init__()

        if intermediate_filters == None:
            self.intermediate_filters = filters
        else:
            self.intermediate_filters = intermediate_filters
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.shape = shapes
        self.axis_conv_t = axis_conv_t
        self.use_3D_convs = use_3D_convs

        self.kernel_size = validate_input_dimension('2Dt', kernel_size)
        self.strides = validate_input_dimension('2Dt', strides)
        self.dilation_rate = validate_input_dimension('2Dt', dilation_rate)
        self.conv_xy_filters = filters
        if use_3D_convs:
            self.conv_xy = ComplexConv3DTranspose(
                filters=filters,
                kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
                strides=(1, self.strides[1], self.strides[2]),
                padding=padding,
                data_format=data_format,
                dilation_rate=(1, self.dilation_rate[1], self.dilation_rate[2]),
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
        else:
            self.conv_xy = ComplexConv2DTranspose(
                filters=self.conv_xy_filters,
                kernel_size=(self.kernel_size[1], self.kernel_size[2]),
                strides=(self.strides[1], self.strides[2]),
                padding=padding,
                data_format=data_format,
                dilation_rate=(self.dilation_rate[1], self.dilation_rate[2]),
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

            self.conv_t = ComplexConv2DTranspose(
                filters=self.intermediate_filters,
                kernel_size=(self.kernel_size[0], 1),
                strides=(self.strides[0], 1),
                padding=padding,
                data_format=data_format,
                dilation_rate=(self.dilation_rate[0], 1),
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
        shape_xy = input_shape.copy()
        shape_t = input_shape.copy()

        if self.data_format == 'channels_first':  # [batch,  channels, time, x, y]
            if not self.use_3D_convs:
                shape_t.pop(self.axis_conv_t)  # pop selected axis
                shape_xy.pop(2)  # [batch, channels, x,y]
            shape_xy[1] = self.intermediate_filters  # second channel num = intermediate_filters from conv_t output

        else:
            # channels last  [batch, time, x, y, channels]
            if not self.use_3D_convs:
                shape_t.pop(self.axis_conv_t)
                shape_xy.pop(1)  # [batch, x, y, channels]
            shape_xy[-1] = self.intermediate_filters  # last channel num = intermediate_filters from conv_t output

        self.conv_xy.build(shape_xy)
        self.conv_t.build(shape_t)

    def call(self, x):
        if self.use_3D_convs:
            return self.conv_xy(self.conv_t(x))
        else:
            if self.data_format == 'channels_first':  # [batch, chs, time, x, y]
                x_t = self.batch_concat_conv(x, self.conv_t, axis=self.axis_conv_t)
                x_sp = self.batch_concat_conv(x_t, self.conv_xy, axis=2)

            else:  # channels last #[batch, time, x, y, z, chs]
                x_t = self.batch_concat_conv(x, self.conv_t, axis=self.axis_conv_t)
                x_sp = self.batch_concat_conv(x_t, self.conv_xy, axis=1)

            return x_sp

# Aliases
ComplexConvolution2Dt = ComplexConv2Dt
ComplexConvolution2DtTranspose = ComplexConv2DtTranspose
ComplexDeconvolution2Dt = ComplexDeconv2Dt = ComplexConv2DtTranspose
