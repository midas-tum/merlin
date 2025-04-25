import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import conv_utils

from merlintf.keras.layers.convolutional.complex_padconv import ComplexPadConv3D, ComplexPadConvScale3D
from merlintf.keras.utils import validate_input_dimension

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
        self.data_format = conv_utils.normalize_data_format(data_format)

        if strides[1] > 2:
            conv_module = ComplexPadConvScale3D
        else:
            conv_module = ComplexPadConv3D

        kernel_size = validate_input_dimension('3Dt', kernel_size)
        strides = validate_input_dimension('3Dt', strides)
        dilation_rate = validate_input_dimension('3Dt', dilation_rate)

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

    def build(self, input_shape):
        self.shape = input_shape
        input_shape = list(input_shape)
        shape_xyz = input_shape.copy()
        shape_t = input_shape.copy()

        if self.data_format == 'channels_first':
            shape_xyz.pop(2)
            shape_t.pop(5)
            shape_t[1] = self.intermediate_filters
        else:
            shape_xyz.pop(1)
            shape_t.pop(4)
            shape_t[-1] = self.intermediate_filters

        self.conv_xyz.build(shape_xyz)
        self.conv_t.build(shape_t)

    def call(self, x):
        if self.data_format == 'channels_first':
            x_sp = tf.stack([self.conv_xyz(x[:, :, i, :, :, :]) for i in range(0, self.shape[2])], axis=2)
            x_t = tf.stack([self.conv_t(x_sp[:, :, :, :, :, i]) for i in range(0, self.shape[5])], axis=5)
        else:
            x_sp = tf.stack([self.conv_xyz(x[:, i, :, :, :, :]) for i in range(0, self.shape[1])], axis=1)
            x_t = tf.stack([self.conv_t(x_sp[:, :, :, :, i, :]) for i in range(0, self.shape[4])], axis=4)
        return x_t

    def backward(self, x, output_shape=None):
        if output_shape is None:
            shape_xyz = None
            shape_t = None
        else:
            output_shape = list(output_shape)
            shape_xyz = output_shape.copy()
            shape_t = output_shape.copy()

            if self.data_format == 'channels_first':
                shape_xyz.pop(2)
                shape_t.pop(5)
                shape_xyz[1] = self.intermediate_filters
            else:
                shape_xyz.pop(1)
                shape_t.pop(4)
                shape_xyz[-1] = self.intermediate_filters

        if self.data_format == 'channels_first':
            xT_t = tf.stack([self.conv_t.backward(x[:, :, :, :, :, i], shape_t) for i in range(0, self.shape[5])], axis=5)
            xT_sp = tf.stack([self.conv_xyz.backward(xT_t[:, :, i, :, :, :], shape_xyz) for i in range(0, self.shape[2])], axis=2)
        else:
            xT_t = tf.stack([self.conv_t.backward(x[:, :, :, :, i, :], shape_t) for i in range(0, self.shape[4])], axis=4)
            xT_sp = tf.stack([self.conv_xyz.backward(xT_t[:, i, :, :, :, :], shape_xyz) for i in range(0, self.shape[1])], axis=1)
        return xT_sp
