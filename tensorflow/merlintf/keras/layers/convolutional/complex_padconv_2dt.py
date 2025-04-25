import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from merlintf.keras.layers.convolutional.complex_padconv import ComplexPadConv3D, ComplexPadConvScale3D
from merlintf.keras.utils import validate_input_dimension

class ComplexPadConv2Dt(tf.keras.layers.Layer):
    def __init__(self,
               filters,
               intermediate_filters,
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
        super(ComplexPadConv2Dt, self).__init__()

        self.intermediate_filters = intermediate_filters

        if strides[1] > 2:
            conv_module = ComplexPadConvScale3D
        else:
            conv_module = ComplexPadConv3D
        
        kernel_size = validate_input_dimension('2Dt', kernel_size)
        strides = validate_input_dimension('2Dt', strides)
        dilation_rate = validate_input_dimension('2Dt', dilation_rate)

        self.conv_xy = conv_module(
        filters=intermediate_filters,
        kernel_size=(1, kernel_size[1], kernel_size[2]),
        strides=(1, strides[1], strides[2]),
        padding=padding,
        data_format=data_format,
        dilation_rate=(1, dilation_rate[1], dilation_rate[2]),
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
        x_sp = self.conv_xy(x)
        x_t = self.conv_t(x_sp)
        return x_t  

    def backward(self, x, output_shape=None):
        xT_t = self.conv_t.backward(x, output_shape)
        xT_sp = self.conv_xy.backward(xT_t, output_shape)
        return xT_sp
