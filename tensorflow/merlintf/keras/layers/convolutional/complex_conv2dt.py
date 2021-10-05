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


class Conv2Dt(tf.keras.layers.Layer):
    def __init__(self,
                 filters,  # out
                 intermediate_filters,
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
                 axis_conv_t=2,
                 zero_mean=True,
                 bound_norm=True,
                 pad=True,
                 **kwargs):
        super(Conv2Dt, self).__init__()

        self.intermediate_filters = intermediate_filters
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.shape = shapes
        self.axis_conv_t = axis_conv_t
        self.padding=padding
        self.kernel_size = validate_input_dimension('2Dt', kernel_size)
        self.strides = validate_input_dimension('2Dt', strides)
        self.dilation_rate = validate_input_dimension('2Dt', dilation_rate)

        self.conv_xy = ComplexConv2D(
            filters=intermediate_filters,
            kernel_size=(kernel_size[1], kernel_size[2]),
            strides=(int(strides[1]), int(strides[2])),
            padding=padding,
            data_format=data_format,
            dilation_rate=(int(dilation_rate[1]), int(dilation_rate[2])),
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

        self.conv_t = ComplexConv2D(
            filters=conv_t_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=padding,
            data_format=data_format,
            dilation_rate=(dilation_rate[0], 1),
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
    def calculate_output_shape(self,input_spatial_shape):
        # calculate output shape x,y, but not channels
        output_spatial_shape=input_spatial_shape.copy()
        if self.padding.upper()=="SAME":
            if self.data_format == 'channels_first':
                # input_spatial_shape=[batch, channels, x,y]
                for i in range (2): # calculate x,y
                        output_spatial_shape[i+2] = int(np.ceil(input_spatial_shape[i+2] / self.strides[i+1]))
            else: # channel last
                for i in range(2): # calculate x,y
                    output_spatial_shape[i+1] = int(np.ceil(input_spatial_shape[i+1] / self.strides[i + 1]))

        elif self.padding.upper()=="VALID":

            if self.data_format == 'channels_first':
                 for i in range (2):  # calculate x,y
                        output_spatial_shape[i+2] = int(np.ceil((input_spatial_shape[i+2]
                                                                 - (self.kernel_size[i+1]-1)
                                                                 * self.dilation_rate[i+1]) / self.strides[i+1]))
            else: # channel last
                for i in range(2): # calculate x,y
                    output_spatial_shape[i+1] = int(np.ceil((input_spatial_shape[i+1]
                                                             - (self.kernel_size[i+1]-1)
                                                             * self.dilation_rate[i+1]) / self.strides[i+1]))


        return output_spatial_shape  # 4 dimensional shape
    def build(self, input_shape):
        self.shape = input_shape
        input_shape = list(input_shape)
        shape_xy = input_shape.copy()


        if self.data_format == 'channels_first':
            # [batch, channel,time, x,y]
            shape_xy.pop(2) # xy pop time dimension
            shape_t=self.calculate_output_shape(shape_xy)
            shape_t[1] = self.intermediate_filters # channels of input of conv_t = channels of output of shape_xy


        else:  # channels last
            # [batch, time, x,y, channel]
            shape_xy.pop(1) # xy pop time dimension
            shape_t = self.calculate_output_shape(shape_xy)
            shape_t[-1] = self.intermediate_filters   # channels of input of conv_t = channels of output of shape_xy

        self.conv_xy.build(shape_xy)
        self.conv_t.build(shape_t)

    def call(self, x):
        if self.data_format == 'channels_first':  # [batch, chs, time, x,y]

            x_sp_list = [self.conv_xy(x[:, :, i, :, :]) for i in
                         range(0, self.shape[2])]  # split 'time' dimension, and 3D conv (depthwise) for each
            x_sp = tf.stack(x_sp_list, axis=2)
            shape_sp = x_sp.shape



            if self.axis_conv_t == 1:
                x_t_list = [self.conv_t(x_sp[:, i, :, :, :]) for i in range(0, shape_sp[self.axis_conv_t])]
            elif self.axis_conv_t == 3:
                x_t_list = [self.conv_t(x_sp[:, :, :, i, :]) for i in range(0, shape_sp[self.axis_conv_t])]
            elif self.axis_conv_t == 4:
                x_t_list = [self.conv_t(x_sp[:, :, :, :, i]) for i in range(0, shape_sp[self.axis_conv_t])]

            else:
                x_t_list = [self.conv_t(x_sp[:, :, i, :, :]) for i in range(0, shape_sp[self.axis_conv_t])]

            x_t = tf.stack(x_t_list, axis=self.axis_conv_t)  # stack time
        else:  # channels last [batch, time, x,y, chs]

            x_sp_list = [self.conv_xy(x[:, i, :, :, :]) for i in
                         range(0, self.shape[1])]  # split 'time' dimension, and 3D conv (depthwise) for each
            x_sp = tf.stack(x_sp_list, axis=1)
            shape_sp = x_sp.shape

            if self.axis_conv_t == 2:
                x_t_list = [self.conv_t(x_sp[:, :, i, :, :]) for i in range(0, shape_sp[self.axis_conv_t])]
            elif self.axis_conv_t == 3:
                x_t_list = [self.conv_t(x_sp[:, :, :, i, :]) for i in range(0, shape_sp[self.axis_conv_t])]
            elif self.axis_conv_t == 4:
                x_t_list = [self.conv_t(x_sp[:, :, :, :, i]) for i in range(0, shape_sp[self.axis_conv_t])]

            else:
                x_t_list = [self.conv_t(x_sp[:, i, :, :, :]) for i in range(0, shape_sp[self.axis_conv_t])]

            x_t = tf.stack(x_t_list, axis=self.axis_conv_t)  # stack time
        return x_t


class Conv2DtTranspose(tf.keras.layers.Layer):
    def __init__(self,
                 filters,  # out
                 intermediate_filters,
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
                 axis_conv_t=2,
                 zero_mean=True,
                 bound_norm=True,
                 pad=True,
                 **kwargs):
        super(Conv2DtTranspose, self).__init__()

        self.intermediate_filters = intermediate_filters
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.shape = shapes
        self.axis_conv_t = axis_conv_t

        kernel_size = validate_input_dimension('2Dt', kernel_size)
        strides = validate_input_dimension('2Dt', strides)
        dilation_rate = validate_input_dimension('2Dt', dilation_rate)
        self.conv_xy_filters = filters
        self.conv_xy = ComplexConv2DTranspose(
            filters=self.conv_xy_filters,
            kernel_size=(kernel_size[1], kernel_size[2]),
            strides=(int(strides[1]), int(strides[2])),
            padding=padding,
            data_format=data_format,
            dilation_rate=(int(dilation_rate[1]), int(dilation_rate[2])),
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
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=padding,
            data_format=data_format,
            dilation_rate=(dilation_rate[0], 1),
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

    def build(self, input_shape):
        self.shape = input_shape
        input_shape = list(input_shape)
        shape_xy = input_shape.copy()
        shape_t = input_shape.copy()


        if self.data_format == 'channels_first': # [batch,  channels, time, x,y]
            shape_t.pop(self.axis_conv_t) # pop selected axis
            shape_xy.pop(2) # [batch, channels, x,y]
            shape_xy[1] = self.intermediate_filters # second channel num= intermediate_filters from conv_t output


        else:
            # channels last  [batch, time,  x,y,channels]
            shape_t.pop(self.axis_conv_t)
            shape_xy.pop(1)  # [batch, x,y, channels]
            shape_xy[-1] = self.intermediate_filters     # last channel num = intermediate_filters from conv_t output


        self.conv_xy.build(shape_xy)
        self.conv_t.build(shape_t)

    def call(self, x):
        if self.data_format == 'channels_first':  # [batch, chs, time, x,y]

            if self.axis_conv_t == 1:
                x_t_list = [self.conv_t(x[:, i, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 3:
                x_t_list = [self.conv_t(x[:, :, :, i, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 4:
                x_t_list = [self.conv_t(x[:, :, :, :, i]) for i in range(0, self.shape[self.axis_conv_t])]

            else:
                x_t_list = [self.conv_t(x[:, :, i, :, :]) for i in range(0, self.shape[self.axis_conv_t])]

            x_t = tf.stack(x_t_list, axis=self.axis_conv_t)  # stack time

            x_sp_list = [self.conv_xy(x_t[:, :, i, :, :]) for i in
                         range(0, self.shape[2])]  # split 'time' dimension, and 3D conv (depthwise) for each
            x_sp = tf.stack(x_sp_list, axis=2)


        else:  # channels last #[batch,  time, x,y,z,chs]

            if self.axis_conv_t == 2:
                x_t_list = [self.conv_t(x[:, :, i, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 3:
                x_t_list = [self.conv_t(x[:, :, :, i, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 4:
                x_t_list = [self.conv_t(x[:, :, :, :, i]) for i in range(0, self.shape[self.axis_conv_t])]

            else:
                x_t_list = [self.conv_t(x[:, i, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]

            x_t = tf.stack(x_t_list, axis=self.axis_conv_t)  # stack time

            x_sp_list = [self.conv_xy(x_t[:, i, :, :, :]) for i in
                         range(0, self.shape[1])]  # split 'time' dimension, and 3D conv (depthwise) for each
            x_sp = tf.stack(x_sp_list, axis=1)
        return x_sp


if __name__ == "__main__":
    """conv 2Dt """
    nBatch = 2
    M = 32
    N = 32
    T = 8
    nf_in = 3
    nf_out = 20
    shape = [nBatch, T, M, N, nf_in]

    ksz = (3, 5, 5)
    ksz = validate_input_dimension('2Dt', ksz)

    nf_inter = np.ceil(
        (nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2]  + nf_out * ksz[0])).astype(np.int32)

    model = Conv2Dt(nf_out, nf_inter, kernel_size=ksz, shapes=shape, axis_conv_t=3)

    x_real = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x_imag = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x = tf.complex(x_real, x_imag)
    Kx = model(x).numpy()
    print('Conv2Dt input_shape:',shape,'output_shape,channels_last, strides=1:',Kx.shape)

    model2 = Conv2Dt(nf_out, nf_inter, kernel_size=ksz, shapes=shape, axis_conv_t=3, strides=( 2, 2, 2))  # strides =2
    x_real = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x_imag = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x = tf.complex(x_real, x_imag)
    Kx = model2(x).numpy()
    print('Conv2Dt input_shape:',shape,'output_shape,channels_last, strides=2:',Kx.shape)


    # channel first
    shape = [nBatch, nf_in, T, M, N]

    ksz = (3, 5, 5)
    ksz = validate_input_dimension('2Dt', ksz)

    nf_inter = np.ceil(
        (nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2]  + nf_out * ksz[0])).astype(np.int32)

    model = Conv2Dt(nf_out, nf_inter, kernel_size=ksz, shapes=shape, data_format='channels_first', axis_conv_t=2)

    x = tf.random.normal(shape)
    Kx = model(x).numpy()
    print('Conv2Dt input_shape:',shape,'output_shape,channels_first, strides=1:',Kx.shape)


    print('finish conv 2Dt\n')


    """conv 2Dt Transpose"""

    nf_in = 18
    nf_out = 3
    shape = [nBatch, T, M, N, nf_in]
    ksz = (3, 5, 5)
    ksz = validate_input_dimension('2Dt', ksz)

    nf_inter = np.ceil(
        (nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] + nf_out * ksz[0])).astype(np.int32)

    model = Conv2DtTranspose(nf_out, nf_inter, kernel_size=ksz, shapes=shape, axis_conv_t=3)


    x_real = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x_imag = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x = tf.complex(x_real, x_imag)
    Kx = model(x).numpy()
    print('Conv2DtTranspose input_shape:',shape,'output_shape, channels_last,strides=1:',Kx.shape)

    model2 = Conv2DtTranspose(nf_out, nf_inter, kernel_size=ksz, shapes=shape, axis_conv_t=3, strides=(2, 2, 2))  # strides =2
    x_real = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x_imag = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x = tf.complex(x_real, x_imag)
    Kx = model2(x).numpy()
    print('Conv2DtTranspose input_shape:',shape,'output_shape, channels_last,strides=2:',Kx.shape)


    ## channel first
    shape = [nBatch, nf_in, T, M, N]

    ksz = (3, 5, 5)
    ksz = validate_input_dimension('2Dt', ksz)

    nf_inter = np.ceil(
        (nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2]  + nf_out * ksz[0])).astype(np.int32)

    model = Conv2DtTranspose(nf_out, nf_inter, kernel_size=ksz, shapes=shape, data_format='channels_first', axis_conv_t=2)

    x = tf.random.normal(shape)
    Kx = model(x).numpy()
    print('Conv2DtTranspose input_shape:',shape,'output_shape, channels_first,strides=1:',Kx.shape)

    print('finish conv 2DtTranspose')






