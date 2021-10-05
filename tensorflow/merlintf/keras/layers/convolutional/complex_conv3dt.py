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

#import merlintf

'''
def get_ndim(dim):
    if dim == '2D':
        n_dim = 2
    elif dim == '3D':
        n_dim = 3
    elif dim == '2Dt':
        n_dim = 3
    elif dim == '3Dt':
        n_dim = 4
    else: n_dim=0
    return n_dim


def validate_input_dimension(dim, param):
    n_dim = merlintf.keras.utils.get_ndim(dim)
    if isinstance(param, tuple) or isinstance(param, list):
        if not len(param) == n_dim:
            raise RuntimeError("Parameter dimensions {} do not match requested dimensions {}!".format(len(param), n_dim))
        else:
            return param
    else:
        return tuple([param for _ in range(n_dim)])
'''


class Conv3Dt(tf.keras.layers.Layer):
    def __init__(self,
                 filters, # out
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
                 intermediate_filters=None,  # TODO: automatically calculate if no value is specified (i.e. default)!
                 **kwargs):
        super(Conv3Dt, self).__init__()

        self.intermediate_filters = intermediate_filters
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.shape=shapes
        self.axis_conv_t=axis_conv_t

        # TODO: should be handled outside and assumed to be correct here!
        kernel_size = merlintf.keras.utils.validate_input_dimension('3Dt', kernel_size)
        strides = merlintf.keras.utils.validate_input_dimension('3Dt', strides)
        dilation_rate = merlintf.keras.utils.validate_input_dimension('3Dt', dilation_rate)

        self.conv_xyz = ComplexConv3D(
            filters=intermediate_filters,
            kernel_size=(kernel_size[1], kernel_size[2], kernel_size[3]),
            strides=(int(strides[1]), int(strides[2]), int(strides[3])),
            padding=padding,
            data_format=data_format,
            dilation_rate=(int(dilation_rate[1]), int(dilation_rate[2]), int(dilation_rate[3])),
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

        if data_format=='channels_first':
            conv_t_filters = filters * self.shape[1]
        else:
            conv_t_filters = filters * self.shape[-1]

        self.conv_t = ComplexConv3D(
            filters=conv_t_filters,
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
        else: # channels last
            shape_xyz.pop(1)
            shape_t.pop(4)
            shape_t[-1] = self.intermediate_filters

        self.conv_xyz.build(shape_xyz)
        self.conv_t.build(shape_t)
        #if self.data_format == 'channels_first':
        #    self.conv_t.build(shape_t * self.shape[2])
        #else: # channels last
        #    self.conv_t.build(shape_t * self.shape[1])

    def call(self, x):
        if self.data_format == 'channels_first':  # [batch, chs, time, x,y,z]

            # TODO: check if faster option?
            shape_in = x.shape
            x_sp = tf.stack([x[:, :, i, :, :, :] for i in range(0, self.shape[2])], axis=0)
            x_sp = self.conv_xyz(x_sp)
            shape_sp = x_sp.shape
            x_sp_list = tf.split(x_sp, [shape_in[0]] + shape_sp[1:])  # should give a list of len = nTime and each element: [batch, chs, x, y, z]
            x_sp = tf.stack(x_sp_list, axis=2)

            #x_sp = tf.stack([self.conv_xyz(x[:, :, i, :, :, :]) for i in range(0, self.shape[2])], axis=2)  # split 'time' dimension, and 3D conv (depthwise) for each
            #x_sp = tf.stack(x_sp_list, axis=2)
            shape_sp = x_sp.shape

            # TODO: same stacking principle along batch dimension from above
            if self.axis_conv_t==1: # chs
                x_t_list =[self.conv_t(x_sp[:, i, :, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t==3: # x
                x_t_list = [self.conv_t(x_sp[:, :, :, i, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t==4: # y
                x_t_list = [self.conv_t(x_sp[:, :, :, :, i, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t==5: # z
                x_t_list = [self.conv_t(x_sp[:, :, :, :, :, i]) for i in range(0, self.shape[self.axis_conv_t])]
            else: # time
                x_t_list=[self.conv_t(x_sp[:, :, i, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]

            x_t = tf.stack(x_t_list, axis=self.axis_conv_t) # stack time
        else: # channels last #[batch, time, x,y,z,chs]

            # TODO: check if faster option?
            shape_in = x.shape
            x_sp = tf.stack([x[:, i, :, :, :, :] for i in range(0, self.shape[2])], axis=0)
            x_sp = self.conv_xyz(x_sp)
            shape_sp = x_sp.shape
            x_sp_list = tf.split(x_sp, [shape_in[0]] + shape_sp[1:])  # should give a list of len = nTime and each element: [batch, x, y, z, chs]
            x_sp = tf.stack(x_sp_list, axis=1)

            #x_sp_list = [self.conv_xyz(x[:, i, :, :, :, :]) for i in range(0, self.shape[1])]  # split 'time' dimension, and 3D conv (depthwise) for each
            #x_sp = tf.stack(x_sp_list, axis=1)

            # TODO: same stacking principle along batch dimension from above
            if self.axis_conv_t == 2: # x
                x_t_list = [self.conv_t(x_sp[:, :, i, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 3: # y
                x_t_list = [self.conv_t(x_sp[:, :, :, i, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 4: # z
                x_t_list = [self.conv_t(x_sp[:, :, :, :, i, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 5: # chs
                x_t_list = [self.conv_t(x_sp[:, :, :, :, :, i]) for i in range(0, self.shape[self.axis_conv_t])]
            else: # time
                x_t_list = [self.conv_t(x_sp[:, i, :, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]

            x_t = tf.stack(x_t_list, axis=self.axis_conv_t)  # stack time
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

        self.intermediate_filters = intermediate_filters
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.shape = shapes
        self.axis_conv_t = axis_conv_t

        kernel_size = validate_input_dimension('3Dt', kernel_size)
        strides = validate_input_dimension('3Dt', strides)
        dilation_rate = validate_input_dimension('3Dt', dilation_rate)

        self.conv_xyz_filters = filters
        self.conv_xyz = ComplexConv3DTranspose(
            filters=self.conv_xyz_filters,
            kernel_size=(kernel_size[1], kernel_size[2], kernel_size[3]),
            strides=(int(strides[1]), int(strides[2]), int(strides[3])),
            padding=padding,
            data_format=data_format,
            dilation_rate=(int(dilation_rate[1]), int(dilation_rate[2]), int(dilation_rate[3])),
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

            **kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        input_shape = list(input_shape)
        shape_xyz = input_shape.copy()
        shape_t = input_shape.copy()


        if self.data_format == 'channels_first': # [batch, channels, time, x,y,z]
            shape_t.pop(self.axis_conv_t) # pop selected axis
            shape_xyz.pop(2) # [batch, channels, x,y,z]
            shape_xyz[1] = self.intermediate_filters # second channel num= intermediate_filters from conv_t output


        else:  # channels last # [batch, time,  x,y,z,channels]
            shape_t.pop(self.axis_conv_t)
            shape_xyz.pop(1) # [batch, x,y,z, channels]
            shape_xyz[-1] = self.intermediate_filters # last channel num= intermediate_filters from conv_t output

        self.conv_xyz.build(shape_xyz)
        self.conv_t.build(shape_t)

    def call(self, x):
        if self.data_format == 'channels_first':  # [batch, chs, time, x,y,z]

            # TODO: same modifications
            if self.axis_conv_t == 1:
                x_t_list = [self.conv_t(x[:, i, :, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 3:
                x_t_list = [self.conv_t(x[:, :, :, i, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 4:
                x_t_list = [self.conv_t(x[:, :, :, :, i, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 5:
                x_t_list = [self.conv_t(x[:, :, :, :, :, i]) for i in range(0, self.shape[self.axis_conv_t])]
            else:
                x_t_list = [self.conv_t(x[:, :, i, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]

            x_t = tf.stack(x_t_list, axis=self.axis_conv_t)  # stack time

            x_sp_list = [self.conv_xyz(x_t[:, :, i, :, :, :]) for i in
                         range(0, self.shape[2])]  # split 'time' dimension, and 3D conv (depthwise) for each
            x_sp = tf.stack(x_sp_list, axis=2)


        else:

            if self.axis_conv_t == 2:
                x_t_list = [self.conv_t(x[:, :, i, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 3:
                x_t_list = [self.conv_t(x[:, :, :, i, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 4:

                x_t_list = [self.conv_t(x[:, :, :, :, i, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 5:
                x_t_list = [self.conv_t(x[:, :, :, :, :, i]) for i in range(0, self.shape[self.axis_conv_t])]
            else:
                x_t_list = [self.conv_t(x[:, i, :, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]

            x_t = tf.stack(x_t_list, axis=self.axis_conv_t)  # stack time

            x_sp_list = [self.conv_xyz(x_t[:, i, :, :, :, :]) for i in
                         range(0, self.shape[1])]  # split 'time' dimension, and 3D conv (depthwise) for each
            x_sp = tf.stack(x_sp_list, axis=1)
        return x_sp


if __name__ == "__main__":
    # channel last
    nBatch = 2
    M = 48
    N = 32
    D = 12
    T = 8
    nf_in = 2
    nf_out = 16
    shape = [nBatch, T, M, N, D, nf_in]

    ksz = (3, 5, 5, 5)
    ksz = merlintf.keras.utils.validate_input_dimension('3Dt', ksz)

    # TODO: this calculation should be put into the build call of the filter and automatically calculated if specified nf_inter is None
    nf_inter = np.ceil(
        (nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] * ksz[3] + nf_out * ksz[0])).astype(np.int32)



    model = Conv3Dt(nf_out, nf_inter, kernel_size=ksz,shapes=shape,axis_conv_t=4)

    x_real =tf.cast(tf.random.normal(shape),dtype=tf.float32)
    x_imag =tf.cast(tf.random.normal(shape),dtype=tf.float32)
    x=tf.complex(x_real,x_imag)
    Kx = model(x).numpy()
    print(Kx.shape)


    print('------------')
    model2 = Conv3Dt(nf_out, nf_inter, kernel_size=ksz,shapes=shape,axis_conv_t=3,strides=(2,2,2,2)) # strides =2
    x_real =tf.cast(tf.random.normal(shape),dtype=tf.float32)
    x_imag =tf.cast(tf.random.normal(shape),dtype=tf.float32)
    x=tf.complex(x_real,x_imag)
    Kx = model(x).numpy()
    print(Kx.shape)

    print('================')
    # channel first
    shape = [nBatch,nf_in, T, M, N, D]

    ksz = (3, 5, 5, 5)
    ksz = validate_input_dimension('3Dt', ksz)

    nf_inter = np.ceil(
        (nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] * ksz[3] + nf_out * ksz[0])).astype(np.int32)


    model = Conv3Dt(nf_out, nf_inter, kernel_size=ksz,shapes=shape,data_format='channels_first',axis_conv_t=4)

    x =tf.random.normal(shape)
    Kx = model(x).numpy()
    print(Kx.shape)


    # stride=2
    model2 = Conv3Dt(nf_out, nf_inter, kernel_size=ksz, shapes=shape, axis_conv_t=3, strides=(2, 2, 2, 2))  # strides=2
    x_real = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x_imag = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x = tf.complex(x_real, x_imag)
    Kx = model2(x).numpy()
    print('Conv3Dt input_shape:', shape, 'output_shape,channels_last, strides=2:', Kx.shape)


    # channel first
    shape = [nBatch, nf_in, T, M, N, D]

    ksz = (3, 5, 5, 5)
    ksz = validate_input_dimension('3Dt', ksz)

    nf_inter = np.ceil(
        (nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] * ksz[3] + nf_out * ksz[0])).astype(np.int32)

    model = Conv3Dt(nf_out, nf_inter, kernel_size=ksz, shapes=shape, data_format='channels_first', axis_conv_t=4)

    x = tf.random.normal(shape)
    Kx = model(x).numpy()
    print('Conv3Dt input_shape:', shape, 'output_shape,channels_first, strides=1:', Kx.shape)



    """conv 3Dt Transpose"""
    nf_in = 18
    nf_out = 3
    shape = [nBatch, T, M, N, D, nf_in]
    ksz = (3, 5, 5, 5)
    ksz = validate_input_dimension('3Dt', ksz)

    nf_inter = np.ceil(
        (nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] * ksz[3] + nf_out * ksz[0])).astype(np.int32)

    model = Conv3DtTranspose(nf_out, nf_inter, kernel_size=ksz, shapes=shape, axis_conv_t=3)

    x_real = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x_imag = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x = tf.complex(x_real, x_imag)
    Kx = model(x).numpy()
    print('Conv3DtTranspose input_shape:', shape, ' output_shape, channels_last,strides=1:', Kx.shape)

    model2 = Conv3DtTranspose(nf_out, nf_inter, kernel_size=ksz, shapes=shape, axis_conv_t=3,
                              strides=(2, 2, 2, 2))  # strides =2
    x_real = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x_imag = tf.cast(tf.random.normal(shape), dtype=tf.float32)
    x = tf.complex(x_real, x_imag)
    Kx = model2(x).numpy()
    print('Conv3DtTranspose input_shape:', shape, 'output_shape, channels_last,strides=2:', Kx.shape)

    ## channel first
    shape = [nBatch, nf_in, T, M, N, D]

    ksz = (3, 5, 5, 5)
    ksz = validate_input_dimension('3Dt', ksz)

    nf_inter = np.ceil(
        (nf_out * nf_in * np.prod(ksz)) / (nf_in * ksz[1] * ksz[2] * ksz[3] + nf_out * ksz[0])).astype(np.int32)

    model = Conv3DtTranspose(nf_out, nf_inter, kernel_size=ksz, shapes=shape, data_format='channels_first',
                             axis_conv_t=4)

    x = tf.random.normal(shape)
    Kx = model(x).numpy()
    print('Conv3DtTranspose input_shape:', shape, 'output_shape, channels_first,strides=1:', Kx.shape)