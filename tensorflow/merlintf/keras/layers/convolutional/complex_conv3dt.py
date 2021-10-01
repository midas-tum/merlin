import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import conv_utils
import conv_act as activations
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

import merlintf

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

# TODO: can be inherited from complex_convolutional.py
class ComplexConv(Layer):

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=(1,1,1),
               padding='valid',
               data_format=None,
               dilation_rate=(1,1,1),
               groups=1,
               activation=None,
               use_bias=True,
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
               **kwargs):
    super(ComplexConv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank

    if isinstance(filters, float):
      filters = int(filters)
    self.filters = filters
    self.groups = groups or 1
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')

    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(min_ndim=self.rank + 2)

    self._validate_init()
    self._is_causal = self.padding == 'causal'
    self._channels_first = self.data_format == 'channels_first'
    self._tf_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)

  def _validate_init(self):
    if self.filters is not None and self.filters % self.groups != 0:
      raise ValueError(
          'The number of filters must be evenly divisible by the number of '
          'groups. Received: groups={}, filters={}'.format(
              self.groups, self.filters))

    if not all(self.kernel_size):
      raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                       'Received: %s' % (self.kernel_size,))

    if (self.padding == 'causal' and not isinstance(self,
                                                    (ComplexConv1D))):
      raise ValueError('Causal padding is only supported for `ComplexConv1D`'
                       'and `SeparableComplexConv1D`.')

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    if input_channel % self.groups != 0:
      raise ValueError(
          'The number of input channels must be evenly divisible by the number '
          'of groups. Received groups={}, but the input has {} channels '
          '(full input shape is {}).'.format(self.groups, input_channel,
                                             input_shape))
    kernel_shape = self.kernel_size + (input_channel // self.groups,
                                       self.filters) + (2,)

    self._kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self._bias = self.add_weight(
          name='bias',
          shape=(self.filters,2,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self._bias = None
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                axes={channel_axis: input_channel})

    # ComplexConvert Keras formats to TF native formats.
    if self.padding == 'causal':
      tf_padding = 'VALID'  # Causal padding handled in `call`.
    elif isinstance(self.padding, six.string_types):
      tf_padding = self.padding.upper()
    else:
      tf_padding = self.padding
    tf_dilations = list(self.dilation_rate)
    tf_strides = list(self.strides)

    tf_op_name = self.__class__.__name__
    if tf_op_name == 'ComplexConv1D':
      tf_op_name = 'conv1d'  # Backwards compat.

    self._convolution_op = functools.partial(
        nn_ops.convolution_v2,
        strides=tf_strides,
        padding=tf_padding,
        dilations=tf_dilations,
        data_format=self._tf_data_format,
        name=tf_op_name)
    self.built = True

  def _complex_convolution_op(self, x, weight):
    xre = tf.math.real(x)
    xim = tf.math.imag(x)

    wre = tf.math.real(weight)
    wim = tf.math.imag(weight)

    conv_rr = self._convolution_op(xre, wre)
    conv_ii = self._convolution_op(xim, wim)
    conv_ri = self._convolution_op(xre, wim)
    conv_ir = self._convolution_op(xim, wre)

    conv_re = conv_rr - conv_ii
    conv_im = conv_ir + conv_ri

    return tf.complex(conv_re, conv_im)

  @property
  def kernel(self):
      return tf.complex(self._kernel[...,0], self._kernel[...,1])

  @property
  def bias(self):
      if self.use_bias:
        return tf.complex(self._bias[...,0], self._bias[...,1])
      else:
        return self._bias

  def call(self, inputs):
    if self._is_causal:  # Apply causal padding to inputs for ComplexConv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

    outputs = self._complex_convolution_op(inputs, self.kernel)

    if self.use_bias:
      output_rank = outputs.shape.rank
      if self.rank == 1 and self._channels_first:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(self.bias, (1, self.filters, 1))
        outputs += bias
      else:
        # Handle multiple batch dimensions.
        if output_rank is not None and output_rank > 2 + self.rank:

          def _apply_fn(o):
            return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

          outputs = nn_ops.squeeze_batch_dims(
              outputs, _apply_fn, inner_rank=self.rank + 1)
        else:
          outputs = nn.bias_add(
              outputs, self.bias, data_format=self._tf_data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _spatial_output_shape(self, spatial_input_shape):
    return [
        conv_utils.conv_output_length(
            length,
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        for i, length in enumerate(spatial_input_shape)
    ]

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    batch_rank = len(input_shape) - self.rank - 1
    if self.data_format == 'channels_last':
      return tensor_shape.TensorShape(
          input_shape[:batch_rank]
          + self._spatial_output_shape(input_shape[batch_rank:-1])
          + [self.filters])
    else:
      return tensor_shape.TensorShape(
          input_shape[:batch_rank] + [self.filters] +
          self._spatial_output_shape(input_shape[batch_rank + 1:]))

  def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
    return False

  def get_config(self):
    config = {
        'filters':
            self.filters,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
            self.data_format,
        'dilation_rate':
            self.dilation_rate,
        'groups':
            self.groups,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    }
    base_config = super(ComplexConv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self, inputs):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if getattr(inputs.shape, 'ndims', None) is None:
      batch_rank = 1
    else:
      batch_rank = len(inputs.shape) - 2
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
      return -1 - self.rank
    else:
      return -1

  def _get_input_channel(self, input_shape):
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding


class ComplexConv3D(ComplexConv):
  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(ComplexConv3D, self).__init__(
        rank=3,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)



class Conv3Dt(tf.keras.layers.Layer):
    def __init__(self,
                 filters, # out
                 intermediate_filters, # TODO: not needed!
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

            x_sp = tf.stack([self.conv_xyz(x[:, :, i, :, :, :]) for i in range(0, self.shape[2])], axis=2)  # split 'time' dimension, and 3D conv (depthwise) for each

            if self.axis_conv_t==1:
                x_t_list =[self.conv_t(x_sp[:, i,:, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t==3:
                x_t_list = [self.conv_t(x_sp[:, :, :, i, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t==4:
                x_t_list = [self.conv_t(x_sp[:, :, :, :, i, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t==5:
                x_t_list = [self.conv_t(x_sp[:, :, :, :, :, i]) for i in range(0, self.shape[self.axis_conv_t])]
            else:
                x_t_list=[self.conv_t(x_sp[:, :, i, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]

            x_t = tf.stack(x_t_list, axis=self.axis_conv_t) # stack time
        else: # channels last #[batch, time, x,y,z,chs]

            x_sp_list = [self.conv_xyz(x[:, i, :, :, :, :]) for i in range(0, self.shape[1])]  # split 'time' dimension, and 3D conv (depthwise) for each
            x_sp = tf.stack(x_sp_list, axis=1)

            if self.axis_conv_t == 2:
                x_t_list = [self.conv_t(x_sp[:, :, i, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 3:
                x_t_list = [self.conv_t(x_sp[:, :, :, i, :, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 4:
                x_t_list = [self.conv_t(x_sp[:, :, :, :, i, :]) for i in range(0, self.shape[self.axis_conv_t])]
            elif self.axis_conv_t == 5:
                x_t_list = [self.conv_t(x_sp[:, :, :, :, :, i]) for i in range(0, self.shape[self.axis_conv_t])]
            else:
                x_t_list = [self.conv_t(x_sp[:, i, :, :, :, :]) for i in range(0, self.shape[self.axis_conv_t])]

            x_t = tf.stack(x_t_list, axis=self.axis_conv_t)  # stack time
        return x_t


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




