# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras convolution layers and image transformation layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
#from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
# from tensorflow.python.keras.layers.pooling import AveragePooling1D
# from tensorflow.python.keras.layers.pooling import AveragePooling2D
# from tensorflow.python.keras.layers.pooling import AveragePooling3D
# from tensorflow.python.keras.layers.pooling import MaxPooling1D
# from tensorflow.python.keras.layers.pooling import MaxPooling2D
# from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export
# pylint: disable=g-classes-have-attributes
import unittest

from merlintf.keras.layers.convolutional import complex_conv as complex_nn_ops
from merlintf.keras.layers import complex_act as activations
import merlintf

def ComplexConvolution(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'ComplexConvolution' + (str(identifier).upper() if len(identifier) == 2 else str(identifier[0:2]).upper() + str(identifier[-1]))
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret convolution function identifier: {}'.format(identifier))

def ComplexConvolutionTranspose(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'ComplexConvolution' + (str(identifier).upper() if len(identifier) == 2 else str(identifier[0:2]).upper() + str(identifier[-1])) + 'Transpose'
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret convolution function identifier: {}'.format(identifier))

def UpSampling(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'UpSampling' + (str(identifier).upper() if len(identifier) == 2 else str(identifier[0:2]).upper() + str(identifier[-1]))
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret upsampling function identifier: {}'.format(identifier))

def ZeroPadding(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'ZeroPadding' + (str(identifier).upper() if len(identifier) == 2 else str(identifier[0:2]).upper() + str(identifier[-1]))
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret zeropadding function identifier: {}'.format(identifier))

def Cropping(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'Cropping' + (str(identifier).upper() if len(identifier) == 2 else str(identifier[0:2]).upper() + str(identifier[-1]))
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret cropping function identifier: {}'.format(identifier))

def deserialize(op):
    if op == 'ComplexConv1D' or op == 'ComplexConvolution1D':
        return ComplexConv1D
    elif op == 'ComplexConv2D' or op == 'ComplexConvolution2D':
        return ComplexConv2D
    elif op == 'ComplexConv2Dt' or op == 'ComplexConvolution2Dt':
        return merlintf.keras.layers.convolutional.complex_conv2dt.ComplexConv2Dt
    elif op == 'ComplexConv3D' or op == 'ComplexConvolution3D':
        return ComplexConv3D
    elif op == 'ComplexConv3Dt' or op == 'ComplexConvolution3Dt':
        return merlintf.keras.layers.convolutional.complex_conv3dt.ComplexConv3Dt
    elif op == 'ComplexConv1DTranspose' or op == 'ComplexConvolution1DTranspose':
        return ComplexConv1DTranspose
    elif op == 'ComplexConv2DTranspose' or op == 'ComplexConvolution2DTranspose':
        return ComplexConv2DTranspose
    elif op == 'ComplexConv2DtTranspose' or op == 'ComplexConvolution2DtTranspose':
        return merlintf.keras.layers.convolutional.complex_conv2dt.ComplexConv2DtTranspose
    elif op == 'ComplexConv3DTranspose' or op == 'ComplexConvolution3DTranspose':
        return ComplexConv3DTranspose
    elif op == 'ComplexConv3DtTranspose' or op == 'ComplexConvolution3DtTranspose':
        return merlintf.keras.layers.convolutional.complex_conv3dt.ComplexConv3DtTranspose
    elif op == 'UpSampling1D':
        return UpSampling1D
    elif op == 'UpSampling2D':
        return UpSampling2D
    elif op == 'UpSampling3D' or op == 'UpSampling2Dt':
        return UpSampling3D
    elif op == 'UpSampling4D' or op == 'UpSampling3Dt':
        return UpSampling4D
    elif op == 'ZeroPadding1D':
        return ZeroPadding1D
    elif op == 'ZeroPadding2D':
        return ZeroPadding2D
    elif op == 'ZeroPadding3D' or op == 'ZeroPadding2Dt':
        return ZeroPadding3D
    elif op == 'ZeroPadding4D' or op == 'ZeroPadding3Dt':
        return ZeroPadding4D
    elif op == 'Cropping1D':
        return Cropping1D
    elif op == 'Cropping2D':
        return Cropping2D
    elif op == 'Cropping3D' or op == 'Cropping2Dt':
        return Cropping3D
    elif op == 'Cropping4D' or op == 'Cropping3Dt':
        return Cropping4D
    else:
        raise ValueError(f"Selected operation '{op}' not implemented in complex convolutional")

def serialize(func):
    return func.__name__

class ComplexConv(Layer):
  """Abstract N-D convolution layer (private, used as implementation base).
  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.
  Note: layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).
  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to 
      the left/right or up/down of the input such that output has the same 
      height/width dimension as the input. `"causal"` results in causal 
      (dilated) convolutions, e.g. `output[t]` does not depend on `input[t+1:]`.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    groups: A positive integer specifying the number of groups in which the
      input is split along the channel axis. Each group is convolved
      separately with `filters / groups` filters. The output is the
      concatenation of all the `groups` results along the channel axis.
      Input channels and `filters` must both be divisible by `groups`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
  """

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
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
        dtype=tf.keras.backend.floatx())
    if self.use_bias:
      self._bias = self.add_weight(
          name='bias',
          shape=(self.filters,2,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=tf.keras.backend.floatx())
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


#@keras_export('keras.layers.ComplexConv1D', 'keras.layers.ComplexConvolution1D')
class ComplexConv1D(ComplexConv):
  __doc__ = r"""Applies a complex-valued 1D convolution over an input signal composed of several input planes."""
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
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
    super(ComplexConv1D, self).__init__(
        rank=1,
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


#@keras_export('keras.layers.ComplexConv2D', 'keras.layers.ComplexConvolution2D')
class ComplexConv2D(ComplexConv):
  __doc__ = r"""2D convolution layer (e.g. spatial convolution over images).
  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.
  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.
  Examples:
  >>> # The inputs are 28x28 RGB images with `channels_last` and the batch
  >>> # size is 4.
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.ComplexConv2D(
  ... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 26, 26, 2)
  >>> # With `dilation_rate` as 2.
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.ComplexConv2D(
  ... 2, 3, activation='relu', dilation_rate=2, input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 24, 24, 2)
  >>> # With `padding` as "same".
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.ComplexConv2D(
  ... 2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 28, 28, 2)
  >>> # With extended batch shape [4, 7]:
  >>> input_shape = (4, 7, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.ComplexConv2D(
  ... 2, 3, activation='relu', input_shape=input_shape[2:])(x)
  >>> print(y.shape)
  (4, 7, 26, 26, 2)
  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of
      output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the height
      and width of the 2D convolution window. Can be a single integer to specify
      the same value for all spatial dimensions.
    strides: An integer or tuple/list of 2 integers, specifying the strides of
      the convolution along the height and width. Can be a single integer to
      specify the same value for all spatial dimensions. Specifying any stride
      value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs. `channels_last` corresponds
      to inputs with shape `(batch_size, height, width, channels)` while
      `channels_first` corresponds to inputs with shape `(batch_size, channels,
      height, width)`. It defaults to the `image_data_format` value found in
      your Keras config file at `~/.keras/keras.json`. If you never set it, then
      it will be `channels_last`.
    dilation_rate: an integer or tuple/list of 2 integers, specifying the
      dilation rate to use for dilated convolution. Can be a single integer to
      specify the same value for all spatial dimensions. Currently, specifying
      any `dilation_rate` value != 1 is incompatible with specifying any stride
      value != 1.
    groups: A positive integer specifying the number of groups in which the
      input is split along the channel axis. Each group is convolved separately
      with `filters / groups` filters. The output is the concatenation of all
      the `groups` results along the channel axis. Input channels and `filters`
      must both be divisible by `groups`.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied (see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (see
      `keras.initializers`).
    bias_initializer: Initializer for the bias vector (see
      `keras.initializers`).
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (see
      `keras.regularizers`).
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (see
      `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (see
      `keras.constraints`).
  Input shape:
    4+D tensor with shape: `batch_shape + (channels, rows, cols)` if
      `data_format='channels_first'`
    or 4+D tensor with shape: `batch_shape + (rows, cols, channels)` if
      `data_format='channels_last'`.
  Output shape:
    4+D tensor with shape: `batch_shape + (filters, new_rows, new_cols)` if
    `data_format='channels_first'` or 4+D tensor with shape: `batch_shape +
      (new_rows, new_cols, filters)` if `data_format='channels_last'`.  `rows`
      and `cols` values might have changed due to padding.
  Returns:
    A tensor of rank 4+ representing
    `activation(conv2d(inputs, kernel) + bias)`.
  Raises:
    ValueError: if `padding` is `"causal"`.
    ValueError: when both `strides > 1` and `dilation_rate > 1`.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
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
    super(ComplexConv2D, self).__init__(
        rank=2,
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


#@keras_export('keras.layers.ComplexConv3D', 'keras.layers.ComplexConvolution3D')
class ComplexConv3D(ComplexConv):
  __doc__ = r"""3D convolution layer (e.g. spatial convolution over volumes).
  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.
  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
  with a single channel,
  in `data_format="channels_last"`.
  Examples:
  >>> # The inputs are 28x28x28 volumes with a single channel, and the
  >>> # batch size is 4
  >>> input_shape =(4, 28, 28, 28, 1)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.ComplexConv3D(
  ... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 26, 26, 26, 2)
  >>> # With extended batch shape [4, 7], e.g. a batch of 4 videos of 3D frames,
  >>> # with 7 frames per video.
  >>> input_shape = (4, 7, 28, 28, 28, 1)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.ComplexConv3D(
  ... 2, 3, activation='relu', input_shape=input_shape[2:])(x)
  >>> print(y.shape)
  (4, 7, 26, 26, 26, 2)
  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of
      output filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the depth,
      height and width of the 3D convolution window. Can be a single integer to
      specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 3 integers, specifying the strides of
      the convolution along each spatial dimension. Can be a single integer to
      specify the same value for all spatial dimensions. Specifying any stride
      value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs. `channels_last` corresponds
      to inputs with shape `batch_shape + (spatial_dim1, spatial_dim2,
      spatial_dim3, channels)` while `channels_first` corresponds to inputs with
      shape `batch_shape + (channels, spatial_dim1, spatial_dim2,
      spatial_dim3)`. It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`. If you never set it, then it
      will be "channels_last".
    dilation_rate: an integer or tuple/list of 3 integers, specifying the
      dilation rate to use for dilated convolution. Can be a single integer to
      specify the same value for all spatial dimensions. Currently, specifying
      any `dilation_rate` value != 1 is incompatible with specifying any stride
      value != 1.
    groups: A positive integer specifying the number of groups in which the
      input is split along the channel axis. Each group is convolved separately
      with `filters / groups` filters. The output is the concatenation of all
      the `groups` results along the channel axis. Input channels and `filters`
      must both be divisible by `groups`.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied (see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (see
      `keras.initializers`).
    bias_initializer: Initializer for the bias vector (see
      `keras.initializers`).
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (see
      `keras.regularizers`).
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (see
      `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (see
      `keras.constraints`).
  Input shape:
    5+D tensor with shape: `batch_shape + (channels, conv_dim1, conv_dim2,
      conv_dim3)` if data_format='channels_first'
    or 5+D tensor with shape: `batch_shape + (conv_dim1, conv_dim2, conv_dim3,
      channels)` if data_format='channels_last'.
  Output shape:
    5+D tensor with shape: `batch_shape + (filters, new_conv_dim1,
      new_conv_dim2, new_conv_dim3)` if data_format='channels_first'
    or 5+D tensor with shape: `batch_shape + (new_conv_dim1, new_conv_dim2,
      new_conv_dim3, filters)` if data_format='channels_last'. `new_conv_dim1`,
      `new_conv_dim2` and `new_conv_dim3` values might have changed due to
      padding.
  Returns:
    A tensor of rank 5+ representing
    `activation(conv3d(inputs, kernel) + bias)`.
  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides > 1` and `dilation_rate > 1`.
  """

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


#@keras_export('keras.layers.ComplexConv1DTranspose',
#              'keras.layers.ComplexConvolution1DTranspose')
class ComplexConv1DTranspose(ComplexConv1D):
  """Transposed convolution layer (sometimes called Deconvolution).
  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.
  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 3)` for data with 128 time steps and 3 channels.
  Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer length of the 1D convolution window.
    strides: An integer specifying the stride of the convolution along the
      time dimension. Specifying a stride value != 1 is incompatible with
      specifying a `dilation_rate` value != 1. Defaults to 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    output_padding: An integer specifying the amount of padding along
      the time dimension of the output tensor.
      The amount of output padding must be lower than the stride.
      If set to `None` (default), the output shape is inferred.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, length)`.
    dilation_rate: an integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying a `dilation_rate` value != 1 is
      incompatible with specifying a stride value != 1.
      Also dilation rate larger than 1 is not currently supported.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).
  Input shape:
    3D tensor with shape:
    `(batch_size, steps, channels)`
  Output shape:
    3D tensor with shape:
    `(batch_size, new_steps, filters)`
    If `output_padding` is specified:
    ```
    new_timesteps = ((timesteps - 1) * strides + kernel_size -
    2 * padding + output_padding)
    ```
  Returns:
    A tensor of rank 3 representing
    `activation(conv1dtranspose(inputs, kernel) + bias)`.
  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  References:
    - [A guide to convolution arithmetic for deep learning](
      https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional Networks](
      https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               output_padding=None,
               data_format=None,
               dilation_rate=1,
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
    super(ComplexConv1DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 1, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 3:
      raise ValueError('Inputs should have rank 3. Received input shape: ' +
                       str(input_shape))
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim, 2)

    self._kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=tf.keras.backend.floatx())
    if self.use_bias:
      self._bias = self.add_weight(
          name='bias',
          shape=(self.filters, 2, ),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=tf.keras.backend.floatx())
    else:
      self._bias = None
    self.built = True

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      t_axis = 2
    else:
      t_axis = 1

    length = inputs_shape[t_axis]
    if self.output_padding is None:
      output_padding = None
    else:
      output_padding = self.output_padding[0]

    # Infer the dynamic output shape:
    out_length = conv_utils.deconv_output_length(
        length, self.kernel_size[0], padding=self.padding,
        output_padding=output_padding, stride=self.strides[0],
        dilation=self.dilation_rate[0])
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_length)
    else:
      output_shape = (batch_size, out_length, self.filters)
    data_format = conv_utils.convert_data_format(self.data_format, ndim=3)

    output_shape_tensor = array_ops.stack(output_shape)
    outputs = complex_nn_ops.complex_conv1d_transpose(
        inputs,
        self.kernel,
        output_shape_tensor,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=data_format,
        dilations=self.dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, t_axis = 1, 2
    else:
      c_axis, t_axis = 2, 1

    if self.output_padding is None:
      output_padding = None
    else:
      output_padding = self.output_padding[0]
    output_shape[c_axis] = self.filters
    output_shape[t_axis] = conv_utils.deconv_output_length(
        output_shape[t_axis],
        self.kernel_size[0],
        padding=self.padding,
        output_padding=output_padding,
        stride=self.strides[0],
        dilation=self.dilation_rate[0])
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(ComplexConv1DTranspose, self).get_config()
    config['output_padding'] = self.output_padding
    return config


#@keras_export('keras.layers.ComplexConv2DTranspose',
#              'keras.layers.ComplexConvolution2DTranspose')
class ComplexConv2DTranspose(ComplexConv2D):
  """Transposed convolution layer (sometimes called Deconvolution).
  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.
  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.
  Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    output_padding: An integer or tuple/list of 2 integers,
      specifying the amount of padding along the height and width
      of the output tensor.
      Can be a single integer to specify the same value for all
      spatial dimensions.
      The amount of output padding along a given dimension must be
      lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).
  Input shape:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, rows, cols, channels)` if data_format='channels_last'.
  Output shape:
    4D tensor with shape:
    `(batch_size, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
    If `output_padding` is specified:
    ```
    new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
    output_padding[0])
    new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
    output_padding[1])
    ```
  Returns:
    A tensor of rank 4 representing
    `activation(conv2dtranspose(inputs, kernel) + bias)`.
  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  References:
    - [A guide to convolution arithmetic for deep
      learning](https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional
      Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               output_padding=None,
               data_format=None,
               dilation_rate=(1, 1),
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
    super(ComplexConv2DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 2, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank 4. Received input '
                       'shape: ' + str(input_shape))
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim, 2)

    self._kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=tf.keras.backend.floatx())
    if self.use_bias:
      self._bias = self.add_weight(
          name='bias',
          shape=(self.filters, 2),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=tf.keras.backend.floatx())
    else:
      self._bias = None
    self.built = True

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      h_axis, w_axis = 2, 3
    else:
      h_axis, w_axis = 1, 2

    # Use the constant height and weight when possible.
    # TODO(scottzhu): Extract this into a utility function that can be applied
    # to all convolutional layers, which currently lost the static shape
    # information due to tf.shape().
    height, width = None, None
    if inputs.shape.rank is not None:
      dims = inputs.shape.as_list()
      height = dims[h_axis]
      width = dims[w_axis]
    height = height if height is not None else inputs_shape[h_axis]
    width = width if width is not None else inputs_shape[w_axis]

    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_output_length(height,
                                                 kernel_h,
                                                 padding=self.padding,
                                                 output_padding=out_pad_h,
                                                 stride=stride_h,
                                                 dilation=self.dilation_rate[0])
    out_width = conv_utils.deconv_output_length(width,
                                                kernel_w,
                                                padding=self.padding,
                                                output_padding=out_pad_w,
                                                stride=stride_w,
                                                dilation=self.dilation_rate[1])
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, self.filters)

    output_shape_tensor = array_ops.stack(output_shape)
    outputs = complex_nn_ops.complex_conv2d_transpose(
        inputs,
        self.kernel,
        output_shape_tensor,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=self._tf_data_format,
        dilations=self.dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, h_axis, w_axis = 1, 2, 3
    else:
      c_axis, h_axis, w_axis = 3, 1, 2

    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    output_shape[c_axis] = self.filters
    output_shape[h_axis] = conv_utils.deconv_output_length(
        output_shape[h_axis],
        kernel_h,
        padding=self.padding,
        output_padding=out_pad_h,
        stride=stride_h,
        dilation=self.dilation_rate[0])
    output_shape[w_axis] = conv_utils.deconv_output_length(
        output_shape[w_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w,
        dilation=self.dilation_rate[1])
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(ComplexConv2DTranspose, self).get_config()
    config['output_padding'] = self.output_padding
    return config


#@keras_export('keras.layers.ComplexConv3DTranspose',
#              'keras.layers.ComplexConvolution3DTranspose')
class ComplexConv3DTranspose(ComplexConv3D):
  """Transposed convolution layer (sometimes called Deconvolution).
  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.
  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 128, 3)` for a 128x128x128 volume with 3 channels
  if `data_format="channels_last"`.
  Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the convolution along the depth, height
        and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
    output_padding: An integer or tuple/list of 3 integers,
      specifying the amount of padding along the depth, height, and
      width.
      Can be a single integer to specify the same value for all
      spatial dimensions.
      The amount of output padding along a given dimension must be
      lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, depth, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 3 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (
      see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (
      see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).
  Input shape:
    5D tensor with shape:
    `(batch_size, channels, depth, rows, cols)` if data_format='channels_first'
    or 5D tensor with shape:
    `(batch_size, depth, rows, cols, channels)` if data_format='channels_last'.
  Output shape:
    5D tensor with shape:
    `(batch_size, filters, new_depth, new_rows, new_cols)` if
      data_format='channels_first'
    or 5D tensor with shape:
    `(batch_size, new_depth, new_rows, new_cols, filters)` if
      data_format='channels_last'.
    `depth` and `rows` and `cols` values might have changed due to padding.
    If `output_padding` is specified::
    ```
    new_depth = ((depth - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
    output_padding[0])
    new_rows = ((rows - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
    output_padding[1])
    new_cols = ((cols - 1) * strides[2] + kernel_size[2] - 2 * padding[2] +
    output_padding[2])
    ```
  Returns:
    A tensor of rank 5 representing
    `activation(conv3dtranspose(inputs, kernel) + bias)`.
  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  References:
    - [A guide to convolution arithmetic for deep
      learning](https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional
      Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               output_padding=None,
               data_format=None,
               dilation_rate=(1, 1, 1),
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
    super(ComplexConv3DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 3, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 5:
      raise ValueError('Inputs should have rank 5, received input shape:',
                       str(input_shape))
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined, found None: ' + str(input_shape))
    input_dim = int(input_shape[channel_axis])
    kernel_shape = self.kernel_size + (self.filters, input_dim, 2)
    self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})

    self._kernel = self.add_weight(
        'kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=tf.keras.backend.floatx())
    if self.use_bias:
      self._bias = self.add_weight(
          'bias',
          shape=(self.filters, 2, ),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=tf.keras.backend.floatx())
    else:
      self._bias = None
    self.built = True

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      d_axis, h_axis, w_axis = 2, 3, 4
    else:
      d_axis, h_axis, w_axis = 1, 2, 3

    depth = inputs_shape[d_axis]
    height = inputs_shape[h_axis]
    width = inputs_shape[w_axis]

    kernel_d, kernel_h, kernel_w = self.kernel_size
    stride_d, stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_d = out_pad_h = out_pad_w = None
    else:
      out_pad_d, out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_depth = conv_utils.deconv_output_length(depth,
                                                kernel_d,
                                                padding=self.padding,
                                                output_padding=out_pad_d,
                                                stride=stride_d)
    out_height = conv_utils.deconv_output_length(height,
                                                 kernel_h,
                                                 padding=self.padding,
                                                 output_padding=out_pad_h,
                                                 stride=stride_h)
    out_width = conv_utils.deconv_output_length(width,
                                                kernel_w,
                                                padding=self.padding,
                                                output_padding=out_pad_w,
                                                stride=stride_w)
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_depth, out_height,
                      out_width)
      strides = (1, 1, stride_d, stride_h, stride_w)
    else:
      output_shape = (batch_size, out_depth, out_height, out_width,
                      self.filters)
      strides = (1, stride_d, stride_h, stride_w, 1)

    output_shape_tensor = array_ops.stack(output_shape)
    outputs = complex_nn_ops.complex_conv3d_transpose(
        inputs,
        self.kernel,
        output_shape_tensor,
        strides,
        data_format=conv_utils.convert_data_format(self.data_format, ndim=5),
        padding=self.padding.upper())

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, d_axis, h_axis, w_axis = 1, 2, 3, 4
    else:
      c_axis, d_axis, h_axis, w_axis = 4, 1, 2, 3

    kernel_d, kernel_h, kernel_w = self.kernel_size
    stride_d, stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_d = out_pad_h = out_pad_w = None
    else:
      out_pad_d, out_pad_h, out_pad_w = self.output_padding

    output_shape[c_axis] = self.filters
    output_shape[d_axis] = conv_utils.deconv_output_length(
        output_shape[d_axis],
        kernel_d,
        padding=self.padding,
        output_padding=out_pad_d,
        stride=stride_d)
    output_shape[h_axis] = conv_utils.deconv_output_length(
        output_shape[h_axis],
        kernel_h,
        padding=self.padding,
        output_padding=out_pad_h,
        stride=stride_h)
    output_shape[w_axis] = conv_utils.deconv_output_length(
        output_shape[w_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w)
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(ComplexConv3DTranspose, self).get_config()
    config.pop('dilation_rate')
    config['output_padding'] = self.output_padding
    return config

#@keras_export('keras.layers.UpSampling1D')
class UpSampling1D(Layer):
  """Upsampling layer for 1D inputs.
  Repeats each temporal step `size` times along the time axis.
  Examples:
  >>> input_shape = (2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1  2]
    [ 3  4  5]]
   [[ 6  7  8]
    [ 9 10 11]]]
  >>> y = tf.keras.layers.UpSampling1D(size=2)(x)
  >>> print(y)
  tf.Tensor(
    [[[ 0  1  2]
      [ 0  1  2]
      [ 3  4  5]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 6  7  8]
      [ 9 10 11]
      [ 9 10 11]]], shape=(2, 4, 3), dtype=int64)
  Arguments:
    size: Integer. Upsampling factor.
  Input shape:
    3D tensor with shape: `(batch_size, steps, features)`.
  Output shape:
    3D tensor with shape: `(batch_size, upsampled_steps, features)`.
  """

  def __init__(self, size=2, **kwargs):
    super(UpSampling1D, self).__init__(**kwargs)
    self.size = int(size)
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    size = self.size * input_shape[1] if input_shape[1] is not None else None
    return tensor_shape.TensorShape([input_shape[0], size, input_shape[2]])

  def call(self, inputs):
    output = backend.repeat_elements(inputs, self.size, axis=1)
    return output

  def get_config(self):
    config = {'size': self.size}
    base_config = super(UpSampling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.UpSampling2D')
class UpSampling2D(Layer):
  """Upsampling layer for 2D inputs.
  Repeats the rows and columns of the data
  by `size[0]` and `size[1]` respectively.
  Examples:
  >>> input_shape = (2, 2, 1, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[[ 0  1  2]]
    [[ 3  4  5]]]
   [[[ 6  7  8]]
    [[ 9 10 11]]]]
  >>> y = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
  >>> print(y)
  tf.Tensor(
    [[[[ 0  1  2]
       [ 0  1  2]]
      [[ 3  4  5]
       [ 3  4  5]]]
     [[[ 6  7  8]
       [ 6  7  8]]
      [[ 9 10 11]
       [ 9 10 11]]]], shape=(2, 2, 2, 3), dtype=int64)
  Arguments:
    size: Int, or tuple of 2 integers.
      The upsampling factors for rows and columns.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    interpolation: A string, one of `nearest` or `bilinear`.
  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols)`
  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_rows, upsampled_cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_rows, upsampled_cols)`
  """

  def __init__(self,
               size=(2, 2),
               data_format=None,
               interpolation='nearest',
               **kwargs):
    super(UpSampling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 2, 'size')
    if interpolation not in {'nearest', 'bilinear'}:
      raise ValueError('`interpolation` argument should be one of `"nearest"` '
                       'or `"bilinear"`.')
    self.interpolation = interpolation
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      height = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      width = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], height, width])
    else:
      height = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      width = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], height, width, input_shape[3]])

  def call(self, inputs):
        x_re = backend.resize_images(
        tf.math.real(inputs), self.size[0], self.size[1], self.data_format,
        interpolation=self.interpolation)
        x_im = backend.resize_images(
        tf.math.imag(inputs), self.size[0], self.size[1], self.data_format,
        interpolation=self.interpolation)
        return tf.complex(x_re, x_im)

  def get_config(self):
    config = {
        'size': self.size,
        'data_format': self.data_format,
        'interpolation': self.interpolation
    }
    base_config = super(UpSampling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.UpSampling3D')
class UpSampling3D(Layer):
  """Upsampling layer for 3D inputs.
  Repeats the 1st, 2nd and 3rd dimensions
  of the data by `size[0]`, `size[1]` and `size[2]` respectively.
  Examples:
  >>> input_shape = (2, 1, 2, 1, 3)
  >>> x = tf.constant(1, shape=input_shape)
  >>> y = tf.keras.layers.UpSampling3D(size=2)(x)
  >>> print(y.shape)
  (2, 2, 4, 2, 3)
  Arguments:
    size: Int, or tuple of 3 integers.
      The upsampling factors for dim1, dim2 and dim3.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, dim1, dim2, dim3, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, dim1, dim2, dim3)`
  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
  """

  def __init__(self, size=(2, 2, 2), data_format=None, **kwargs):
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 3, 'size')
    self.input_spec = InputSpec(ndim=5)
    super(UpSampling3D, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      dim1 = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      dim2 = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      dim3 = self.size[2] * input_shape[
          4] if input_shape[4] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    else:
      dim1 = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      dim2 = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      dim3 = self.size[2] * input_shape[
          3] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    return backend.resize_volumes(
        inputs, self.size[0], self.size[1], self.size[2], self.data_format)

  def get_config(self):
    config = {'size': self.size, 'data_format': self.data_format}
    base_config = super(UpSampling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.UpSampling4D')
class UpSampling4D(Layer):
  """Upsampling layer for 4D inputs.
  Repeats the 1st, 2nd, 3rd and 4th dimensions
  of the data by `size[0]`, `size[1]`, `size[2]` and `size[3]` respectively.
  Examples:
  >>> input_shape = (2, 1, 2, 1, 1, 3)
  >>> x = tf.constant(1, shape=input_shape)
  >>> y = tf.keras.layers.UpSampling4D(size=2)(x)
  >>> print(y.shape)
  (2, 2, 4, 2, 2, 3)
  Arguments:
    size: Int, or tuple of 3 integers.
      The upsampling factors for dim1, dim2, dim3 and dim4.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, spatial_dim4, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3, spatial_dim4)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
  Input shape:
    6D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, dim1, dim2, dim3, dim4, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, dim1, dim2, dim3, dim4)`
  Output shape:
    6D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3, upsampled_dim4, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3, upsampled_dim4)`
  """

  def __init__(self, size=(2, 2, 2, 2), data_format=None, **kwargs):
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 4, 'size')
    self.input_spec = InputSpec(ndim=6)
    super(UpSampling4D, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      dim1 = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      dim2 = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      dim3 = self.size[2] * input_shape[
          4] if input_shape[4] is not None else None
      dim4 = self.size[3] * input_shape[
          5] if input_shape[5] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3, dim4])
    else:
      dim1 = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      dim2 = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      dim3 = self.size[2] * input_shape[
          3] if input_shape[3] is not None else None
      dim4 = self.size[3] * input_shape[
          4] if input_shape[4] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, dim4, input_shape[5]])

  def batch_concat_conv(self, inputs, sizes, axis=0):
        shape_in = inputs.shape
        x_list = tf.split(inputs, shape_in[axis], axis=axis)
        x = tf.concat(x_list, axis=0)
        x = tf.squeeze(x, axis=axis)
        x = backend.resize_volumes(
            x, sizes[0], sizes[1], sizes[2], self.data_format)
        x_list = tf.split(x, shape_in[axis], axis=0)
        return tf.stack(x_list, axis=axis)

  def call(self, inputs):
      if self.data_format == 'channels_first':
          axis = 2
      elif self.data_format == 'channels_last':
          axis = 1
      # xyz upsampling
      x = self.batch_concat_conv(inputs, self.size[1:], axis=axis)
      axis += 1
      # t upsampling
      return self.batch_concat_conv(x, (self.size[0], 1, 1), axis=axis)

  def get_config(self):
    config = {'size': self.size, 'data_format': self.data_format}
    base_config = super(UpSampling4D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.ZeroPadding1D')
class ZeroPadding1D(Layer):
  """Zero-padding layer for 1D input (e.g. temporal sequence).
  Examples:
  >>> input_shape = (2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1  2]
    [ 3  4  5]]
   [[ 6  7  8]
    [ 9 10 11]]]
  >>> y = tf.keras.layers.ZeroPadding1D(padding=2)(x)
  >>> print(y)
  tf.Tensor(
    [[[ 0  0  0]
      [ 0  0  0]
      [ 0  1  2]
      [ 3  4  5]
      [ 0  0  0]
      [ 0  0  0]]
     [[ 0  0  0]
      [ 0  0  0]
      [ 6  7  8]
      [ 9 10 11]
      [ 0  0  0]
      [ 0  0  0]]], shape=(2, 6, 3), dtype=int64)
  Arguments:
      padding: Int, or tuple of int (length 2), or dictionary.
          - If int:
          How many zeros to add at the beginning and end of
          the padding dimension (axis 1).
          - If tuple of int (length 2):
          How many zeros to add at the beginning and the end of
          the padding dimension (`(left_pad, right_pad)`).
  Input shape:
      3D tensor with shape `(batch_size, axis_to_pad, features)`
  Output shape:
      3D tensor with shape `(batch_size, padded_axis, features)`
  """

  def __init__(self, padding=1, **kwargs):
    super(ZeroPadding1D, self).__init__(**kwargs)
    self.padding = conv_utils.normalize_tuple(padding, 2, 'padding')
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    if input_shape[1] is not None:
      length = input_shape[1] + self.padding[0] + self.padding[1]
    else:
      length = None
    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

  def call(self, inputs):
    return backend.temporal_padding(inputs, padding=self.padding)

  def get_config(self):
    config = {'padding': self.padding}
    base_config = super(ZeroPadding1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.ZeroPadding2D')
class ZeroPadding2D(Layer):
  """Zero-padding layer for 2D input (e.g. picture).
  This layer can add rows and columns of zeros
  at the top, bottom, left and right side of an image tensor.
  Examples:
  >>> input_shape = (1, 1, 2, 2)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[[0 1]
     [2 3]]]]
  >>> y = tf.keras.layers.ZeroPadding2D(padding=1)(x)
  >>> print(y)
  tf.Tensor(
    [[[[0 0]
       [0 0]
       [0 0]
       [0 0]]
      [[0 0]
       [0 1]
       [2 3]
       [0 0]]
      [[0 0]
       [0 0]
       [0 0]
       [0 0]]]], shape=(1, 3, 4, 2), dtype=int64)
  Arguments:
    padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 2 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_height_pad, symmetric_width_pad)`.
      - If tuple of 2 tuples of 2 ints:
        interpreted as
        `((top_pad, bottom_pad), (left_pad, right_pad))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols)`
  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, padded_rows, padded_cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, padded_rows, padded_cols)`
  """

  def __init__(self, padding=(1, 1), data_format=None, **kwargs):
    super(ZeroPadding2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding))
    elif hasattr(padding, '__len__'):
      if len(padding) != 2:
        raise ValueError('`padding` should have two elements. '
                         'Found: ' + str(padding))
      height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                  '1st entry of padding')
      width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                 '2nd entry of padding')
      self.padding = (height_padding, width_padding)
    else:
      raise ValueError('`padding` should be either an int, '
                       'a tuple of 2 ints '
                       '(symmetric_height_pad, symmetric_width_pad), '
                       'or a tuple of 2 tuples of 2 ints '
                       '((top_pad, bottom_pad), (left_pad, right_pad)). '
                       'Found: ' + str(padding))
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
      else:
        rows = None
      if input_shape[3] is not None:
        cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
      else:
        cols = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
      else:
        rows = None
      if input_shape[2] is not None:
        cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
      else:
        cols = None
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def call(self, inputs):
    return backend.spatial_2d_padding(
        inputs, padding=self.padding, data_format=self.data_format)

  def get_config(self):
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.ZeroPadding3D')
class ZeroPadding3D(Layer):
  """Zero-padding layer for 3D data (spatial or spatio-temporal).
  Examples:
  >>> input_shape = (1, 1, 2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.ZeroPadding3D(padding=2)(x)
  >>> print(y.shape)
  (1, 5, 6, 6, 3)
  Arguments:
    padding: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 3 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
      - If tuple of 3 tuples of 2 ints:
        interpreted as
        `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
          right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,
          depth)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_axis_to_pad, second_axis_to_pad,
          third_axis_to_pad)`
  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, first_padded_axis, second_padded_axis, third_axis_to_pad,
          depth)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_padded_axis, second_padded_axis,
          third_axis_to_pad)`
  """

  def __init__(self, padding=(1, 1, 1), data_format=None, **kwargs):
    super(ZeroPadding3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding), (padding,
                                                               padding))
    elif hasattr(padding, '__len__'):
      if len(padding) != 3:
        raise ValueError('`padding` should have 3 elements. '
                         'Found: ' + str(padding))
      dim1_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                '1st entry of padding')
      dim2_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                '2nd entry of padding')
      dim3_padding = conv_utils.normalize_tuple(padding[2], 2,
                                                '3rd entry of padding')
      self.padding = (dim1_padding, dim2_padding, dim3_padding)
    else:
      raise ValueError(
          '`padding` should be either an int, '
          'a tuple of 3 ints '
          '(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad), '
          'or a tuple of 3 tuples of 2 ints '
          '((left_dim1_pad, right_dim1_pad),'
          ' (left_dim2_pad, right_dim2_pad),'
          ' (left_dim3_pad, right_dim2_pad)). '
          'Found: ' + str(padding))
    self.input_spec = InputSpec(ndim=5)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        dim1 = input_shape[2] + self.padding[0][0] + self.padding[0][1]
      else:
        dim1 = None
      if input_shape[3] is not None:
        dim2 = input_shape[3] + self.padding[1][0] + self.padding[1][1]
      else:
        dim2 = None
      if input_shape[4] is not None:
        dim3 = input_shape[4] + self.padding[2][0] + self.padding[2][1]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        dim1 = input_shape[1] + self.padding[0][0] + self.padding[0][1]
      else:
        dim1 = None
      if input_shape[2] is not None:
        dim2 = input_shape[2] + self.padding[1][0] + self.padding[1][1]
      else:
        dim2 = None
      if input_shape[3] is not None:
        dim3 = input_shape[3] + self.padding[2][0] + self.padding[2][1]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    return backend.spatial_3d_padding(
        inputs, padding=self.padding, data_format=self.data_format)

  def get_config(self):
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.ZeroPadding4D')
class ZeroPadding4D(Layer):
  """Zero-padding layer for 4D data (spatial or spatio-temporal).
  Examples:
  >>> input_shape = (1, 1, 2, 2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.ZeroPadding4D(padding=2)(x)
  >>> print(y.shape)
  (1, 5, 6, 6, 6, 3)
  Arguments:
    padding: Int, or tuple of 4 ints, or tuple of 4 tuples of 2 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 4 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad, symmetric_dim4_pad)`.
      - If tuple of 4 tuples of 2 ints:
        interpreted as
        `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
          right_dim2_pad), (left_dim3_pad, right_dim3_pad),
          (left_dim4_pad, right_dim4_pad))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, spatial_dim4, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3, spatial_dim4)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
  Input shape:
    6D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad, fourth_axis_to_pad
          depth)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_axis_to_pad, second_axis_to_pad,
          third_axis_to_pad, fourth_axis_to_pad)`
  Output shape:
    6D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, first_padded_axis, second_padded_axis, third_axis_to_pad, fourth_axis_to_pad
          depth)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_padded_axis, second_padded_axis,
          third_axis_to_pad, fourth_axis_to_pad)`
  """

  def __init__(self, padding=(1, 1, 1, 1), data_format=None, **kwargs):
    super(ZeroPadding4D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding),
                      (padding, padding), (padding, padding))
    elif hasattr(padding, '__len__'):
      if len(padding) != 4:
        raise ValueError('`padding` should have 4 elements. '
                         'Found: ' + str(padding))
      dim1_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                '1st entry of padding')
      dim2_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                '2nd entry of padding')
      dim3_padding = conv_utils.normalize_tuple(padding[2], 2,
                                                '3rd entry of padding')
      dim4_padding = conv_utils.normalize_tuple(padding[3], 2,
                                                '4th entry of padding')
      self.padding = (dim1_padding, dim2_padding, dim3_padding, dim4_padding)
    else:
      raise ValueError(
          '`padding` should be either an int, '
          'a tuple of 4 ints '
          '(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad, symmetric_dim4_pad), '
          'or a tuple of 4 tuples of 2 ints '
          '((left_dim1_pad, right_dim1_pad),'
          ' (left_dim2_pad, right_dim2_pad),'
          ' (left_dim3_pad, right_dim3_pad),'
          ' (left_dim4_pad, right_dim4_pad)). '
          'Found: ' + str(padding))
    self.input_spec = InputSpec(ndim=6)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        dim1 = input_shape[2] + self.padding[0][0] + self.padding[0][1]
      else:
        dim1 = None
      if input_shape[3] is not None:
        dim2 = input_shape[3] + self.padding[1][0] + self.padding[1][1]
      else:
        dim2 = None
      if input_shape[4] is not None:
        dim3 = input_shape[4] + self.padding[2][0] + self.padding[2][1]
      else:
        dim3 = None
      if input_shape[5] is not None:
        dim4 = input_shape[5] + self.padding[3][0] + self.padding[3][1]
      else:
        dim4 = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3, dim4])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        dim1 = input_shape[1] + self.padding[0][0] + self.padding[0][1]
      else:
        dim1 = None
      if input_shape[2] is not None:
        dim2 = input_shape[2] + self.padding[1][0] + self.padding[1][1]
      else:
        dim2 = None
      if input_shape[3] is not None:
        dim3 = input_shape[3] + self.padding[2][0] + self.padding[2][1]
      else:
        dim3 = None
      if input_shape[4] is not None:
        dim4 = input_shape[4] + self.padding[3][0] + self.padding[3][1]
      else:
        dim4 = None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, dim4, input_shape[5]])

  def call(self, inputs):
      if self.data_format == 'channels_first':
          axis = 2
      elif self.data_format == 'channels_last':
          axis = 1
      # xyz padding
      shape_in = inputs.shape
      x_list = tf.split(inputs, shape_in[axis], axis=axis)
      x = tf.concat(x_list, axis=0)
      x = tf.squeeze(x, axis=axis)
      x = backend.spatial_3d_padding(
        x, padding=self.padding[1:], data_format=self.data_format)
      x_list = tf.split(x, shape_in[axis], axis=0)
      x = tf.stack(x_list, axis=axis)
      # t padding
      axis += 1
      shape_in = x.shape
      x_list = tf.split(x, shape_in[axis], axis=axis)
      x = tf.concat(x_list, axis=0)
      x = tf.squeeze(x, axis=axis)
      x = backend.spatial_3d_padding(
          x, padding=(self.padding[0], (0, 0), (0, 0)), data_format=self.data_format)
      x_list = tf.split(x, shape_in[axis], axis=0)
      return tf.stack(x_list, axis=axis)

  def get_config(self):
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding4D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.Cropping1D')
class Cropping1D(Layer):
  """Cropping layer for 1D input (e.g. temporal sequence).
  It crops along the time dimension (axis 1).
  Examples:
  >>> input_shape = (2, 3, 2)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1]
    [ 2  3]
    [ 4  5]]
   [[ 6  7]
    [ 8  9]
    [10 11]]]
  >>> y = tf.keras.layers.Cropping1D(cropping=1)(x)
  >>> print(y)
  tf.Tensor(
    [[[2 3]]
     [[8 9]]], shape=(2, 1, 2), dtype=int64)
  Arguments:
    cropping: Int or tuple of int (length 2)
      How many units should be trimmed off at the beginning and end of
      the cropping dimension (axis 1).
      If a single int is provided, the same value will be used for both.
  Input shape:
    3D tensor with shape `(batch_size, axis_to_crop, features)`
  Output shape:
    3D tensor with shape `(batch_size, cropped_axis, features)`
  """

  def __init__(self, cropping=(1, 1), **kwargs):
    super(Cropping1D, self).__init__(**kwargs)
    self.cropping = conv_utils.normalize_tuple(cropping, 2, 'cropping')
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if input_shape[1] is not None:
      length = input_shape[1] - self.cropping[0] - self.cropping[1]
    else:
      length = None
    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

  def call(self, inputs):
    if self.cropping[1] == 0:
      return inputs[:, self.cropping[0]:, :]
    else:
      return inputs[:, self.cropping[0]:-self.cropping[1], :]

  def get_config(self):
    config = {'cropping': self.cropping}
    base_config = super(Cropping1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.Cropping2D')
class Cropping2D(Layer):
  """Cropping layer for 2D input (e.g. picture).
  It crops along spatial dimensions, i.e. height and width.
  Examples:
  >>> input_shape = (2, 28, 28, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)
  >>> print(y.shape)
  (2, 24, 20, 3)
  Arguments:
    cropping: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
      - If int: the same symmetric cropping
        is applied to height and width.
      - If tuple of 2 ints:
        interpreted as two different
        symmetric cropping values for height and width:
        `(symmetric_height_crop, symmetric_width_crop)`.
      - If tuple of 2 tuples of 2 ints:
        interpreted as
        `((top_crop, bottom_crop), (left_crop, right_crop))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, channels, rows, cols)`
  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, cropped_rows, cropped_cols, channels)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, channels, cropped_rows, cropped_cols)`
  """

  def __init__(self, cropping=((0, 0), (0, 0)), data_format=None, **kwargs):
    super(Cropping2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(cropping, int):
      self.cropping = ((cropping, cropping), (cropping, cropping))
    elif hasattr(cropping, '__len__'):
      if len(cropping) != 2:
        raise ValueError('`cropping` should have two elements. '
                         'Found: ' + str(cropping))
      height_cropping = conv_utils.normalize_tuple(cropping[0], 2,
                                                   '1st entry of cropping')
      width_cropping = conv_utils.normalize_tuple(cropping[1], 2,
                                                  '2nd entry of cropping')
      self.cropping = (height_cropping, width_cropping)
    else:
      raise ValueError('`cropping` should be either an int, '
                       'a tuple of 2 ints '
                       '(symmetric_height_crop, symmetric_width_crop), '
                       'or a tuple of 2 tuples of 2 ints '
                       '((top_crop, bottom_crop), (left_crop, right_crop)). '
                       'Found: ' + str(cropping))
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape([
          input_shape[0], input_shape[1],
          input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
          if input_shape[2] else None,
          input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
          if input_shape[3] else None
      ])
    else:
      return tensor_shape.TensorShape([
          input_shape[0],
          input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
          if input_shape[1] else None,
          input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
          if input_shape[2] else None, input_shape[3]
      ])
    # pylint: enable=invalid-unary-operand-type

  def call(self, inputs):
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:]
      elif self.cropping[0][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1]]
      elif self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:]
      return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                    self.cropping[1][0]:-self.cropping[1][1]]
    else:
      if self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:, :]
      elif self.cropping[0][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1], :]
      elif self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:, :]
      return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[
          1][0]:-self.cropping[1][1], :]  # pylint: disable=invalid-unary-operand-type
    # pylint: enable=invalid-unary-operand-type

  def get_config(self):
    config = {'cropping': self.cropping, 'data_format': self.data_format}
    base_config = super(Cropping2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.Cropping3D')
class Cropping3D(Layer):
  """Cropping layer for 3D data (e.g. spatial or spatio-temporal).
    Examples:
  >>> input_shape = (2, 28, 28, 10, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.Cropping3D(cropping=(2, 4, 2))(x)
  >>> print(y.shape)
  (2, 24, 20, 6, 3)
  Arguments:
    cropping: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
      - If int: the same symmetric cropping
        is applied to depth, height, and width.
      - If tuple of 3 ints: interpreted as two different
        symmetric cropping values for depth, height, and width:
        `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
      - If tuple of 3 tuples of 2 ints: interpreted as
        `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
          right_dim2_crop), (left_dim3_crop, right_dim3_crop))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop,
        depth)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, depth, first_axis_to_crop, second_axis_to_crop,
        third_axis_to_crop)`
  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, first_cropped_axis, second_cropped_axis, third_cropped_axis,
        depth)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, depth, first_cropped_axis, second_cropped_axis,
        third_cropped_axis)`
  """

  def __init__(self,
               cropping=((1, 1), (1, 1), (1, 1)),
               data_format=None,
               **kwargs):
    super(Cropping3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(cropping, int):
      self.cropping = ((cropping, cropping), (cropping, cropping), (cropping,
                                                                    cropping))
    elif hasattr(cropping, '__len__'):
      if len(cropping) != 3:
        raise ValueError('`cropping` should have 3 elements. '
                         'Found: ' + str(cropping))
      dim1_cropping = conv_utils.normalize_tuple(cropping[0], 2,
                                                 '1st entry of cropping')
      dim2_cropping = conv_utils.normalize_tuple(cropping[1], 2,
                                                 '2nd entry of cropping')
      dim3_cropping = conv_utils.normalize_tuple(cropping[2], 2,
                                                 '3rd entry of cropping')
      self.cropping = (dim1_cropping, dim2_cropping, dim3_cropping)
    else:
      raise ValueError(
          '`cropping` should be either an int, '
          'a tuple of 3 ints '
          '(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop), '
          'or a tuple of 3 tuples of 2 ints '
          '((left_dim1_crop, right_dim1_crop),'
          ' (left_dim2_crop, right_dim2_crop),'
          ' (left_dim3_crop, right_dim2_crop)). '
          'Found: ' + str(cropping))
    self.input_spec = InputSpec(ndim=5)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        dim1 = input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
      else:
        dim1 = None
      if input_shape[3] is not None:
        dim2 = input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
      else:
        dim2 = None
      if input_shape[4] is not None:
        dim3 = input_shape[4] - self.cropping[2][0] - self.cropping[2][1]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        dim1 = input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
      else:
        dim1 = None
      if input_shape[2] is not None:
        dim2 = input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
      else:
        dim2 = None
      if input_shape[3] is not None:
        dim3 = input_shape[3] - self.cropping[2][0] - self.cropping[2][1]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])
    # pylint: enable=invalid-unary-operand-type

  def call(self, inputs):
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:]
      elif self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:-self.cropping[2][1]]
      elif self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:, self.cropping[2][0]:]
      elif self.cropping[0][1] == self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1], self.cropping[2][0]:]
      elif self.cropping[0][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][
            0]:-self.cropping[1][1], self.cropping[2][0]:-self.cropping[2][1]]
      elif self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
                      cropping[1][0]:, self.cropping[2][0]:-self.cropping[2][1]]
      elif self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
                      cropping[1][0]:-self.cropping[1][1], self.cropping[2][0]:]
      return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                    self.cropping[1][0]:-self.cropping[1][1], self.cropping[2][
                        0]:-self.cropping[2][1]]
    else:
      if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:, :]
      elif self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:-self.cropping[2][1], :]
      elif self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:, self.cropping[2][0]:, :]
      elif self.cropping[0][1] == self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1], self.cropping[2][0]:, :]
      elif self.cropping[0][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][
            0]:-self.cropping[1][1], self.cropping[2][0]:
                      -self.cropping[2][1], :]
      elif self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][
            0]:-self.cropping[0][1], self.cropping[1][0]:, self.cropping[2][0]:
                      -self.cropping[2][1], :]
      elif self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:-self.cropping[1][1], self.cropping[
                          2][0]:, :]
      return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[
          1][0]:-self.cropping[1][1], self.cropping[2][0]:  # pylint: disable=invalid-unary-operand-type
                    -self.cropping[2][1], :]  # pylint: disable=invalid-unary-operand-type
    # pylint: enable=invalid-unary-operand-type

  def get_config(self):
    config = {'cropping': self.cropping, 'data_format': self.data_format}
    base_config = super(Cropping3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


#@keras_export('keras.layers.Cropping4D')
class Cropping4D(Layer):
  """Cropping layer for 4D data (e.g. spatial or spatio-temporal).
    Examples:
  >>> input_shape = (2, 28, 28, 10, 10, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.Cropping4D(cropping=(2, 4, 2, 2))(x)
  >>> print(y.shape)
  (2, 24, 20, 6, 3)
  Arguments:
    cropping: Int, or tuple of 4 ints, or tuple of 4 tuples of 2 ints.
      - If int: the same symmetric cropping
        is applied to depth, height, and width.
      - If tuple of 4 ints: interpreted as two different
        symmetric cropping values for depth, height, and width:
        `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop, symmetric_dim4_crop)`.
      - If tuple of 4 tuples of 2 ints: interpreted as
        `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
          right_dim2_crop), (left_dim3_crop, right_dim3_crop), )`
          (left_dim4_crop, right_dim4_crop)
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
  Input shape:
    6D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop, fourth_axis_to_crop,
        depth)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, depth, first_axis_to_crop, second_axis_to_crop,
        third_axis_to_crop, fourth_axis_to_crop)`
  Output shape:
    6D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, first_cropped_axis, second_cropped_axis, third_cropped_axis, fourth_cropped_axis,
        depth)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, depth, first_cropped_axis, second_cropped_axis,
        third_cropped_axis, fourth_cropped_axis)`
  """

  def __init__(self,
               cropping=((1, 1), (1, 1), (1, 1), (1, 1)),
               data_format=None,
               **kwargs):
    super(Cropping4D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(cropping, int):
      self.cropping = ((cropping, cropping), (cropping, cropping),
                       (cropping, cropping), (cropping, cropping))
    elif hasattr(cropping, '__len__'):
      if len(cropping) != 4:
        raise ValueError('`cropping` should have 4 elements. '
                         'Found: ' + str(cropping))
      dim1_cropping = conv_utils.normalize_tuple(cropping[0], 2,
                                                 '1st entry of cropping')
      dim2_cropping = conv_utils.normalize_tuple(cropping[1], 2,
                                                 '2nd entry of cropping')
      dim3_cropping = conv_utils.normalize_tuple(cropping[2], 2,
                                                 '3rd entry of cropping')
      dim4_cropping = conv_utils.normalize_tuple(cropping[3], 2,
                                                 '4th entry of cropping')
      self.cropping = (dim1_cropping, dim2_cropping, dim3_cropping, dim4_cropping)
    else:
      raise ValueError(
          '`cropping` should be either an int, '
          'a tuple of 4 ints '
          '(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop, symmetric_dim4_crop), '
          'or a tuple of 4 tuples of 2 ints '
          '((left_dim1_crop, right_dim1_crop),'
          ' (left_dim2_crop, right_dim2_crop),'
          ' (left_dim3_crop, right_dim3_crop),'
          ' (left_dim4_crop, right_dim4_crop)). '
          'Found: ' + str(cropping))
    self.input_spec = InputSpec(ndim=6)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        dim1 = input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
      else:
        dim1 = None
      if input_shape[3] is not None:
        dim2 = input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
      else:
        dim2 = None
      if input_shape[4] is not None:
        dim3 = input_shape[4] - self.cropping[2][0] - self.cropping[2][1]
      else:
        dim3 = None
      if input_shape[5] is not None:
        dim4 = input_shape[5] - self.cropping[3][0] - self.cropping[3][1]
      else:
        dim4 = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3, dim4])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        dim1 = input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
      else:
        dim1 = None
      if input_shape[2] is not None:
        dim2 = input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
      else:
        dim2 = None
      if input_shape[3] is not None:
        dim3 = input_shape[3] - self.cropping[2][0] - self.cropping[2][1]
      else:
        dim3 = None
      if input_shape[4] is not None:
        dim4 = input_shape[4] - self.cropping[3][0] - self.cropping[3][1]
      else:
        dim4 = None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, dim4, input_shape[5]])
    # pylint: enable=invalid-unary-operand-type

  def call(self, inputs):
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if self.cropping[3][1] == 0:
          if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
            return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
                          self.cropping[2][0]:, self.cropping[3][0]:]
          elif self.cropping[0][1] == self.cropping[1][1] == 0:
            return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
                          self.cropping[2][0]:-self.cropping[2][1], self.cropping[3][0]:]
          elif self.cropping[1][1] == self.cropping[2][1] == 0:
            return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                          self.cropping[1][0]:, self.cropping[2][0]:, self.cropping[3][0]:]
          elif self.cropping[0][1] == self.cropping[2][1] == 0:
            return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
                          -self.cropping[1][1], self.cropping[2][0]:, self.cropping[3][0]:]
          elif self.cropping[0][1] == 0:
            return inputs[:, :, self.cropping[0][0]:, self.cropping[1][
                0]:-self.cropping[1][1], self.cropping[2][0]:-self.cropping[2][1], self.cropping[3][0]:]
          elif self.cropping[1][1] == 0:
            return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
                          cropping[1][0]:, self.cropping[2][0]:-self.cropping[2][1], self.cropping[3][0]:]
          elif self.cropping[2][1] == 0:
            return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
                          cropping[1][0]:-self.cropping[1][1], self.cropping[2][0]:, self.cropping[3][0]:]
      else:
          if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
            return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
                          self.cropping[2][0]:, self.cropping[3][0]:-self.cropping[3][1]]
          elif self.cropping[0][1] == self.cropping[1][1] == 0:
            return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
                          self.cropping[2][0]:-self.cropping[2][1], self.cropping[3][0]:-self.cropping[3][1]]
          elif self.cropping[1][1] == self.cropping[2][1] == 0:
            return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                          self.cropping[1][0]:, self.cropping[2][0]:, self.cropping[3][0]:-self.cropping[3][1]]
          elif self.cropping[0][1] == self.cropping[2][1] == 0:
            return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
                          -self.cropping[1][1], self.cropping[2][0]:, self.cropping[3][0]:-self.cropping[3][1]]
          elif self.cropping[0][1] == 0:
            return inputs[:, :, self.cropping[0][0]:, self.cropping[1][
                0]:-self.cropping[1][1], self.cropping[2][0]:-self.cropping[2][1], self.cropping[3][0]:-self.cropping[3][1]]
          elif self.cropping[1][1] == 0:
            return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
                          cropping[1][0]:, self.cropping[2][0]:-self.cropping[2][1], self.cropping[3][0]:-self.cropping[3][1]]
          elif self.cropping[2][1] == 0:
            return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
                          cropping[1][0]:-self.cropping[1][1], self.cropping[2][0]:, self.cropping[3][0]:-self.cropping[3][1]]
      return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                    self.cropping[1][0]:-self.cropping[1][1], self.cropping[2][
                        0]:-self.cropping[2][1], self.cropping[3][0]:-self.cropping[3][1]]
    else:
      if self.cropping[3][1] == 0:
          if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
            return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
                          self.cropping[2][0]:, self.cropping[3][0]:, :]
          elif self.cropping[0][1] == self.cropping[1][1] == 0:
            return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
                          self.cropping[2][0]:-self.cropping[2][1], self.cropping[3][0]:, :]
          elif self.cropping[1][1] == self.cropping[2][1] == 0:
            return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                          self.cropping[1][0]:, self.cropping[2][0]:, self.cropping[3][0]:, :]
          elif self.cropping[0][1] == self.cropping[2][1] == 0:
            return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:
                          -self.cropping[1][1], self.cropping[2][0]:, self.cropping[3][0]:, :]
          elif self.cropping[0][1] == 0:
            return inputs[:, self.cropping[0][0]:, self.cropping[1][
                0]:-self.cropping[1][1], self.cropping[2][0]:
                          -self.cropping[2][1], self.cropping[3][0]:, :]
          elif self.cropping[1][1] == 0:
            return inputs[:, self.cropping[0][
                0]:-self.cropping[0][1], self.cropping[1][0]:, self.cropping[2][0]:
                          -self.cropping[2][1], self.cropping[3][0]:, :]
          elif self.cropping[2][1] == 0:
            return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                          self.cropping[1][0]:-self.cropping[1][1], self.cropping[
                              2][0]:, self.cropping[3][0]:, :]
      else:
          if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
              return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
                     self.cropping[2][0]:, self.cropping[3][0]:-self.cropping[3][1], :]
          elif self.cropping[0][1] == self.cropping[1][1] == 0:
              return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
                     self.cropping[2][0]:-self.cropping[2][1], self.cropping[3][0]:-self.cropping[3][1], :]
          elif self.cropping[1][1] == self.cropping[2][1] == 0:
              return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                     self.cropping[1][0]:, self.cropping[2][0]:, self.cropping[3][0]:-self.cropping[3][1], :]
          elif self.cropping[0][1] == self.cropping[2][1] == 0:
              return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:-self.cropping[1][1],
                     self.cropping[2][0]:, self.cropping[3][0]:-self.cropping[3][1], :]
          elif self.cropping[0][1] == 0:
              return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:-self.cropping[1][1],
                     self.cropping[2][0]:-self.cropping[2][1], self.cropping[3][0]:-self.cropping[3][1], :]
          elif self.cropping[1][1] == 0:
              return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[1][0]:,
                     self.cropping[2][0]:-self.cropping[2][1], self.cropping[3][0]:-self.cropping[3][1], :]
          elif self.cropping[2][1] == 0:
              return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[1][0]:-self.cropping[1][1],
                     self.cropping[2][0]:, self.cropping[3][0]:-self.cropping[3][1], :]
      return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[
          1][0]:-self.cropping[1][1], self.cropping[2][0]:  # pylint: disable=invalid-unary-operand-type
                    -self.cropping[2][1], self.cropping[3][0]:-self.cropping[3][1], :]  # pylint: disable=invalid-unary-operand-type
    # pylint: enable=invalid-unary-operand-type

  def get_config(self):
    config = {'cropping': self.cropping, 'data_format': self.data_format}
    base_config = super(Cropping4D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# Aliases
ComplexConvolution1D = ComplexConv1D
ComplexConvolution2D = ComplexConv2D
ComplexConvolution3D = ComplexConv3D
ComplexConvolution2DTranspose = ComplexConv2DTranspose
ComplexConvolution3DTranspose = ComplexConv3DTranspose
ComplexDeconvolution2D = ComplexDeconv2D = ComplexConv2DTranspose
ComplexDeconvolution3D = ComplexDeconv3D = ComplexConv3DTranspose
Cropping2Dt = Cropping3D
ZeroPadding2Dt = ZeroPadding3D
UpSampling2Dt = UpSampling3D
Cropping3Dt = Cropping4D
ZeroPadding3Dt = ZeroPadding4D
UpSampling3Dt = UpSampling4D
