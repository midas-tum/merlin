import tensorflow as tf

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils

import merlintf

try:
    import optotf.pad
except:
    print('optotf could not be imported')

def Padding(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'Padding' + (str(identifier).upper() if len(identifier) == 2 else str(identifier[0:2]).upper() + str(identifier[-1]))
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret padding function identifier: {}'.format(identifier))

def PaddingTranspose(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'Padding' + (str(identifier).upper() if len(identifier) == 2 else str(identifier[0:2]).upper() + str(identifier[-1]))
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret padding function identifier: {}'.format(identifier))

def deserialize(op):
    if op == 'Padding1D':
        return Padding1D
    elif op == 'Padding2D':
        return Padding2D
    elif op == 'Padding3D' or op == 'Padding2Dt':
        return Padding3D
    elif op == 'Padding4D' or op == 'Padding3Dt':
        return Padding4D
    elif op == 'Padding1DTranspose':
        return Padding1DTranspose
    elif op == 'Padding2DTranspose':
        return Padding2DTranspose
    elif op == 'Padding3DTranspose' or op == 'Padding2DtTranspose':
        return Padding3DTranspose
    elif op == 'Padding4DTranspose' or op == 'Padding3DtTranspose':
        return Padding4DTranspose
    else:
        raise ValueError('Unknown padding operation: {}'.format(op))

def flatten(object):
    for item in object:
        if isinstance(item, (list, tuple, set)):
            yield from flatten(item)
        else:
            yield item

class Padding1D(Layer):
    """Padding layer for 1D input (e.g. picture).
    This layer can add rows and columns of symmetric, reflected or replicated values,
    at the top, bottom, left and right side of an image tensor.
    Arguments:
    padding: Int
      - If int: the same symmetric padding
        is applied to height and width.
    mode: Padding mode (string)
        - symmetric: the symmetric value of border pixels x-1, x-2, ... are appended
        - reflect: the reflected value of border pixels x, x-1, x-2, ... are appended
        - replicate: the same border pixels x, x, ... are appended
        - constant: a constant value is appended
    constant_values: float. Constant value to be appended if (mode='constant').
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    Input shape:
    3D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows)`
    Output shape:
    3D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, padded_rows, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, padded_rows)`
    """

    def __init__(self, padding=1, mode='symmetric', constant_values=0, data_format=None, **kwargs):
        super(Padding1D, self).__init__(**kwargs)
        self.mode = mode
        self.optox = (True if 'optotf.pad' in sys.modules else False)
        if not self.optox and mode != 'constant':
            raise ValueError('Only constant padding is supported without optotf')
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding))
        elif hasattr(padding, '__len__'):
            self.padding = conv_utils.normalize_tuple(padding[0], 2, '1st entry of padding')
        else:
            raise ValueError('`padding` should be an int. Found: ' + str(padding))
        if isinstance(constant_values, float):
            self.constant_values = ((constant_values, constant_values))
        elif hasattr(constant_values, '__len__'):
            self.constant_values = conv_utils.normalize_tuple(constant_values[0], 2, '1st entry of constant_values')
        else:
            raise ValueError('`constant_values` should be a float, Found: ' + str(constant_values))
        self.optotf_padding = list(flatten(self.padding))[::-1]
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], rows])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            return tensor_shape.TensorShape([input_shape[0], rows, input_shape[3]])

    def call(self, inputs):
        if self.mode == 'constant':
            if merlintf.iscomplex(inputs):
                return tf.complex(tf.pad(tf.real(inputs), self.padding, mode=self.mode, constant_values=self.constant_values),
                                  tf.pad(tf.imag(inputs), self.padding, mode=self.mode, constant_values=self.constant_values))
            else:
                return tf.pad(inputs, self.padding, 'constant', constant_values=self.constant_values)
        else:
            return optotf.pad.pad1d(inputs, self.optotf_padding, mode=self.mode)  # gradient registered in optotf

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(Padding1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Padding2D(Layer):
    """Padding layer for 2D input (e.g. picture).
    This layer can add rows and columns of symmetric, reflected or replicated values,
    at the top, bottom, left and right side of an image tensor.
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
    mode: Padding mode (string)
        - symmetric: the symmetric value of border pixels x-1, x-2, ... are appended
        - reflect: the reflected value of border pixels x, x-1, x-2, ... are appended
        - replicate: the same border pixels x, x, ... are appended
        - constant: a constant value is appended
    constant_values: float, or tuple of 2 floats, or tuple of 2 tuples of 2 floats. Constant value to be appended if
        (mode='constant').
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

    def __init__(self, padding=(1, 1), mode='symmetric', constant_values=0, data_format=None, **kwargs):
        super(Padding2D, self).__init__(**kwargs)
        self.mode = mode
        self.optox = (True if 'optotf.pad' in sys.modules else False)
        if not self.optox and mode != 'constant':
            raise ValueError('Only constant padding is supported without optotf')
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2, '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2, '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        if isinstance(constant_values, float):
            self.constant_values = ((constant_values, constant_values), (constant_values, constant_values))
        elif hasattr(constant_values, '__len__'):
            if len(constant_values) != 2:
                raise ValueError('`constant_values` should have two elements. Found: ' + str(constant_values))
            height_constant_values = conv_utils.normalize_tuple(constant_values[0], 2, '1st entry of constant_values')
            width_constant_values = conv_utils.normalize_tuple(constant_values[1], 2, '2nd entry of constant_values')
            self.constant_values = (height_constant_values, width_constant_values)
        else:
            raise ValueError('`constant_values` should be either a float, '
                             'a tuple of 2 float '
                             '(height_constant, width_constant), '
                             'or a tuple of 2 tuples of 2 float '
                             '((top_constant, bottom_constant), (left_constant, right_constant)). '
                             'Found: ' + str(constant_values))
        self.optotf_padding = list(flatten(self.padding))[::-1]
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
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], rows, cols])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return tensor_shape.TensorShape([input_shape[0], rows, cols, input_shape[3]])

    def call(self, inputs):
        if self.mode == 'constant':
            if merlintf.iscomplex(inputs):
                return tf.complex(tf.pad(tf.real(inputs), self.padding, mode=self.mode, constant_values=self.constant_values),
                                  tf.pad(tf.imag(inputs), self.padding, mode=self.mode, constant_values=self.constant_values))
            else:
                return tf.pad(inputs, self.padding, 'constant', constant_values=self.constant_values)
        else:
            return optotf.pad.pad2d(inputs, self.optotf_padding, mode=self.mode)  # gradient registered in optotf

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(Padding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Padding3D(Layer):
    """Padding layer for 3D input (e.g. picture).
    This layer can add rows and columns of symmetric, reflected or replicated values,
    at the top, bottom, left, right and fron, back side of an image tensor.
    Arguments:
    padding: Int, or tuple of 3 ints, or tuple of 3 tuples of 3 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 3 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_height_pad, symmetric_width_pad)`.
      - If tuple of 3 tuples of 3 ints:
        interpreted as
        `((top_pad, bottom_pad), (left_pad, right_pad), (front_pad, back_pad))`
    mode: Padding mode (string)
        - symmetric: the symmetric value of border pixels x-1, x-2, ... are appended
        - reflect: the reflected value of border pixels x, x-1, x-2, ... are appended
        - replicate: the same border pixels x, x, ... are appended
        - constant: a constant value is appended
    constant_values: float, or tuple of 3 floats, or tuple of 3 tuples of 3 floats. Constant value to be appended if
        (mode='constant').
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, depths, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width, depths)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, depths, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols, depths)`
    Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, padded_rows, padded_cols, padded_depths, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, padded_rows, padded_cols, padded_depths)`
    """

    def __init__(self, padding=(1, 1, 1), mode='symmetric', constant_values=0, data_format=None, **kwargs):
        super(Padding3D, self).__init__(**kwargs)
        self.mode = mode
        self.optox = (True if 'optotf.pad' in sys.modules else False)
        if not self.optox and mode != 'constant':
            raise ValueError('Only constant padding is supported without optotf')
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 3:
                raise ValueError('`padding` should have three elements. Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2, '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2, '2nd entry of padding')
            depth_padding = conv_utils.normalize_tuple(padding[2], 2, '3rd entry of padding')
            self.padding = (height_padding, width_padding, depth_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 3 ints '
                             '(symmetric_height_pad, symmetric_width_pad, symmetric_depth_pad), '
                             'or a tuple of 3 tuples of 3 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad), (front_pad, back_pad)). '
                             'Found: ' + str(padding))
        if isinstance(constant_values, float):
            self.constant_values = ((constant_values, constant_values), (constant_values, constant_values),
                                    (constant_values, constant_values))
        elif hasattr(constant_values, '__len__'):
            if len(constant_values) != 3:
                raise ValueError('`constant_values` should have three elements. Found: ' + str(constant_values))
            height_constant_values = conv_utils.normalize_tuple(constant_values[0], 2, '1st entry of constant_values')
            width_constant_values = conv_utils.normalize_tuple(constant_values[1], 2, '2nd entry of constant_values')
            depth_constant_values = conv_utils.normalize_tuple(constant_values[2], 2, '3rd entry of constant_values')
            self.constant_values = (height_constant_values, width_constant_values, depth_constant_values)
        else:
            raise ValueError('`constant_values` should be either a float, '
                             'a tuple of 3 floats '
                             '(height_constant, width_constant, depth_constant), '
                             'or a tuple of 3 tuples of 3 floats '
                             '((top_constant, bottom_constant), (left_constant, right_constant), (front_constant, back_constant)). '
                             'Found: ' + str(constant_values))
        self.optotf_padding = list(flatten(self.padding))[::-1]
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
            if input_shape[4] is not None:
                depth = input_shape[4] + self.padding[2][0] + self.padding[2][1]
            else:
                depth = None
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], rows, cols, depth])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            if input_shape[3] is not None:
                depth = input_shape[3] + self.padding[2][0] + self.padding[2][1]
            else:
                depth = None
            return tensor_shape.TensorShape([input_shape[0], rows, cols, depth, input_shape[4]])

    def call(self, inputs):
        if self.mode == 'constant':
            if merlintf.iscomplex(inputs):
                return tf.complex(tf.pad(tf.real(inputs), self.padding, mode=self.mode, constant_values=self.constant_values),
                                  tf.pad(tf.imag(inputs), self.padding, mode=self.mode, constant_values=self.constant_values))
            else:
                return tf.pad(inputs, self.padding, 'constant', constant_values=self.constant_values)
        else:
            return optotf.pad.pad3d(inputs, self.optotf_padding, mode=self.mode)  # gradient registered in optotf

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(Padding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Padding4D(Layer):
    """Padding layer for 4D input (e.g. picture).
    This layer can add rows and columns of symmetric, reflected or replicated values,
    at the timestart, timeend, top, bottom, left, right, front, back side of an image tensor.
    Arguments:
    padding: Int, or tuple of 4 ints, or tuple of 4 tuples of 4 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 4 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_height_pad, symmetric_width_pad)`.
      - If tuple of 4 tuples of 4 ints:
        interpreted as
        `((timestart_pad, timeend_pad), (top_pad, bottom_pad), (left_pad, right_pad), (front_pad, back_pad))`
    mode: Padding mode (string)
        - symmetric: the symmetric value of border pixels x-1, x-2, ... are appended
        - reflect: the reflected value of border pixels x, x-1, x-2, ... are appended
        - replicate: the same border pixels x, x, ... are appended
        - constant: a constant value is appended
    constant_values: float, or tuple of 4 floats, or tuple of 4 tuples of 4 floats. Constant value to be appended if
        (mode='constant').
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, time, height, width, depths, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, time, height, width, depths)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    Input shape:
    6D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, time, rows, cols, depths, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, time, rows, cols, depths)`
    Output shape:
    6D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, padded_times, padded_rows, padded_cols, padded_depths, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, padded_times, padded_rows, padded_cols, padded_depths)`
    """

    def __init__(self, padding=(1, 1, 1, 1), mode='symmetric', constant_values=0, data_format=None, **kwargs):
        super(Padding4D, self).__init__(**kwargs)
        self.mode = mode
        self.optox = (True if 'optotf.pad' in sys.modules else False)
        if not self.optox and mode != 'constant':
            raise ValueError('Only constant padding is supported without optotf')
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding), (padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 4:
                raise ValueError('`padding` should have four elements. Found: ' + str(padding))
            time_padding = conv_utils.normalize_tuple(padding[0], 2, '1st entry of padding')
            height_padding = conv_utils.normalize_tuple(padding[1], 2, '2nd entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[2], 2, '3rd entry of padding')
            depth_padding = conv_utils.normalize_tuple(padding[3], 2, '4th entry of padding')
            self.padding = (time_padding, height_padding, width_padding, depth_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 4 ints '
                             '(symmetric_time_pad, symmetric_height_pad, symmetric_width_pad, symmetric_depth_pad), '
                             'or a tuple of 4 tuples of 4 ints '
                             '((timestart_pad, timeend_pad), (top_pad, bottom_pad), (left_pad, right_pad), (front_pad, back_pad)). '
                             'Found: ' + str(padding))
        if isinstance(constant_values, float):
            self.constant_values = ((constant_values, constant_values), (constant_values, constant_values),
                                    (constant_values, constant_values), (constant_values, constant_values))
        elif hasattr(constant_values, '__len__'):
            if len(constant_values) != 4:
                raise ValueError('`constant_values` should have four elements. Found: ' + str(constant_values))
            time_constant_values = conv_utils.normalize_tuple(constant_values[0], 2, '1st entry of constant_values')
            height_constant_values = conv_utils.normalize_tuple(constant_values[1], 2, '2nd entry of constant_values')
            width_constant_values = conv_utils.normalize_tuple(constant_values[2], 2, '3rd entry of constant_values')
            depth_constant_values = conv_utils.normalize_tuple(constant_values[3], 2, '4th entry of constant_values')
            self.constant_values = (time_constant_values, height_constant_values, width_constant_values, depth_constant_values)
        else:
            raise ValueError('`constant_values` should be either a float, '
                             'a tuple of 4 floats '
                             '(time_constant, height_constant, width_constant, depth_constant), '
                             'or a tuple of 4 tuples of 4 floats '
                             '((starttime_constant, endtime_constant), (top_constant, bottom_constant), (left_constant, right_constant), (front_constant, back_constant)). '
                             'Found: ' + str(constant_values))
        self.optotf_padding = list(flatten(self.padding))[::-1]
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                times = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                times = None
            if input_shape[3] is not None:
                rows = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                rows = None
            if input_shape[4] is not None:
                cols = input_shape[4] + self.padding[2][0] + self.padding[2][1]
            else:
                cols = None
            if input_shape[5] is not None:
                depth = input_shape[5] + self.padding[3][0] + self.padding[3][1]
            else:
                depth = None
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], times, rows, cols, depth])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                times = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                times = None
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[2][0] + self.padding[2][1]
            else:
                cols = None
            if input_shape[4] is not None:
                depth = input_shape[4] + self.padding[3][0] + self.padding[3][1]
            else:
                depth = None
            return tensor_shape.TensorShape([input_shape[0], times, rows, cols, depth, input_shape[5]])

    def call(self, inputs):
        if self.mode == 'constant':
            if merlintf.iscomplex(inputs):
                return tf.complex(tf.pad(tf.real(inputs), self.padding, mode=self.mode, constant_values=self.constant_values),
                                  tf.pad(tf.imag(inputs), self.padding, mode=self.mode, constant_values=self.constant_values))
            else:
                return tf.pad(inputs, self.padding, 'constant', constant_values=self.constant_values)
        else:
            if self.data_format == 'channels_first':
                axis = 2
            elif self.data_format == 'channels_last':
                axis = 1
            # xyz padding
            shape_in = inputs.shape
            x_list = tf.split(inputs, shape_in[axis], axis=axis)
            x = tf.concat(x_list, axis=0)
            x = tf.squeeze(x, axis=axis)
            x = optotf.pad.pad3d(x, self.optotf_padding[0:3], mode=self.mode)
            x_list = tf.split(x, shape_in[axis], axis=0)
            x = tf.stack(x_list, axis=axis)
            # t padding
            axis += 1
            shape_in = x.shape
            x_list = tf.split(x, shape_in[axis], axis=axis)
            x = tf.concat(x_list, axis=0)
            x = tf.squeeze(x, axis=axis)
            x = optotf.pad.pad3d(x, [self.optotf_padding[-1], 0, 0, 0, 0], mode=self.mode)
            x_list = tf.split(x, shape_in[axis], axis=0)
            return tf.stack(x_list, axis=axis)

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(Padding4D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Padding1DTranspose(Layer):
    """Transpose Padding layer for 1D input (e.g. picture).
    This layer can add rows and columns of symmetric, reflected or replicated values,
    at the top, bottom, left and right side of an image tensor.
    Arguments:
    padding: Int
      - If int: the same symmetric padding
        is applied to height and width.
    mode: Padding mode (string)
        - symmetric: the symmetric value of border pixels x-1, x-2, ... are appended
        - reflect: the reflected value of border pixels x, x-1, x-2, ... are appended
        - replicate: the same border pixels x, x, ... are appended
        - constant: a constant value is appended
    constant_values: float. Constant value to be appended if (mode='constant').
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    Input shape:
    3D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows)`
    Output shape:
    3D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, padded_rows, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, padded_rows)`
    """

    def __init__(self, padding=1, mode='symmetric', constant_values=0, data_format=None, **kwargs):
        super(Padding1DTranspose, self).__init__(**kwargs)
        self.mode = mode
        self.optox = (True if 'optotf.pad' in sys.modules else False)
        if not self.optox and mode != 'constant':
            raise ValueError('Only constant padding is supported without optotf')
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding))
        elif hasattr(padding, '__len__'):
            self.padding = conv_utils.normalize_tuple(padding[0], 2, '1st entry of padding')
        else:
            raise ValueError('`padding` should be an int. Found: ' + str(padding))
        if isinstance(constant_values, float):
            self.constant_values = ((constant_values, constant_values))
        elif hasattr(constant_values, '__len__'):
            self.constant_values = conv_utils.normalize_tuple(constant_values[0], 2, '1st entry of constant_values')
        else:
            raise ValueError('`constant_values` should be a float, Found: ' + str(constant_values))
        self.optotf_padding = list(flatten(self.padding))[::-1]
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], rows])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            return tensor_shape.TensorShape([input_shape[0], rows, input_shape[3]])

    def call(self, inputs):
        if self.mode == 'constant':
            if merlintf.iscomplex(inputs):
                return tf.complex(tf.pad(tf.real(inputs), self.padding, mode=self.mode, constant_values=self.constant_values),
                                  tf.pad(tf.imag(inputs), self.padding, mode=self.mode, constant_values=self.constant_values))
            else:
                return tf.pad(inputs, self.padding, 'constant', constant_values=self.constant_values)
        else:
            return optotf.pad.pad1d_transpose(inputs, self.optotf_padding, mode=self.mode)  # gradient registered in optotf

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(Padding1DTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Padding2DTranspose(Layer):
    """Transpose Padding layer for 2D input (e.g. picture).
    This layer can add rows and columns of symmetric, reflected or replicated values,
    at the top, bottom, left and right side of an image tensor.
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
    mode: Padding mode (string)
        - symmetric: the symmetric value of border pixels x-1, x-2, ... are appended
        - reflect: the reflected value of border pixels x, x-1, x-2, ... are appended
        - replicate: the same border pixels x, x, ... are appended
        - constant: a constant value is appended
    constant_values: float, or tuple of 2 floats, or tuple of 2 tuples of 2 floats. Constant value to be appended if
        (mode='constant').
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

    def __init__(self, padding=(1, 1), mode='symmetric', constant_values=0, data_format=None, **kwargs):
        super(Padding2DTranspose, self).__init__(**kwargs)
        self.mode = mode
        self.optox = (True if 'optotf.pad' in sys.modules else False)
        if not self.optox and mode != 'constant':
            raise ValueError('Only constant padding is supported without optotf')
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2, '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2, '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        if isinstance(constant_values, float):
            self.constant_values = ((constant_values, constant_values), (constant_values, constant_values))
        elif hasattr(constant_values, '__len__'):
            if len(constant_values) != 2:
                raise ValueError('`constant_values` should have two elements. Found: ' + str(constant_values))
            height_constant_values = conv_utils.normalize_tuple(constant_values[0], 2, '1st entry of constant_values')
            width_constant_values = conv_utils.normalize_tuple(constant_values[1], 2, '2nd entry of constant_values')
            self.constant_values = (height_constant_values, width_constant_values)
        else:
            raise ValueError('`constant_values` should be either a float, '
                             'a tuple of 2 float '
                             '(height_constant, width_constant), '
                             'or a tuple of 2 tuples of 2 float '
                             '((top_constant, bottom_constant), (left_constant, right_constant)). '
                             'Found: ' + str(constant_values))
        self.optotf_padding = list(flatten(self.padding))[::-1]
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
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], rows, cols])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return tensor_shape.TensorShape([input_shape[0], rows, cols, input_shape[3]])

    def call(self, inputs):
        if self.mode == 'constant':
            if merlintf.iscomplex(inputs):
                return tf.complex(tf.pad(tf.real(inputs), self.padding, mode=self.mode, constant_values=self.constant_values),
                                  tf.pad(tf.imag(inputs), self.padding, mode=self.mode, constant_values=self.constant_values))
            else:
                return tf.pad(inputs, self.padding, 'constant', constant_values=self.constant_values)
        else:
            return optotf.pad.pad2d_transpose(inputs, self.optotf_padding, mode=self.mode)  # gradient registered in optotf

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(Padding2DTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Padding3DTranspose(Layer):
    """Transpose Padding layer for 3D input (e.g. picture).
    This layer can add rows and columns of symmetric, reflected or replicated values,
    at the top, bottom, left, right and fron, back side of an image tensor.
    Arguments:
    padding: Int, or tuple of 3 ints, or tuple of 3 tuples of 3 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 3 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_height_pad, symmetric_width_pad)`.
      - If tuple of 3 tuples of 3 ints:
        interpreted as
        `((top_pad, bottom_pad), (left_pad, right_pad), (front_pad, back_pad))`
    mode: Padding mode (string)
        - symmetric: the symmetric value of border pixels x-1, x-2, ... are appended
        - reflect: the reflected value of border pixels x, x-1, x-2, ... are appended
        - replicate: the same border pixels x, x, ... are appended
        - constant: a constant value is appended
    constant_values: float, or tuple of 3 floats, or tuple of 3 tuples of 3 floats. Constant value to be appended if
        (mode='constant').
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, depths, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width, depths)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, depths, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols, depths)`
    Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, padded_rows, padded_cols, padded_depths, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, padded_rows, padded_cols, padded_depths)`
    """

    def __init__(self, padding=(1, 1, 1), mode='symmetric', constant_values=0, data_format=None, **kwargs):
        super(Padding3DTranspose, self).__init__(**kwargs)
        self.mode = mode
        self.optox = (True if 'optotf.pad' in sys.modules else False)
        if not self.optox and mode != 'constant':
            raise ValueError('Only constant padding is supported without optotf')
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 3:
                raise ValueError('`padding` should have three elements. Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2, '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2, '2nd entry of padding')
            depth_padding = conv_utils.normalize_tuple(padding[2], 2, '3rd entry of padding')
            self.padding = (height_padding, width_padding, depth_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 3 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 3 tuples of 3 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad), (front_pad, back_pad)). '
                             'Found: ' + str(padding))
        if isinstance(constant_values, float):
            self.constant_values = ((constant_values, constant_values), (constant_values, constant_values),
                                    (constant_values, constant_values))
        elif hasattr(constant_values, '__len__'):
            if len(constant_values) != 3:
                raise ValueError('`constant_values` should have three elements. Found: ' + str(constant_values))
            height_constant_values = conv_utils.normalize_tuple(constant_values[0], 2, '1st entry of constant_values')
            width_constant_values = conv_utils.normalize_tuple(constant_values[1], 2, '2nd entry of constant_values')
            depth_constant_values = conv_utils.normalize_tuple(constant_values[2], 2, '3rd entry of constant_values')
            self.constant_values = (height_constant_values, width_constant_values, depth_constant_values)
        else:
            raise ValueError('`constant_values` should be either a float, '
                             'a tuple of 3 floats '
                             '(height_constant, width_constant, depth_constant), '
                             'or a tuple of 3 tuples of 3 floats '
                             '((top_constant, bottom_constant), (left_constant, right_constant), (front_constant, back_constant)). '
                             'Found: ' + str(constant_values))
        self.optotf_padding = list(flatten(self.padding))[::-1]
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
            if input_shape[4] is not None:
                depth = input_shape[4] + self.padding[2][0] + self.padding[2][1]
            else:
                depth = None
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], rows, cols, depth])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            if input_shape[3] is not None:
                depth = input_shape[3] + self.padding[2][0] + self.padding[2][1]
            else:
                depth = None
            return tensor_shape.TensorShape([input_shape[0], rows, cols, depth, input_shape[4]])

    def call(self, inputs):
        if self.mode == 'constant':
            if merlintf.iscomplex(inputs):
                return tf.complex(tf.pad(tf.real(inputs), self.padding, mode=self.mode, constant_values=self.constant_values),
                                  tf.pad(tf.imag(inputs), self.padding, mode=self.mode, constant_values=self.constant_values))
            else:
                return tf.pad(inputs, self.padding, 'constant', constant_values=self.constant_values)
        else:
            return optotf.pad.pad3d_transpose(inputs, self.optotf_padding, mode=self.mode)  # gradient registered in optotf

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(Padding3DTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Padding4DTranspose(Layer):
    """Transpose Padding layer for 4D input (e.g. picture).
    This layer can add rows and columns of symmetric, reflected or replicated values,
    at the timestart, timeend, top, bottom, left, right, front, back side of an image tensor.
    Arguments:
    padding: Int, or tuple of 4 ints, or tuple of 4 tuples of 4 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 4 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_height_pad, symmetric_width_pad)`.
      - If tuple of 4 tuples of 4 ints:
        interpreted as
        `((timestart_pad, timeend_pad), (top_pad, bottom_pad), (left_pad, right_pad), (front_pad, back_pad))`
    mode: Padding mode (string)
        - symmetric: the symmetric value of border pixels x-1, x-2, ... are appended
        - reflect: the reflected value of border pixels x, x-1, x-2, ... are appended
        - replicate: the same border pixels x, x, ... are appended
        - constant: a constant value is appended
    constant_values: float, or tuple of 4 floats, or tuple of 4 tuples of 4 floats. Constant value to be appended if
        (mode='constant').
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, time, height, width, depths, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, time, height, width, depths)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    Input shape:
    6D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, time, rows, cols, depths, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, time, rows, cols, depths)`
    Output shape:
    6D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, padded_times, padded_rows, padded_cols, padded_depths, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, padded_times, padded_rows, padded_cols, padded_depths)`
    """

    def __init__(self, padding=(1, 1, 1, 1), mode='symmetric', constant_values=0, data_format=None, **kwargs):
        super(Padding4DTranspose, self).__init__(**kwargs)
        self.mode = mode
        self.optox = (True if 'optotf.pad' in sys.modules else False)
        if not self.optox and mode != 'constant':
            raise ValueError('Only constant padding is supported without optotf')
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding), (padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 4:
                raise ValueError('`padding` should have four elements. Found: ' + str(padding))
            time_padding = conv_utils.normalize_tuple(padding[0], 2, '1st entry of padding')
            height_padding = conv_utils.normalize_tuple(padding[1], 2, '2nd entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[2], 2, '3rd entry of padding')
            depth_padding = conv_utils.normalize_tuple(padding[3], 2, '4th entry of padding')
            self.padding = (time_padding, height_padding, width_padding, depth_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 4 ints '
                             '(symmetric_time_pad, symmetric_height_pad, symmetric_width_pad, symmetric_depth_pad), '
                             'or a tuple of 4 tuples of 4 ints '
                             '((timestart_pad, timeend_pad), (top_pad, bottom_pad), (left_pad, right_pad), (front_pad, back_pad)). '
                             'Found: ' + str(padding))
        if isinstance(constant_values, float):
            self.constant_values = ((constant_values, constant_values), (constant_values, constant_values),
                                    (constant_values, constant_values), (constant_values, constant_values))
        elif hasattr(constant_values, '__len__'):
            if len(constant_values) != 4:
                raise ValueError('`constant_values` should have four elements. Found: ' + str(constant_values))
            time_constant_values = conv_utils.normalize_tuple(constant_values[0], 2, '1st entry of constant_values')
            height_constant_values = conv_utils.normalize_tuple(constant_values[1], 2, '2nd entry of constant_values')
            width_constant_values = conv_utils.normalize_tuple(constant_values[2], 2, '3rd entry of constant_values')
            depth_constant_values = conv_utils.normalize_tuple(constant_values[3], 2, '4th entry of constant_values')
            self.constant_values = (time_constant_values, height_constant_values, width_constant_values, depth_constant_values)
        else:
            raise ValueError('`constant_values` should be either a float, '
                             'a tuple of 4 floats '
                             '(time_constant, height_constant, width_constant, depth_constant), '
                             'or a tuple of 4 tuples of 4 floats '
                             '((starttime_constant, endtime_constant), (top_constant, bottom_constant), (left_constant, right_constant), (front_constant, back_constant)). '
                             'Found: ' + str(constant_values))
        self.optotf_padding = list(flatten(self.padding))[::-1]
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                times = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                times = None
            if input_shape[3] is not None:
                rows = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                rows = None
            if input_shape[4] is not None:
                cols = input_shape[4] + self.padding[2][0] + self.padding[2][1]
            else:
                cols = None
            if input_shape[5] is not None:
                depth = input_shape[5] + self.padding[3][0] + self.padding[3][1]
            else:
                depth = None
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], times, rows, cols, depth])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                times = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                times = None
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[2][0] + self.padding[2][1]
            else:
                cols = None
            if input_shape[4] is not None:
                depth = input_shape[4] + self.padding[3][0] + self.padding[3][1]
            else:
                depth = None
            return tensor_shape.TensorShape([input_shape[0], times, rows, cols, depth, input_shape[5]])

    def call(self, inputs):
        if self.mode == 'constant':
            if merlintf.iscomplex(inputs):
                return tf.complex(tf.pad(tf.real(inputs), self.padding, mode=self.mode, constant_values=self.constant_values),
                                  tf.pad(tf.imag(inputs), self.padding, mode=self.mode, constant_values=self.constant_values))
            else:
                return tf.pad(inputs, self.padding, 'constant', constant_values=self.constant_values)
        else:
            if self.data_format == 'channels_first':
                axis = 2
            elif self.data_format == 'channels_last':
                axis = 1
            # xyz padding
            shape_in = inputs.shape
            x_list = tf.split(inputs, shape_in[axis], axis=axis)
            x = tf.concat(x_list, axis=0)
            x = tf.squeeze(x, axis=axis)
            x = optotf.pad.pad3d_transpose(x, self.optotf_padding[0:3], mode=self.mode)
            x_list = tf.split(x, shape_in[axis], axis=0)
            x = tf.stack(x_list, axis=axis)
            # t padding
            axis += 1
            shape_in = x.shape
            x_list = tf.split(x, shape_in[axis], axis=axis)
            x = tf.concat(x_list, axis=0)
            x = tf.squeeze(x, axis=axis)
            x = optotf.pad.pad3d_transpose(x, [self.optotf_padding[-1], 0, 0, 0, 0], mode=self.mode)
            x_list = tf.split(x, shape_in[axis], axis=0)
            return tf.stack(x_list, axis=axis)

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(Padding4DTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Aliases
Padding2Dt = Padding3D
Padding3Dt = Padding4D
Padding2DtTranspose = Padding3DTranspose
Padding3DtTranspose = Padding4DTranspose