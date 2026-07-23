import tensorflow as tf

__all__ = [
    "complex_conv2d",
    "complex_conv3d",
    "complex_conv1d_transpose",
    "complex_conv2d_transpose",
    "complex_conv3d_transpose",
    "complex_conv2d_real_weight",
    "complex_conv3d_real_weight",
    "complex_conv2d_real_weight_transpose",
    "complex_conv3d_real_weight_transpose",
]

def complex_conv(conv_fun, x, weight, padding="VALID", strides=1, dilations=1, data_format=None):
    xre = tf.math.real(x)
    xim = tf.math.imag(x)

    wre = tf.math.real(weight)
    wim = tf.math.imag(weight)

    if data_format is not None:
        conv_rr = conv_fun(xre, wre, padding=padding, strides=strides, dilations=dilations, data_format=data_format)
        conv_ii = conv_fun(xim, wim, padding=padding, strides=strides, dilations=dilations, data_format=data_format)
        conv_ri = conv_fun(xre, wim, padding=padding, strides=strides, dilations=dilations, data_format=data_format)
        conv_ir = conv_fun(xim, wre, padding=padding, strides=strides, dilations=dilations, data_format=data_format)
    else:
        conv_rr = conv_fun(xre, wre, padding=padding, strides=strides, dilations=dilations)
        conv_ii = conv_fun(xim, wim, padding=padding, strides=strides, dilations=dilations)
        conv_ri = conv_fun(xre, wim, padding=padding, strides=strides, dilations=dilations)
        conv_ir = conv_fun(xim, wre, padding=padding, strides=strides, dilations=dilations)

    conv_re = conv_rr - conv_ii
    conv_im = conv_ir + conv_ri

    return tf.complex(conv_re, conv_im)

def complex_conv_transpose(conv_fun, x, weight, output_shape, strides, padding='SAME', data_format=None,
    dilations=None, name=None):
    xre = tf.math.real(x)
    xim = tf.math.imag(x)

    wre = tf.math.real(weight)
    wim = tf.math.imag(weight)

    convT_rr = conv_fun(xre, wre, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)
    convT_ii = conv_fun(xim, wim, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)
    convT_ri = conv_fun(xre, wim, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)
    convT_ir = conv_fun(xim, wre, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)

    convT_re = convT_rr + convT_ii
    convT_im = convT_ir - convT_ri

    return tf.complex(convT_re, convT_im)

def complex_conv_real_weight(conv_fun, x, weight, padding="VALID", strides=1, dilations=1, data_format=None):
    if data_format is not None:
        conv_rr = conv_fun(tf.math.real(x), weight, padding=padding, strides=strides, dilations=dilations, data_format=data_format)
        conv_ir = conv_fun(tf.math.imag(x), weight, padding=padding, strides=strides, dilations=dilations, data_format=data_format)
    else:
        conv_rr = conv_fun(tf.math.real(x), weight, padding=padding, strides=strides, dilations=dilations)
        conv_ir = conv_fun(tf.math.imag(x), weight, padding=padding, strides=strides, dilations=dilations)

    conv_re = conv_rr
    conv_im = conv_ir

    return tf.complex(conv_re, conv_im)

def complex_conv_real_weight_transpose(conv_fun, x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    convT_rr = conv_fun(tf.math.real(x), weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)
    convT_ir = conv_fun(tf.math.imag(x), weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)

    convT_re = convT_rr
    convT_im = convT_ir

    return tf.complex(convT_re, convT_im)

def complex_conv2d(x, weight, padding="VALID", strides=1, dilations=1):
    return complex_conv(tf.nn.conv2d, x, weight, padding=padding, strides=strides, dilations=dilations)

def safe_conv3d(x, weight, padding="VALID", strides=1, dilations=1, data_format=None, name=None):
    df = data_format or "NDHWC"
    if df in ["NCDHW", "channels_first"] and not tf.config.list_physical_devices('GPU'):
        x_last = tf.transpose(x, [0, 2, 3, 4, 1])
        if isinstance(strides, (list, tuple)) and len(strides) == 5:
            strides = [strides[0], strides[2], strides[3], strides[4], strides[1]]
        if isinstance(dilations, (list, tuple)) and len(dilations) == 5:
            dilations = [dilations[0], dilations[2], dilations[3], dilations[4], dilations[1]]
        out_last = tf.nn.conv3d(x_last, weight, strides=strides, padding=padding, data_format="NDHWC", dilations=dilations, name=name)
        return tf.transpose(out_last, [0, 4, 1, 2, 3])
    return tf.nn.conv3d(x, weight, strides=strides, padding=padding, data_format=df, dilations=dilations, name=name)

def safe_conv3d_transpose(x, weight, output_shape, strides, padding="SAME", data_format=None, dilations=None, name=None):
    df = data_format or "NDHWC"
    if df in ["NCDHW", "channels_first"] and not tf.config.list_physical_devices('GPU'):
        x_last = tf.transpose(x, [0, 2, 3, 4, 1])
        output_shape_last = [output_shape[0], output_shape[2], output_shape[3], output_shape[4], output_shape[1]]
        if isinstance(strides, (list, tuple)) and len(strides) == 5:
            strides = [strides[0], strides[2], strides[3], strides[4], strides[1]]
        if isinstance(dilations, (list, tuple)) and len(dilations) == 5:
            dilations = [dilations[0], dilations[2], dilations[3], dilations[4], dilations[1]]
        out_last = tf.nn.conv3d_transpose(x_last, weight, output_shape_last, strides=strides, padding=padding, data_format="NDHWC", dilations=dilations, name=name)
        return tf.transpose(out_last, [0, 4, 1, 2, 3])
    return tf.nn.conv3d_transpose(x, weight, output_shape, strides=strides, padding=padding, data_format=df, dilations=dilations, name=name)

def complex_conv3d(x, weight, padding="VALID", strides=1, dilations=1, data_format=None):
    return complex_conv(safe_conv3d, x, weight, padding=padding, strides=strides, dilations=dilations, data_format=data_format)

def complex_conv1d_transpose(x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    return complex_conv_transpose(tf.nn.conv1d_transpose, x, weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)

def complex_conv2d_transpose(x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    return complex_conv_transpose(tf.nn.conv2d_transpose, x, weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)

def complex_conv3d_transpose(x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    return complex_conv_transpose(safe_conv3d_transpose, x, weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)

def complex_conv2d_real_weight(x, weight, padding="VALID", strides=1, dilations=1):
    return complex_conv_real_weight(tf.nn.conv2d, x, weight, padding=padding, strides=strides, dilations=dilations)

def complex_conv3d_real_weight(x, weight, padding="VALID", strides=1, dilations=1, data_format=None):
    return complex_conv_real_weight(safe_conv3d, x, weight, padding=padding, strides=strides, dilations=dilations, data_format=data_format)

def complex_conv2d_real_weight_transpose(x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    return complex_conv_real_weight_transpose(tf.nn.conv2d_transpose, x, weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)

def complex_conv3d_real_weight_transpose(x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    return complex_conv_real_weight_transpose(safe_conv3d_transpose, x, weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)