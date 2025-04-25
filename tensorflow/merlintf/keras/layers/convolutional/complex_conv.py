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

def complex_conv(conv_fun, x, weight, padding="VALID", strides=1, dilations=1):
    xre = tf.math.real(x)
    xim = tf.math.imag(x)

    wre = tf.math.real(weight)
    wim = tf.math.imag(weight)

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

def complex_conv_real_weight(conv_fun, x, weight, padding="VALID", strides=1, dilations=1):
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

def complex_conv3d(x, weight, padding="VALID", strides=1, dilations=1):
    return complex_conv(tf.nn.conv3d, x, weight, padding=padding, strides=strides, dilations=dilations)

def complex_conv1d_transpose(x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    return complex_conv_transpose(tf.nn.conv1d_transpose, x, weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)

def complex_conv2d_transpose(x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    return complex_conv_transpose(tf.nn.conv2d_transpose, x, weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)

def complex_conv3d_transpose(x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    return complex_conv_transpose(tf.nn.conv3d_transpose, x, weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)

def complex_conv2d_real_weight(x, weight, padding="VALID", strides=1, dilations=1):
    return complex_conv_real_weight(tf.nn.conv2d, x, weight, padding=padding, strides=strides, dilations=dilations)

def complex_conv3d_real_weight(x, weight, padding="VALID", strides=1, dilations=1):
    return complex_conv_real_weight(tf.nn.conv3d, x, weight, padding=padding, strides=strides, dilations=dilations)

def complex_conv2d_real_weight_transpose(x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    return complex_conv_real_weight_transpose(tf.nn.conv2d_transpose, x, weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)

def complex_conv3d_real_weight_transpose(x, weight, output_shape, strides, padding="SAME", dilations=None, data_format=None, name=None):
    return complex_conv_real_weight_transpose(tf.nn.conv3d_transpose, x, weight, output_shape, padding=padding, strides=strides, dilations=dilations, data_format=data_format, name=name)