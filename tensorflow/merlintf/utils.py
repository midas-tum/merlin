import tensorflow as tf

def nchw2nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc2nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def ncdhw2ndhwc(x):
    return tf.transpose(x, [0, 2, 3, 4, 1])

def ndhwc2ncdhw(x):
    return tf.transpose(x, [0, 4, 1, 2, 3])