import tensorflow as tf

def complex_abs(z, eps=1e-9):
    return tf.math.sqrt(tf.math.real(tf.math.conj(z) * z) + eps)

def complex_norm(z, eps=1e-9):
    return z / tf.cast(complex_abs(z, eps), z.dtype)

def complex_angle(z, eps=1e-9):
    return tf.math.atan2(tf.math.imag(z), tf.math.real(z) + eps)

def complex_scale(x, scale):
    return tf.complex(tf.math.real(x) * scale, tf.math.imag(x) * scale)
    
def complex_dot(x, y, axis=None):
    return tf.reduce_sum(tf.math.conj(x) * y, axis=axis)

def complex2real(z, channel_last=True):
    stack_dim = -1 if channel_last else 1
    return tf.concat([tf.math.real(z), tf.math.imag(z)], stack_dim)

def real2complex(z, channel_last=True):
    stack_dim = -1 if channel_last else 1
    (real, imag) = tf.split(z, 2, axis=stack_dim)
    return tf.complex(real, imag)

def iscomplextf(x):
    if x.dtype == tf.complex64 or x.dtype == tf.complex128:
          return True
    else:
          return False

def random_normal_complex(shape, dtype=tf.float64):
    return tf.complex(tf.random.normal(shape, dtype=dtype), 
                      tf.random.normal(shape, dtype=dtype))