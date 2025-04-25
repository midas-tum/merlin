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

def complex2real(z, channel_last=True, dtype=tf.keras.backend.floatx()):
    stack_dim = -1 if channel_last else 1
    return tf.cast(tf.concat([tf.math.real(z), tf.math.imag(z)], stack_dim), dtype)

def real2complex(z, channel_last=True):
    stack_dim = -1 if channel_last else 1
    (real, imag) = tf.split(z, 2, axis=stack_dim)
    return tf.complex(real, imag)

def complex2magpha(z, channel_last=True):
    stack_dim = -1 if channel_last else 1
    return tf.concat([complex_abs(z), complex_angle(z)], stack_dim)

def magpha2complex(z, channel_last=True):
    stack_dim = -1 if channel_last else 1
    (mag, pha) = tf.split(z, 2, axis=stack_dim)
    return tf.complex(mag * tf.math.cos(pha), mag * tf.math.sin(pha))

def iscomplex(x):
    if x.dtype == tf.complex64 or x.dtype == tf.complex128:
          return True
    else:
          return False

def random_normal_complex(shape, dtype=tf.keras.backend.floatx()):
    return tf.complex(tf.random.normal(shape, dtype=dtype), 
                      tf.random.normal(shape, dtype=dtype))

def numpy2tensor(x, add_batch_dim=False, add_channel_dim=False, channel_last=True, dtype=tf.dtypes.complex64):
    x = tf.cast(tf.convert_to_tensor(x), dtype)
    if add_batch_dim:
        x = tf.expand_dims(x, 0)
    if add_channel_dim:
        if channel_last:
            x = tf.expand_dims(x, -1)
        else:
            x = tf.expand_dims(x, 1)
    return x

def tensor2numpy(x):
    return x.numpy()