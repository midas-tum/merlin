# import tensorflow as tf

# def complex_abs(z, eps=1e-6):
#     return tf.math.sqrt(tf.cast(tf.math.conj(z) * z, tf.float32) + eps)

# def complex_norm(z, eps=1e-6):
#     return z / tf.cast(complex_abs(z, eps), tf.complex64)

# def complex_angle(z, eps=1e-6):
#     return tf.math.atan2(tf.math.imag(z), tf.math.real(z) + eps)