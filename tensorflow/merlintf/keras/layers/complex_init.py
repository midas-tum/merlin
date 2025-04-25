import tensorflow as tf

def complex_initializer(base_initializer):
    f = base_initializer()

    def initializer(*args, dtype=tf.complex64, **kwargs):
        real = f(*args, **kwargs)
        imag = f(*args, **kwargs)
        return tf.complex(real, imag)

    return initializer