import tensorflow as tf
import merlintf
import six
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object

__all__ = ['cReLU',
           'ModReLU',
           'cPReLU',
           'ModPReLU',
           'cStudentT',
           'ModStudentT',
           'cStudentT2',
           'ModStudentT2',
           'Identity',
           'get',
           'Cardioid',
           'Cardioid2'
         ]

def get(identifier):
    if identifier is None:
        return Identity()
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif isinstance(identifier, dict):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError(
            'Could not interpret activation function identifier: {}'.format(
                identifier))


def Activation(identifier):
    if identifier is None:
        return Identity()
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError(
            'Could not interpret activation function identifier: {}'.format(
                identifier))


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='activation function')


def serialize(act):
    return serialize_keras_object(act)


class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, bias=0.1, trainable=True, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.bias_init = bias
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer = tf.keras.initializers.Constant(self.bias_init)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      dtype=self.dtype
                                      )
    def call(self, z):
        return tf.cast(tf.keras.activations.hard_sigmoid(merlintf.complex_abs(z) + self.bias), z.dtype)
    
    def get_config(self):
        config = {'bias': float(self.bias_init), 'trainable' : self.trainable}
        base_config = super(HardSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

class cReLU(tf.keras.layers.Layer):
    def call(self, z):
        actre = tf.keras.activations.relu(tf.math.real(z))
        actim = tf.keras.activations.relu(tf.math.imag(z))
        return tf.complex(actre, actim)
       
class Identity(tf.keras.layers.Layer):
    def call(self, z):
        return z

class ModReLU(tf.keras.layers.Layer):
    def __init__(self, bias=0.0, trainable=True, **kwargs):
        super(ModReLU, self).__init__(**kwargs)
        self.bias_init = bias
        self.trainable = trainable
    
    def build(self, input_shape):
        super(ModReLU, self).build(input_shape)
        initializer = tf.keras.initializers.Constant(self.bias_init)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      dtype=self.dtype
                                      )
    def call(self, z):
        return tf.cast(tf.keras.activations.relu(merlintf.complex_abs(z) + self.bias), z.dtype) * merlintf.complex_norm(z)

    def __str__(self):
        s = f"ModReLU: bias_init={self.bias_init}, trainable={self.trainable}"
        return s

    def get_config(self):
        config = {'bias': float(self.bias_init), 'trainable' : self.trainable}
        base_config = super(ModReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

class cPReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.1, trainable=False, **kwargs):
        super(cPReLU, self).__init__(**kwargs)
        self.alpha_init = alpha
        self.trainable = trainable

    def build(self, input_shape):
        super(cPReLU, self).build(input_shape)
        initializer = tf.keras.initializers.Constant(self.alpha_init)
        self.alpha_real = self.add_weight('alpha_real',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      dtype=self.dtype
                                      )
        # self.alpha_imag = self.add_weight('alpha_imag',
        #                               shape=(input_shape[-1]),
        #                               initializer=initializer,
        #                               )

    def call(self, z):
        zre = tf.math.real(z)
        zim = tf.math.imag(z)
        actre = tf.maximum(0.0, zre) + self.alpha_real * tf.minimum(0.0, zre)
        actim = tf.maximum(0.0, zim) + self.alpha_real * tf.minimum(0.0, zim)

        return tf.complex(actre, actim)

    def __str__(self):
        s = f"cPReLU: alpha_init={self.alpha_init}, trainable={self.trainable}"
        return s

    def get_config(self):
        config = {'alpha': float(self.alpha_init), 'trainable' : self.trainable}
        base_config = super(cPReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class ModPReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.1, bias=0, trainable=False, **kwargs):
        super(ModPReLU, self).__init__(**kwargs)
        self.alpha_init = alpha
        self.bias_init = bias
        self.trainable = trainable

    def build(self, input_shape):
        super(ModPReLU, self).build(input_shape)
        initializer = tf.keras.initializers.Constant(self.bias_init)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer,
                                      dtype=self.dtype
                                      )
        initializer_alpha = tf.keras.initializers.Constant(self.alpha_init)
        self.alpha = self.add_weight('alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      dtype=self.dtype
                                      )
    def call(self, z):
        act = tf.maximum(0.0, merlintf.complex_abs(z) + self.bias) + self.alpha * tf.minimum(0.0, merlintf.complex_abs(z) + self.bias)
        return tf.cast(act, z.dtype) * merlintf.complex_norm(z)

    def __str__(self):
        s = f"ModPReLU: alpha_init={self.alpha_init}, bias_init={self.bias_init}, trainable={self.trainable}"
        return s

    def get_config(self):
        config = {'bias': float(self.bias_init), 'alpha' : float(self.alpha_init), 'trainable' : self.trainable}
        base_config = super(ModPReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class Cardioid(tf.keras.layers.Layer):
    def __init__(self, bias=2.0, trainable=True, **kwargs):
        super(Cardioid, self).__init__(**kwargs)
        self.bias_init = bias
        self.trainable = trainable

    def build(self, input_shape):
        super(Cardioid, self).build(input_shape)
        initializer_bias = tf.keras.initializers.Constant(self.bias_init)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_bias,
                                      trainable=self.trainable,
                                      dtype=self.dtype
                                      )
    def call(self, z):
        phase = merlintf.complex_angle(z) + self.bias
        cos = tf.cast(tf.math.cos(phase), z.dtype) 

        return 0.5 * (1 + cos) * z
        
    def __str__(self):
        s = f"Cardioid: bias_init={self.bias_init}, trainable={self.trainable}"
        return s

    def get_config(self):
        config = {'bias': float(self.bias_init), 'trainable' : self.trainable}
        base_config = super(Cardioid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

class Cardioid2(tf.keras.layers.Layer):
    def __init__(self, bias=2.0, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.bias_init = bias
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer_bias = tf.keras.initializers.Constant(self.bias_init)
        self.bias = self.add_weight('bias',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_bias,
                                      trainable=self.trainable,
                                      dtype=self.dtype
                                      )
    def call(self, z):
        phase = merlintf.complex_angle(z) + self.bias
        sin = tf.cast(tf.math.sin(phase), z.dtype) 
        mz = tf.cast(merlintf.complex_abs(z), z.dtype)
        cos = tf.cast(tf.math.cos(phase), z.dtype) 

        fx = 0.5 * (1 + cos) * z

        dfx = 0.5 + 0.5 * cos + 0.25 * 1j * sin
        
        dfxH = - 0.25 * 1j * sin * (z * z) / (mz * mz)

        return fx, dfx, dfxH
        
    def __str__(self):
        s = f"Cardioid2: bias_init={self.bias_init}, trainable={self.trainable}"
        return s

    def get_config(self):
        config = {'bias': float(self.bias_init), 'trainable' : self.trainable}
        base_config = super(Cardioid2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class cStudentT(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0, trainable=False, **kwargs):
        super(cStudentT, self).__init__(**kwargs)
        self.alpha_init = alpha
        self.trainable = trainable

    def build(self, input_shape):
        super(cStudentT, self).build(input_shape)
        initializer_alpha = tf.keras.initializers.Constant(self.alpha_init)
        self.alpha = self.add_weight('alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      trainable=self.trainable,
                                      dtype=self.dtype
                                      )
    def _calc(self, x):
        d = 1 + self.alpha * x**2
        return tf.math.log(d) / (2 * self.alpha)

    def call(self, z):
        zre = tf.math.real(z)
        zim = tf.math.imag(z)

        actre = self._calc(zre)
        actim = self._calc(zim)
        return tf.complex(actre, actim)

    def __str__(self):
        s = f"cStudentT: alpha_init={self.alpha_init}, trainable={self.trainable}"
        return s

    def get_config(self):
        config = {'alpha': float(self.alpha_init), 'trainable' : self.trainable}
        base_config = super(cStudentT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class cStudentT2(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha_init = alpha
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer_alpha = tf.keras.initializers.Constant(self.alpha_init)
        self.alpha = self.add_weight('alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      trainable=self.trainable,
                                      dtype=self.dtype
                                      )
    def call(self, z):
        zre = tf.math.real(z)
        zim = tf.math.imag(z)

        def g(x):
            d = 1 + self.alpha * x**2
            return tf.math.log(d) / (2 * self.alpha)

        def h(x):
            d = 1 + self.alpha * x**2
            return x / d

        act = tf.complex(g(zre), g(zim))

        dfzH = tf.cast(0.5 * (h(zre) - h(zim)), z.dtype)
        dfz  = tf.cast(0.5 * (h(zre) + h(zim)), z.dtype)
       
        return act, dfz, dfzH

    def __str__(self):
        s = f"cStudentT2: alpha_init={self.alpha_init}, trainable={self.trainable}"
        return s

    def get_config(self):
        config = {'alpha': float(self.alpha_init), 'trainable' : self.trainable}
        base_config = super(cStudentT2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ModStudentT(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0, beta=0.1, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha_init = alpha
        self.beta_init = beta
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer_beta = tf.keras.initializers.Constant(self.beta_init)
        self.beta = self.add_weight(name='beta',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_beta,
                                      trainable=self.trainable,
                                      dtype=self.dtype
                                      )
        initializer_alpha = tf.keras.initializers.Constant(self.alpha_init)
        self.alpha = self.add_weight(name='alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      trainable=self.trainable,
                                      dtype=self.dtype,
                                      )
    def _calc(self, x):
        d = 1 + self.alpha * x**2
        return tf.math.log(d) / (2 * self.alpha)

    def call(self, z):
        act = self._calc(merlintf.complex_abs(z) + self.beta)
        return tf.cast(act, z.dtype) * merlintf.complex_norm(z)

    def __str__(self):
        s = f"ModStudentT: alpha_init={self.alpha_init}, beta_init={self.beta_init}, trainable={self.trainable}"
        return s

    def get_config(self):
        config = {'alpha': float(self.alpha_init), 'beta': float(self.beta_init), 'trainable' : self.trainable}
        base_config = super(ModStudentT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ModStudentT2(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0, beta=0.1, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha_init = alpha
        self.beta_init = beta
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        initializer_beta = tf.keras.initializers.Constant(self.beta_init)
        self._beta = self.add_weight('beta',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_beta,
                                      trainable=self.trainable,
                                      dtype=self.dtype
                                      )

        initializer_alpha = tf.keras.initializers.Constant(self.alpha_init)
        self._alpha = self.add_weight('alpha',
                                      shape=(input_shape[-1]),
                                      initializer=initializer_alpha,
                                      trainable=self.trainable,
                                      dtype=self.dtype
                                      )

    def call(self, z):
        mz = tf.cast(merlintf.complex_abs(z), z.dtype)
        nz = merlintf.complex_norm(z)

        alpha = tf.cast(self._alpha, z.dtype)
        beta = tf.cast(self._beta, z.dtype)

        d = 1 + alpha * (mz + beta)**2
        
        act = tf.math.log(d) / (2 * alpha) * nz

        dfx = (mz + beta) / (2 * d) + tf.math.log(d)  / (4 * alpha * mz)
        dfxH = (z * z) / (2 * mz ** 2) * ((mz + beta)/d - tf.math.log(d) / (2 * alpha * mz))

        return act, dfx, dfxH

    def __str__(self):
        s = f"ModStudentT2: alpha_init={self.alpha_init}, beta_init={self.beta_init}, trainable={self.trainable}"
        return s

    def get_config(self):
        config = {'alpha': float(self.alpha_init), 'beta': float(self.beta_init), 'trainable' : self.trainable}
        base_config = super(ModStudentT2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
