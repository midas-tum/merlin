import tensorflow as tf
import tensorflow.keras.backend as K

class Clip(tf.keras.constraints.Constraint):
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {
            'min_value' : self.min_value,
            'max_value' : self.max_value,
        }