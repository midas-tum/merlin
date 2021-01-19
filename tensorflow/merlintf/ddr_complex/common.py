import tensorflow as tf
import unittest
import merlintf.keras_utils
import numpy as np
import six

__all__ = ['Scalar']

class Scalar(tf.keras.layers.Layer):
    def __init__(self, name, constraint=None, initializer=None):
        super().__init__(name=name)
        self.weight = self.add_weight(shape=(1,),
                name=name,
                constraint=constraint,
                initializer=initializer)
        
    def call(self, inputs):
        if merlintf.keras_utils.iscomplextf(inputs):
            return merlintf.keras_utils.complex_scale(inputs, self.weight)
        else:
            return self.weight * inputs
