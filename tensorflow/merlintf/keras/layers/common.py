import tensorflow as tf
import unittest
import merlintf
import numpy as np

__all__ = ['Scalar']

class Scalar(tf.keras.layers.Layer):
    def __init__(self, name, scale=1, trainable=True,
                    constraint=None, initializer=None):
        super().__init__(name=name)

        self.scale = scale # to potentially accelerate training!
        self.weight_constraint = constraint
        self.weight_initializer = initializer
        self.weight_trainable = trainable
        self.weight_name = name

    def build(self, input_shape):
        self._weight = self.add_weight(shape=(1,),
                name=self.weight_name,
                constraint=self.weight_constraint,
                initializer=self.weight_initializer,
                trainable=self.weight_trainable,
                dtype=self.dtype
                )

    @property
    def weight(self):
        return self._weight * self.scale
        
    def call(self, inputs):
        if merlintf.iscomplex(inputs):
            return merlintf.complex_scale(inputs, self.weight)
        else:
            return self.weight * inputs
