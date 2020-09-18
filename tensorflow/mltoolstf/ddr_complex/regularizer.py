import tensorflow as tf

__all__ = ['Regularizer']

class Regularizer(tf.keras.Model):
    """
    Basic regularization function
    """

    def __init__(self):
        super(Regularizer, self).__init__()

    def forward(self, x):
        return self.grad(x)

    def energy(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

