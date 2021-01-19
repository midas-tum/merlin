import tensorflow as tf
from .complex import complex_scale
import tensorflow.keras.backend as K
from .complex_cg import CGClass
import unittest
import numpy as np

class DCGD(tf.keras.layers.Layer):
    def __init__(self, A, AH, weight_init=1.0, weight_scale=1.0, name='dc-gd', **kwargs):
        super().__init__()
        self.A = A
        self.AH = AH

        self.weight_init = weight_init
        self.weight_scale = weight_scale
    
    def build(self, input_shape):
        self._weight = self.add_weight(name='weight',
                                     shape=(1,),
                                     constraint=tf.keras.constraints.NonNeg(),
                                     initializer=tf.keras.initializers.Constant(self.weight_init))
    @property
    def weight(self):
        return self._weight * self.weight_scale

    def call(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        return x - complex_scale(self.AH(self.A(x, *constants) - y, *constants), self.weight * scale)


class DCPM(tf.keras.layers.Layer):
    def __init__(self, A, AH, weight_init=1.0, weight_scale=1.0, name='dc-pm', **kwargs):
        super().__init__()
        self.A = A
        self.AH = AH
        max_iter = kwargs.get('max_iter', 10)
        tol = kwargs.get('tol', 1e-10)
        self.prox = CGClass(A, AH, max_iter=max_iter, tol=tol)

        self.weight_init = weight_init
        self.weight_scale = weight_scale

    def build(self, input_shape):
        self._weight = self.add_weight(name='weight',
                                     shape=(1,),
                                     constraint=tf.keras.constraints.NonNeg(),
                                     initializer=tf.keras.initializers.Constant(self.weight_init))

    @property
    def weight(self):
        return self._weight * self.weight_scale

    def call(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        lambdaa = 1.0 / tf.math.maximum(self.weight * scale, 1e-9)
        return self.prox(lambdaa, x, y, *constants)

class CgTest(unittest.TestCase):
    def testcg(self):
        K.set_floatx('float64')

        from merlintf.keras_utils import mri
        
        A = mri.MulticoilForwardOp(center=True)
        AH = mri.MulticoilAdjointOp(center=True)

        dc = DCPM(A, AH, weight_init=1.0, max_iter=50, tol=1e-12)

        shape=(5,10,10,1)
        kshape=(5,3,10,10)
        x = tf.complex(tf.random.normal(shape, dtype=tf.float64), tf.random.normal(shape, dtype=tf.float64))
        y = tf.complex(tf.random.normal(kshape, dtype=tf.float64), tf.random.normal(kshape, dtype=tf.float64))
        mask = tf.ones(kshape, dtype=tf.float64)
        smaps = tf.complex(tf.random.normal(kshape, dtype=tf.float64), tf.random.normal(kshape, dtype=tf.float64))

        tf_a = tf.Variable(np.array([1.1]), trainable=True, dtype=tf.float64)
        tf_b = tf.Variable(np.array([1.1]), trainable=True, dtype=tf.float64)

        # perform a gradient check:
        epsilon = 1e-5

        def compute_loss(a, b):
            arg = dc([x*tf.complex(a, tf.zeros_like(a)), y, mask, smaps], scale=1/b) # take 1/b
            return 0.5 * tf.math.real(tf.reduce_sum(tf.math.conj(arg) * arg))

        with tf.GradientTape() as g:
            g.watch(x)
            # setup the model
            tf_loss = compute_loss(tf_a, tf_b)

        # backpropagate the gradient
        dLoss = g.gradient(tf_loss, [tf_a, tf_b])

        grad_a = dLoss[0].numpy()[0]
        grad_b = dLoss[1].numpy()[0]

        # numerical gradient w.r.t. the input
        l_ap = compute_loss(tf_a+epsilon, tf_b).numpy()
        l_an = compute_loss(tf_a-epsilon, tf_b).numpy()
        grad_a_num = (l_ap - l_an) / (2 * epsilon)
        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

        # numerical gradient w.r.t. the weights
        l_bp = compute_loss(tf_a, tf_b+epsilon).numpy()
        l_bn = compute_loss(tf_a, tf_b-epsilon).numpy()
        grad_b_num = (l_bp - l_bn) / (2 * epsilon)

        print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
            grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
        self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)

if __name__ == "__main__":
    unittest.test()