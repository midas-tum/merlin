import unittest
import numpy as np
import tensorflow as tf
from merlintf.keras.layers.data_consistency import DCPM
import merlintf
from merlintf.keras.layers.mri import (
    MulticoilForwardOp,
    MulticoilAdjointOp
)

import tensorflow.keras.backend as K
#K.set_floatx('float64')

class CgTest(unittest.TestCase):
    def testcg(self):

        A = MulticoilForwardOp(center=True)
        AH = MulticoilAdjointOp(center=True)

        dc = DCPM(A, AH, weight_init=1.0, max_iter=50, tol=1e-12)

        shape=(5,10,10,1)
        kshape=(5,3,10,10)
        x = merlintf.random_normal_complex(shape, dtype=K.floatx())
        y =  merlintf.random_normal_complex(kshape, dtype=K.floatx())

        mask = tf.ones(kshape, dtype=K.floatx())
        smaps = merlintf.random_normal_complex(kshape, dtype=K.floatx())

        tf_a = tf.Variable(np.array([1.1]), dtype=K.floatx(), trainable=True)
        tf_b = tf.Variable(np.array([1.1]), dtype=K.floatx(), trainable=True)

        # perform a gradient check:
        epsilon = 1e-5

        def compute_loss(a, b):
            arg = dc([x * tf.complex(a, tf.zeros_like(a)), y, mask, smaps], 
                     scale=1/b) # take 1/b
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
        print("grad_a: {:.7f} num_grad_a {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1)  # 1e-4

        # numerical gradient w.r.t. the weights
        l_bp = compute_loss(tf_a, tf_b+epsilon).numpy()
        l_bn = compute_loss(tf_a, tf_b-epsilon).numpy()
        grad_b_num = (l_bp - l_bn) / (2 * epsilon)

        print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
            grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1))
        self.assertTrue(np.abs(grad_b - grad_b_num) < 1)  # 1e-4

if __name__ == "__main__":
    unittest.main()