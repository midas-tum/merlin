import tensorflow as tf
from tensorflow.signal import fft2d, ifft2d, ifftshift, fftshift
from .complex import complex_scale, complex_dot
import tensorflow.keras.backend as K
from .complex_cg import CGClass
import unittest
import numpy as np

class Smaps(tf.keras.layers.Layer):
    def call(self, img, smaps):
        return tf.expand_dims(img, -3) * smaps

class SmapsAdj(tf.keras.layers.Layer):
    def call(self, coilimg, smaps):
        return tf.reduce_sum(coilimg * tf.math.conj(smaps), -3)

class MaskKspace(tf.keras.layers.Layer):
    def call(self, kspace, mask):
        return complex_scale(kspace, mask)

class IFFT2c(tf.keras.layers.Layer):
    def call(self, kspace, *args):
        axes = [tf.rank(kspace)-2, tf.rank(kspace)-1] # axes have to be positive...
        dtype = tf.math.real(kspace).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), dtype))
        return complex_scale(fftshift(ifft2d(ifftshift(kspace, axes=axes)), axes=axes), scale)

class FFT2c(tf.keras.layers.Layer):
    def call(self, image, *args):
        dtype = tf.math.real(image).dtype
        axes = [tf.rank(image)-2, tf.rank(image)-1] # axes have to be positive...
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), dtype))
        return  complex_scale(fftshift(fft2d(ifftshift(image, axes=axes)), axes=axes), 1/scale)

class IFFT2(tf.keras.layers.Layer):
    def call(self, kspace, *args):
        dtype = tf.math.real(kspace).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), dtype))
        return complex_scale(ifft2d(kspace), scale)

class FFT2(tf.keras.layers.Layer):
    def call(self, image, *args):
        dtype = tf.math.real(image).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), dtype))
        return  complex_scale(fft2d(image), 1/scale)

class ForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = FFT2c()
        else:
            self.fft2 = FFT2()
        self.mask = MaskKspace()

    def call(self, image, mask):
        kspace = self.fft2(image[...,0])
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace

class AdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = IFFT2c()
        else:
            self.ifft2 = IFFT2()

    def call(self, kspace, mask):
        masked_kspace = self.mask(kspace, mask)
        img = self.ifft2(masked_kspace)
        return tf.expand_dims(img, -1)

class MulticoilForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = FFT2c()
        else:
            self.fft2 = FFT2()
        self.mask = MaskKspace()
        self.smaps = Smaps()

    def call(self, image, mask, smaps):
        coilimg = self.smaps(image[...,0], smaps)
        kspace = self.fft2(coilimg)
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace

class MulticoilAdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = IFFT2c()
        else:
            self.ifft2 = IFFT2()
        self.adj_smaps = SmapsAdj()

    def call(self, kspace, mask, smaps):
        masked_kspace = self.mask(kspace, mask)
        coilimg = self.ifft2(masked_kspace)
        img = self.adj_smaps(coilimg, smaps)
        return tf.expand_dims(img, -1)


class DCGD2D(tf.keras.layers.Layer):
    def __init__(self, config, center=False, multicoil=True, name='dc-gd'):
        super().__init__()
        if multicoil:
            self.A = MulticoilForwardOp(center)
            self.AH = MulticoilAdjointOp(center)
        else:
            self.A = ForwardOp(center)
            self.AH = AdjointOp(center)

        self.weight_config = config
        self.train_scale = config['lambda']['train_scale'] if 'train_scale' in config['lambda'] else 1
    
    def build(self, input_shape):
        self._weight = self.add_weight(name='weight',
                                     shape=(1,),
                                     constraint=tf.keras.constraints.NonNeg(),
                                     initializer=tf.keras.initializers.Constant(self.weight_config['lambda']['init']))
    @property
    def weight(self):
        return self._weight * self.train_scale

    def call(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        return x - complex_scale(self.AH(self.A(x, *constants) - y, *constants), self.weight * scale)


class DCPM2D(tf.keras.layers.Layer):
    def __init__(self, config, center=False, multicoil=True, name='dc-pm', **kwargs):
        super().__init__()
        if multicoil:
            A = MulticoilForwardOp(center)
            AH = MulticoilAdjointOp(center)
            max_iter = kwargs.get('max_iter', 10)
            tol = kwargs.get('tol', 1e-10)
            self.prox = CGClass(A, AH, max_iter=max_iter, tol=tol)
        else:
            raise ValueError

        self.weight_config = config
        self.train_scale = config['lambda']['train_scale'] if 'train_scale' in config['lambda'] else 1

    def build(self, input_shape):
        self._weight = self.add_weight(name='weight',
                                     shape=(1,),
                                     constraint=tf.keras.constraints.NonNeg(),
                                     initializer=tf.keras.initializers.Constant(self.weight_config['lambda']['init']))

    @property
    def weight(self):
        return self._weight * self.train_scale

    def call(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        lambdaa = 1.0 / tf.math.maximum(self.weight * scale, 1e-9)
        return self.prox(lambdaa, x, y, *constants)

class CgTest(unittest.TestCase):
    def testcg(self):
        K.set_floatx('float64')
        config = {'lambda' : {'init' : 1.0}}
        dc = DCPM2D(config, center=True, multicoil=True, max_iter=50, tol=1e-12)

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