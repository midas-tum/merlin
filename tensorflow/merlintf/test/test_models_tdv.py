import unittest
import numpy as np
import tensorflow as tf
from merlintf import random_normal_complex

from merlintf.keras.models.tdv import (
    TDV,
    ComplexMicroBlock,
    MacroBlock
)

import tensorflow.keras.backend as K
K.set_floatx('float64')

# to run execute: python -m unittest [-v] ddr.tdv
class GradientTest(unittest.TestCase):
    def _test_tdv_gradient(self, dim):
        # setup the data
        if dim == '2D':
            x = shape = (2,64,64,1)
        elif dim == '3D':
            x = shape = (2,10,64,64,1)
        else:
            raise ValueError
        x = tf.random.normal(shape, dtype=K.floatx())

        # define the TDV regularizer
        config = {
            'dim': dim,
            'is_complex': False,
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 3,
            'num_mb': 2,
            'multiplier': 2,
        }
        R = TDV(config)

        def compute_loss(scale):
            return tf.reduce_sum(R.energy(scale*x))
        
        scale = 1.
        
        # compute the gradient using the implementation
        grad_scale = tf.reduce_sum(x*R.grad(scale*x))

        # check it numerically
        epsilon = 1e-4

        l_p = compute_loss(scale+epsilon)
        l_n = compute_loss(scale-epsilon)
        grad_scale_num = (l_p - l_n) / (2 * epsilon)

        condition = np.abs(grad_scale - grad_scale_num) < 1e-3
        print(f'grad_scale: {grad_scale:.7f} num_grad_scale {grad_scale_num:.7f} success: {condition}')
        self.assertTrue(condition)

    #@unittest.expectedFailure
    def test_tdv_gradient_2D(self):
        self._test_tdv_gradient('2D')

    #@unittest.expectedFailure
    def test_tdv_gradient_3D(self):
        self._test_tdv_gradient('3D')

class ComplexGradientTest(unittest.TestCase):
    def _test_complex_tdv_gradient(self, dim):
        # setup the data
        if dim == '2D':
            shape = (2,64,64,1)
        elif dim == '3D':
            shape = (2,10,64,64,1)
        else:
            raise ValueError
        x = random_normal_complex(shape, dtype=K.floatx())

        # define the TDV regularizer
        config = {
            'dim': dim,
            'is_complex': True,
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 3,
            'num_mb': 2,
            'multiplier': 1,
        }
        R = TDV(config)
        
        with tf.GradientTape() as g:
            g.watch(x)
            loss = 0.5 * tf.reduce_sum(R.energy(x))
        grad_x = g.gradient(loss, x)
        x_autograd = grad_x.numpy()

        KHKx = R.grad(x)
        x_bwd = KHKx.numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_complex_tdv_gradient_2D(self):
        self._test_complex_tdv_gradient('2D')

    def test_complex_tdv_gradient_3D(self):
        self._test_complex_tdv_gradient('3D')

class TestComplexMicroBlock(unittest.TestCase):
    def _test_gradient(self, dim):
        # setup the data
        nf = 32
        if dim == '2D':
            shape = (2,64,64,nf)
        elif dim == '3D':
            shape = (2,10,64,64,nf)
        else:
            raise ValueError
        x = random_normal_complex(shape, dtype=K.floatx())
        
        R = ComplexMicroBlock(dim, nf)

        with tf.GradientTape() as g:
            g.watch(x)
            Kx = R(x)
            loss = 0.5 * tf.reduce_sum(tf.math.conj(Kx) * Kx)
        grad_x = g.gradient(loss, x)
        x_autograd = grad_x.numpy()

        KHKx = R.backward(Kx)
        x_bwd = KHKx.numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_2D(self):
        self._test_gradient('2D')
        
    def test_3D(self):
        self._test_gradient('3D')

class TestComplexMacroBlock(unittest.TestCase):
    def _test_gradient(self, dim):
        # setup the data
        nf = 32
        if dim == '2D':
            shape = (2,64,64,nf)
        elif dim == '3D':
            shape = (2,10,64,64,nf)
        else:
            raise ValueError

        x = random_normal_complex(shape, dtype=K.floatx())
        
        R = MacroBlock(dim, nf, num_scales=1, is_complex=True)
        
        with tf.GradientTape() as g:
            g.watch(x)
            Kx = R([x])
            loss = 0.5 * tf.reduce_sum(tf.math.conj(Kx) * Kx)
        grad_x = g.gradient(loss, x)
        x_autograd = grad_x.numpy()

        KHKx = R.backward(Kx)
        x_bwd = KHKx[0].numpy()

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)
    
    def test_2D(self):
        self._test_gradient('2D')

    def test_3D(self):
        self._test_gradient('3D')

class TestEnergy(unittest.TestCase):
    def _test_gradient(self, dim):
        # setup the data
        if dim == '2D':
            shape = (1,1,1,1)
        elif dim == '3D':
            shape = (1,1,1,1,1)
        else:
            raise ValueError

        x = random_normal_complex(shape, dtype=K.floatx())
        
        # define the TDV regularizer
        config ={
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 2,
            'num_mb': 1,
            'multiplier': 2,
            'dim': dim,
            'is_complex': True,
        }
        R = TDV(config)
        
        with tf.GradientTape() as g:
            g.watch(x)
            Kx = R._potential(x)
            loss = 0.5 * tf.reduce_sum(Kx)
        
        grad_x = g.gradient(loss, x)
        x_autograd = grad_x.numpy()

        KHKx = R._activation(x)
        x_bwd = KHKx.numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_2D(self):
        self._test_gradient('2D')

    def test_3D(self):
        self._test_gradient('3D')

class TestTransformation(unittest.TestCase):
    def _test_gradient(self, dim):
        # setup the data
        if dim == '2D':
            shape = (2,64,64,1)
        elif dim == '3D':
            shape = (2, 10, 64, 64, 1)
        else:
            raise ValueError

        x = random_normal_complex(shape, dtype=K.floatx())

        # define the TDV regularizer
        config ={
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 2,
            'num_mb': 1,
            'multiplier': 2,
            'is_complex': True,
            'dim' : dim,
        }
        R = TDV(config)

        with tf.GradientTape() as g:
            g.watch(x)
            Kx = R._transformation(x)
            loss = 0.5 * tf.reduce_sum(tf.math.conj(Kx) * Kx)
        
        grad_x = g.gradient(loss, x)
        x_autograd = grad_x.numpy()

        KHKx = R._transformation_T(Kx)
        x_bwd = KHKx.numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_2D(self):
        self._test_gradient('2D')

    def test_3D(self):
        self._test_gradient('3D')

if __name__ == "__main__":
    unittest.test()