import tensorflow as tf
import numpy as np

from merlintf.keras.models.foe import Regularizer
from merlintf.keras.layers.convolutional.padconv import (
    PadConv2D,
    PadConvScale2D,
    PadConvScale2DTranspose,
    PadConv3D,
    PadConvScale3D,
    PadConvScale3DTranspose
)
from merlintf.keras.layers.convolutional.complex_padconv import (
    ComplexPadConv2D,
    ComplexPadConvScale2D,
    ComplexPadConvScale2DTranspose,
    ComplexPadConv3D,
    ComplexPadConvScale3D,
    ComplexPadConvScale3DTranspose
)
from merlintf.keras.layers.complex_act import ModStudentT2
from merlintf.complex import (
    complex_norm,
    complex_abs
)
import unittest

__all__ = ['TDV2D', 'TDV3D']


class StudentT2(tf.keras.layers.Layer):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    def call(self, x):
        d = 1 + self.alpha * x**2
        return tf.math.log(d) / (2 * self.alpha), x / d

class MicroBlock(tf.keras.Model):
    def __init__(self, dim, num_features, bound_norm=False):
        super(MicroBlock, self).__init__()
        
        if dim == '2D':
            conv_module = PadConv2D
        elif dim == '3D':
            conv_module = PadConv3D
        else:
            raise RuntimeError(f"TDV regularizer not defined for {dim}!")

        self.conv1 = conv_module(num_features, kernel_size=3, bound_norm=bound_norm, use_bias=False)
        self.act = StudentT2(alpha=1)
        self.conv2 = conv_module(num_features, kernel_size=3, bound_norm=bound_norm, use_bias=False)

        # save the gradient of the the activation function for the backward path
        self.act_prime = None

    def call(self, x):
        a, ap = self.act(self.conv1(x))
        self.act_prime = ap
        x = x + self.conv2(a)
        return x

    def backward(self, grad_out):
        assert not self.act_prime is None
        out = grad_out + self.conv1.backward(self.act_prime*self.conv2.backward(grad_out))
        # if not is_trainable(self.act_prime): <-- TODO does this exist in TF?
        #     self.act_prime = None
        return out


class ComplexMicroBlock(tf.keras.Model):
    def __init__(self, dim, num_features, bound_norm=False):
        super(ComplexMicroBlock, self).__init__()

        if dim == '2D':
            conv_module = ComplexPadConv2D
        elif dim == '3D':
            conv_module = ComplexPadConv3D
        else:
            raise RuntimeError(f"TDV regularizer not defined for {dim}!")

        self.conv1 = conv_module(num_features, kernel_size=3, bound_norm=bound_norm, use_bias=False)
        self.act = ModStudentT2(alpha=1)
        self.conv2 = conv_module(num_features, kernel_size=3, bound_norm=bound_norm, use_bias=False)

        # save the gradient of the the activation function for the backward path
        self.act_prime = None

    def call(self, x):
        a, ap, apH = self.act(self.conv1(x))
        self.act_prime = ap
        self.act_primeH = apH
        x = x + self.conv2(a)
        return x

    def backward(self, grad_out):
        assert not self.act_prime is None
        z = self.conv2.backward(grad_out)
        zH = tf.math.conj(z)
        out = grad_out + self.conv1.backward(zH * self.act_primeH + z * tf.math.conj(self.act_prime))
        # if not is_trainable(self.act_prime): <-- TODO does this exist in TF?
        #     self.act_prime = None
        return out

class MacroBlock(tf.keras.Model):
    def __init__(self, dim, num_features, num_scales=3, multiplier=1, bound_norm=False, is_complex=False):
        super(MacroBlock, self).__init__()

        self.num_scales = num_scales

        if is_complex:
            micro_block_module = ComplexMicroBlock
        else:
            micro_block_module = MicroBlock

        if dim == '2D' and is_complex:
            conv_module_scale = ComplexPadConvScale2D
            conv_module_scale_transpose = ComplexPadConvScale2DTranspose
        elif dim == '2D':
            conv_module_scale = PadConvScale2D
            conv_module_scale_transpose = PadConvScale2DTranspose
        elif dim == '3D' and is_complex:
            conv_module_scale = ComplexPadConvScale3D
            conv_module_scale_transpose = ComplexPadConvScale3DTranspose
        elif dim == '3D':
            conv_module_scale = PadConvScale3D
            conv_module_scale_transpose = PadConvScale3DTranspose
        else:
            raise RuntimeError(f"MacroBlock not defined for {dim}!")

        # micro blocks
        self.mb = []
        for i in range(num_scales-1):
            b = [
                micro_block_module(dim, num_features * multiplier**i, bound_norm=bound_norm),
                micro_block_module(dim, num_features * multiplier**i, bound_norm=bound_norm)
            ]
            self.mb.append(b)
        # the coarsest scale has only one microblock
        self.mb.append([
                micro_block_module(dim, num_features * multiplier**(num_scales-1), bound_norm=bound_norm)
        ])

        # get conv module
        if dim == '2D':
            strides = 2
        elif dim == '3D':
            strides = (1,2,2)
        else:
            raise RuntimeError(f"MacroBlock not defined for {dim}!")

        # down/up sample
        self.conv_down = []
        self.conv_up = []
        for i in range(1, num_scales):
            self.conv_down.append(
                conv_module_scale(num_features * multiplier**i, strides=strides, kernel_size=3, use_bias=False, bound_norm=bound_norm)
            )
            self.conv_up.append(
                conv_module_scale_transpose(num_features * multiplier**(i-1), strides=strides, kernel_size=3, use_bias=False, bound_norm=bound_norm)
            )

    def call(self, x):
        assert len(x) == self.num_scales

        # down scale and feature extraction
        for i in range(self.num_scales-1):
            # 1st micro block of scale
            x[i] = self.mb[i][0](x[i])
            # down sample for the next scale
            x_i_down = self.conv_down[i](x[i])
            if x[i+1] is None:
                x[i+1] = x_i_down
            else:
                x[i+1] = x[i+1] + x_i_down
        
        # on the coarsest scale we only have one micro block
        x[self.num_scales-1] = self.mb[self.num_scales-1][0](x[self.num_scales-1])

        # up scale the features
        for i in range(self.num_scales-1)[::-1]:
            # first upsample the next coarsest scale
            x_ip1_up = self.conv_up[i](x[i+1], x[i].shape)
            # skip connection
            x[i] = x[i] + x_ip1_up
            # 2nd micro block of scale
            x[i] = self.mb[i][1](x[i])

        return x

    def backward(self, grad_x):

        # backward of up scale the features
        for i in range(self.num_scales-1):
            # 2nd micro block of scale
            grad_x[i] = self.mb[i][1].backward(grad_x[i])
            # first upsample the next coarsest scale
            grad_x_ip1_up = self.conv_up[i].backward(grad_x[i])
            # skip connection
            if grad_x[i+1] is None:
                grad_x[i+1] = grad_x_ip1_up
            else:
                grad_x[i+1] = grad_x[i+1] + grad_x_ip1_up

        # on the coarsest scale we only have one micro block
        grad_x[self.num_scales-1] = self.mb[self.num_scales-1][0].backward(grad_x[self.num_scales-1])

        # down scale and feature extraction
        for i in range(self.num_scales-1)[::-1]:
            # down sample for the next scale
            grad_x_i_down = self.conv_down[i].backward(grad_x[i+1], grad_x[i].shape)
            grad_x[i] = grad_x[i] + grad_x_i_down
            # 1st micro block of scale
            grad_x[i] = self.mb[i][0].backward(grad_x[i])
        
        return grad_x

class TDV(Regularizer):
    """
    total deep variation (TDV) regularizer
    """
    def __init__(self, config=None, file=None):
        super(TDV, self).__init__()

        if (config is None and file is None) or \
            (not config is None and not file is None):
            raise RuntimeError('specify EITHER a config dictionary OR a `.pth`-file!')

        # if not file is None:
        #     if not file.endswith('.pth'):
        #         raise ValueError('file needs to end with `.pth`!')
        #     checkpoint = torch.load(file)
        #     config = checkpoint['config']
        #     state_dict = checkpoint['model']
        #     self.tau = checkpoint['tau']
        # else:
        #     state_dict = None
        #     self.tau = 1.0

        self.out_channels = config['out_channels']
        self.num_features = config['num_features']
        self.multiplier = config['multiplier']
        self.num_mb = config['num_mb']
        self.is_complex = config['is_complex']

        if 'zero_mean' in config.keys():
            self.zero_mean = config['zero_mean']
        else:
            self.zero_mean = True
        if 'num_scales' in config.keys():
            self.num_scales = config['num_scales']
        else:
            self.num_scales = 3

        if config['dim'] == '2D' and self.is_complex:
            conv_module = ComplexPadConv2D
        elif config['dim'] == '2D':
            conv_module = PadConv2D
        elif config['dim'] == '3D' and self.is_complex:
            conv_module = ComplexPadConv3D
        elif config['dim'] == '3D':
            conv_module = PadConv3D
        else:
            raise RuntimeError(f"TDV regularizer not defined for {config['dim']}!")

        if self.is_complex:
            self._potential = self._potential_complex
            self._activation = self._activation_complex
        else:
            self._potential = self._potential_real
            self._activation = self._activation_real

        # construct the regularizer
        self.K1 = conv_module(self.num_features, 3, zero_mean=self.zero_mean, bound_norm=True, use_bias=False)

        self.mb = [MacroBlock(config['dim'], self.num_features, num_scales=self.num_scales, bound_norm=False, multiplier=self.multiplier, is_complex=self.is_complex) 
                                        for _ in range(self.num_mb)]

        self.KN = conv_module(self.out_channels, 1, bound_norm=False, use_bias=False)

        # if not state_dict is None:
        #     self.load_state_dict(state_dict)

    def _transformation(self, x):
        # extract features
        x = self.K1(x)
        # apply mb
        x = [x,] + [None for i in range(self.num_scales-1)]
        for i in range(self.num_mb):
            x = self.mb[i](x)
        # compute the output
        out = self.KN(x[0])
        return out

    def _activation_real(self, x):
        # scale by the number of features
        return tf.ones_like(x) / self.num_features

    def _potential_real(self, x):
        return x / self.num_features

    def _activation_complex(self, x):
        nx = complex_norm(x)
        return  nx / (2 * self.num_features)

    def _potential_complex(self, x):
        return complex_abs(x) / self.num_features

    def _transformation_T(self, grad_out):
        # compute the output
        grad_x = self.KN.backward(grad_out)
        # apply mb
        grad_x = [grad_x,] + [None for i in range(self.num_scales-1)]
        for i in range(self.num_mb)[::-1]:
            grad_x = self.mb[i].backward(grad_x)
        # extract features
        grad_x = self.K1.backward(grad_x[0])
        return grad_x

    def energy(self, x):
        x = self._transformation(x)
        return self._potential(x)

    def grad(self, x, get_energy=False):
        # compute the energy
        x = self._transformation(x)
        if get_energy:
            energy = self._potential(x)
        # and its gradient
        x = self._activation(x)
        grad = self._transformation_T(x)
        if get_energy:
            return energy, grad
        else:
            return grad


# to run execute: python -m unittest [-v] ddr.tdv
class GradientTest(unittest.TestCase):
    def _test_tdv_gradient(self, dim):
        # setup the data
        if dim == '2D':
            x = np.random.rand(2,64,64,1).astype(np.float32)
        elif dim == '3D':
            x = np.random.rand(2,10,64,64,1).astype(np.float32)
        else:
            raise ValueError
        x = tf.convert_to_tensor(x)

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

    @unittest.expectedFailure
    def test_tdv_gradient_2D(self):
        self._test_tdv_gradient('2D')

    @unittest.expectedFailure
    def test_tdv_gradient_3D(self):
        self._test_tdv_gradient('3D')

class ComplexGradientTest(unittest.TestCase):
    def _test_complex_tdv_gradient(self, dim):
        # setup the data
        if dim == '2D':
            x = np.random.rand(2,64,64,1).astype(np.float32) + 1j * np.random.rand(2,64,64,1).astype(np.float32)
        elif dim == '3D':
            x = np.random.rand(2,10,64,64,1).astype(np.float32) + 1j * np.random.rand(2,10,64,64,1).astype(np.float32) 
        else:
            raise ValueError
        x = tf.convert_to_tensor(x)

        # define the TDV regularizer
        config = {
            'dim': dim,
            'is_complex': True,
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 3,
            'num_mb': 2,
            'multiplier': 2,
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
        x = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        x = tf.convert_to_tensor(x.astype(np.complex64))
        
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

        x = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        x = tf.convert_to_tensor(x.astype(np.complex64))
        
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

        x = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        x = tf.convert_to_tensor(x.astype(np.complex64))
        
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

        x = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        x = tf.convert_to_tensor(x.astype(np.complex64))

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