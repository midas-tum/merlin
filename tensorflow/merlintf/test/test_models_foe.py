import unittest
import numpy as np
import tensorflow as tf
import merlintf

from merlintf.keras.models.foe import (
    FoE,
    PolarFoE,
    Real2chFoE,
    ComplexFoE,
    MagnitudeFoE
)

import tensorflow.keras.backend as K
#K.set_floatx('float64')

class PolarFoETest(unittest.TestCase):
    def test_FoE_polar_2d(self):
        self._test_FoE_polar('2D', 11)

    def test_FoE_polar_3d(self):
        self._test_FoE_polar('3D', (3, 5, 5))

    def test_FoE_polar_2dt(self):
        self._test_FoE_polar('2Dt', (5, 7, 7))

    def _test_FoE_polar(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        config = {
            'dim': dim,
            'K1': {
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1_abs': {
                'vmin': 0,
                'vmax': 2,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
            'f1_phi': {
                'vmin': -np.pi,
                'vmax':  np.pi,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 1,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters': nf_in})

        model = PolarFoE(config)

        if dim == '2D':
            x = merlintf.random_normal_complex((nBatch, M, N, 1), dtype=K.floatx())
        elif dim == '3D' or dim == '2Dt':
            x = merlintf.random_normal_complex((nBatch, D, M, N, 1), dtype=K.floatx())
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class MagnitudeFoETest(unittest.TestCase):
    def test_FoE_magnitude_2d(self):
        self._test_FoE_magnitude('2D', 11)

    def test_FoE_magnitude_3d(self):
        self._test_FoE_magnitude('3D', (3, 5, 5))

    def test_FoE_magnitude_2dt(self):
        self._test_FoE_magnitude('2Dt', (5, 7, 7))

    def _test_FoE_magnitude(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        config = {
            'dim': dim,
            'K1': {
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1_abs': {
                'vmin': 0,
                'vmax': 2,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters' : nf_in})

        model = MagnitudeFoE(config)

        if dim == '2D':
            x = merlintf.random_normal_complex((nBatch, M, N, 1), dtype=K.floatx())
        elif dim == '3D' or dim == '2Dt':
            x = merlintf.random_normal_complex((nBatch, D, M, N, 1), dtype=K.floatx())
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')        
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class ComplexFoETest(unittest.TestCase):
    def test_FoE_complex_2d(self):
        self._test_FoE_complex('2D', 11)

    def test_FoE_complex_3d(self):
        self._test_FoE_complex('3D', (3, 5, 5))

    def test_FoE_complex_2dt(self):
        self._test_FoE_complex('2Dt', (5, 7, 7))

    def _test_FoE_complex(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        config = {
            'dim': dim,
            'K1': {
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'vmin': -1,
                'vmax':  1,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters' : nf_in})

        model = ComplexFoE(config)

        if dim == '2D':
            x = merlintf.random_normal_complex((nBatch, M, N, 1), dtype=K.floatx())
        elif dim == '3D' or dim == '2Dt':
            x = merlintf.random_normal_complex((nBatch, D, M, N, 1), dtype=K.floatx())
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')

        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class Real2chFoETest(unittest.TestCase):
    def test_FoE_real2ch_2d(self):
        self._test_FoE_real2ch('2D', 11)

    def test_FoE_real2ch_3d(self):
        self._test_FoE_real2ch('3D', (3, 5, 5))

    def _test_FoE_real2ch(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        config = {
            'dim': dim,
            'K1': {
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'vmin': -1,
                'vmax':  1,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters': nf_in})

        model = Real2chFoE(config)

        dtype = tf.float32  #K.floatx()
        if dim == '2D':
            x = merlintf.random_normal_complex((nBatch, M, N, 1), dtype=dtype)
        elif dim == '3D' or dim == '2Dt':
            x = merlintf.random_normal_complex((nBatch, D, M, N, 1), dtype=dtype)
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class RealFoETest(unittest.TestCase):
    def test_FoE_real_1d(self):
        self._test_FoE_real('1D', 11)

    def test_FoE_real_2d(self):
        self._test_FoE_real('2D', 11)

    def test_FoE_real_3d(self):
        self._test_FoE_real('3D', (3, 5, 5))

    def _test_FoE_real(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        config = {
            'dim': dim,
            'K1': {
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'vmin': -1,
                'vmax':  1,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters' : nf_in})

        model = FoE(config)

        dtype = tf.float32  #K.floatx()
        if dim == '1D':
            x = tf.random.normal((nBatch, N, 1), dtype=dtype)
        elif dim == '2D':
            x = tf.random.normal((nBatch, M, N, 1), dtype=dtype)
        elif dim == '3D' or dim == '2Dt':
            x = tf.random.normal((nBatch, D, M, N, 1), dtype=dtype)
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

if __name__ == "__main__":
    unittest.main()