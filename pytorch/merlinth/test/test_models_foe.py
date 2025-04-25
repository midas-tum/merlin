import unittest
import torch
import numpy as np
import merlinth
from merlinth.models.foe import (
    FoE,
    MagnitudeFoE,
    ComplexFoE,
    PolarFoE,
    Real2chFoE
)

class PolarFoETest(unittest.TestCase):
    def test_FoE_polar_2d(self):
        self._test_FoE_polar('2D', 11)

    def test_FoE_polar_3d(self):
        self._test_FoE_polar('3D', (3, 5, 5))

    def test_FoE_polar_2Dt(self):
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
                'in_channels' : 1,
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1_abs': {
                'num_channels': nf_in,
                'vmin': 0,
                'vmax': 2,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 0.01,
            },
            'f1_phi': {
                'num_channels': nf_in,
                'vmin': -np.pi,
                'vmax':  np.pi,
                'num_weights': nw,
                'base_type': 'linear',
                'init': 'linear',
                'init_scale': 1,
            },
        }
        if dim == '2Dt':
            config['K1'].update({'intermediate_filters' : nf_in})

        model = PolarFoE(config).cuda()

        if dim == '2D':
            x = merlinth.random_normal_complex((nBatch, 1, M, N)).cuda()
        elif dim == '3D' or dim == '2Dt':
            x =  merlinth.random_normal_complex((nBatch, 1, D, M, N)).cuda()
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class MagnitudeFoETest(unittest.TestCase):
    def test_FoE_magnitude_2d(self):
        self._test_FoE_magnitude('2D', 11)

    def test_FoE_magnitude_3d(self):
        self._test_FoE_magnitude('3D', (3, 5, 5))

    def test_FoE_magnitude_2Dt(self):
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
                'in_channels' : 1,
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1_abs': {
                'num_channels': nf_in,
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

        model = MagnitudeFoE(config).cuda()

        if dim == '2D':
            x = merlinth.random_normal_complex((nBatch, 1, M, N)).cuda()
        elif dim == '3D' or dim == '2Dt':
            x = merlinth.random_normal_complex((nBatch, 1, D, M, N)).cuda()
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')        
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class ComplexFoETest(unittest.TestCase):
    def test_FoE_complex_2d(self):
        self._test_FoE_complex('2D', 11)

    def test_FoE_complex_3d(self):
        self._test_FoE_complex('3D', (3, 5, 5))

    def test_FoE_complex_2Dt(self):
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
                'in_channels' : 1,
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'num_channels': nf_in,
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

        model = ComplexFoE(config).cuda()

        if dim == '2D':
            x = merlinth.random_normal_complex((nBatch, 1, M, N)).cuda()
        elif dim == '3D' or dim == '2Dt':
            x = merlinth.random_normal_complex((nBatch, 1, D, M, N)).cuda()
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
                'in_channels' : 2,
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'num_channels' : nf_in,
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

        model = Real2chFoE(config).cuda()

        if dim == '2D':
            x = merlinth.random_normal_complex((nBatch, 1, M, N)).cuda()
        elif dim == '3D' or dim == '2Dt':
            x = merlinth.random_normal_complex((nBatch, 1, D, M, N)).cuda()
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class RealFoETest(unittest.TestCase):
    def test_FoE_real_2d(self):
        self._test_FoE_real('2D', 11)

    def test_FoE_real_3d(self):
        self._test_FoE_real('3D', (3, 5, 5))

    @unittest.expectedFailure
    def test_FoE_real_2dt(self):
        self._test_FoE_real('2Dt', (3, 5, 5))

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
                'in_channels' : 1,
                'filters': nf_in,
                'kernel_size': kernel_size,
                'bound_norm': True,
                'zero_mean': True,
            },
            'f1': {
                'num_channels': nf_in,
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

        model = FoE(config).cuda()

        if dim == '2D':
            x = torch.randn(nBatch, 1, M, N).cuda()
        elif dim == '3D' or dim == '2Dt':
            x = torch.randn(nBatch, 1, D, M, N).cuda()
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

if __name__ == "__main__":
    unittest.main()