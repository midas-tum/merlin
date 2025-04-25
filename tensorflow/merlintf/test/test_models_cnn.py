import unittest
import merlintf
from merlintf.keras.models.cnn import ComplexCNN, Real2chCNN

import tensorflow.keras.backend as K
#K.set_floatx('float64')

class Real2chCNNTest(unittest.TestCase):
    def test_cnn_real2ch_2d(self):
        self._test_cnn_real2ch('2D', 3)

    def test_cnn_real2ch_2d_2(self):
        self._test_cnn_real2ch('2D', (3,5))

    def test_cnn_real2ch_3d(self):
        self._test_cnn_real2ch('3D', (3, 5, 5))

    def _test_cnn_real2ch(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128

        config = {
            'dim': dim,
            'filters': 64,
            'kernel_size': kernel_size,
            'num_layer': 5,
            'activation': 'relu'
        }

        model = Real2chCNN(**config)

        if dim == '2D':
            x = merlintf.random_normal_complex((nBatch, M, N, 1), dtype=K.floatx())
        elif dim == '3D' or dim == '2Dt':
            x = merlintf.random_normal_complex((nBatch, D, M, N, 1), dtype=K.floatx())
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

class ComplexCNNTest(unittest.TestCase):
    def test_cnn_complex_2d(self):
        self._test_cnn_complex('2D', 3, 'cReLU')

    def test_cnn_complex_2d_2(self):
        self._test_cnn_complex('2D', (3,5), 'ModReLU')

    def test_cnn_complex_3d(self):
        self._test_cnn_complex('3D', (3, 5, 5), 'ModReLU')

    def _test_cnn_complex(self, dim, kernel_size, activation):
        nBatch = 5
        D = 20
        M = 128
        N = 128

        config = {
            'dim': dim,
            'filters': 64,
            'kernel_size': kernel_size,
            'num_layer': 5,
            'activation': activation
        }

        model = ComplexCNN(**config)

        if dim == '2D':
            x = merlintf.random_normal_complex((nBatch, M, N, 1), dtype=K.floatx())
        elif dim == '3D' or dim == '2Dt':
            x = merlintf.random_normal_complex((nBatch, D, M, N, 1), dtype=K.floatx())
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

if __name__ == "__main__":
    unittest.main()