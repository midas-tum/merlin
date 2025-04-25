import unittest
import tensorflow as tf
import merlintf

from merlintf.keras.models.unet import (
    MagUNet,
    ComplexUNet,
    Real2chUNet
)
import tensorflow.keras.backend as K
#K.set_floatx('float64')

class UNetTest(unittest.TestCase):
    #TODO split unittests
    def test_UNet_2chreal_2d(self):
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='2chreal', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='2chreal', complex_input=True)

        # downsampling
        self._test_UNet('2D', 64, (3, 3), (2, 2), downsampling='st', network='2chreal', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), downsampling='st', network='2chreal', complex_input=True)

        # normalization
        self._test_UNet('2D', 64, (3, 3), (2, 2), normalization='BN', network='2chreal', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), normalization='IN', network='2chreal', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), normalization='BN', network='2chreal', complex_input=True)
        self._test_UNet('2D', 64, (3, 3), (2, 2), normalization='IN', network='2chreal', complex_input=True)

        # upsampling
        self._test_UNet('2D', 64, (3, 3), (2, 2), upsampling='us', network='2chreal', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), upsampling='us', network='2chreal', complex_input=True)

    def test_UNet_mag_2d(self):
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='mag', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='mag', complex_input=True)

    def test_UNet_complex_2d(self):
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='complex', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='complex', complex_input=True)

        # downsampling
        self._test_UNet('2D', 64, (3, 3), (2, 2), downsampling='st', network='complex', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), downsampling='st', network='complex', complex_input=True)

        # normalization
        self._test_UNet('2D', 64, (3, 3), (2, 2), normalization='BN', network='complex', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), normalization='IN', network='complex', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), normalization='BN', network='complex', complex_input=True)
        self._test_UNet('2D', 64, (3, 3), (2, 2), normalization='IN', network='complex', complex_input=True)

        # upsampling
        self._test_UNet('2D', 64, (3, 3), (2, 2), upsampling='us', network='complex', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), upsampling='us', network='complex', complex_input=True)

    def test_UNet_2chreal_3d(self):
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='2chreal', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='2chreal', complex_input=True)
        self._test_UNet('3D', 32, (1, 3, 3), (1, 2, 2), network='2chreal', complex_input=True)

        # downsampling
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), downsampling='st', network='2chreal', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), downsampling='st', network='2chreal', complex_input=True)

        # normalization
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), normalization='BN', network='2chreal', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), normalization='IN', network='2chreal', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), normalization='BN', network='2chreal', complex_input=True)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), normalization='IN', network='2chreal', complex_input=True)

        # upsampling
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), upsampling='us', network='2chreal', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), upsampling='us', network='2chreal', complex_input=True)

    def test_UNet_mag_3d(self):
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='mag', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='mag', complex_input=True)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='mag', complex_input=True, D=15, num_level=2)
        self._test_UNet('3D', 32, (1, 3, 3), (1, 2, 2), network='mag', complex_input=True)

    #@unittest.expectedFailure
    def test_UNet_complex_3d(self):
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='complex', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='complex', complex_input=True)

        # downsampling
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), downsampling='st', network='complex', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), downsampling='st', network='complex', complex_input=True)

        # normalization
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), normalization='BN', network='complex', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), normalization='IN', network='complex', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), normalization='BN', network='complex', complex_input=True)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), normalization='IN', network='complex', complex_input=True)

        # upsampling
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), upsampling='us', network='complex', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), upsampling='us', network='complex', complex_input=True)

    @unittest.expectedFailure
    def test_UNet_2chreal_3dt(self):
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), network='2chreal', complex_input=False)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), network='2chreal', complex_input=True)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), network='2chreal', complex_input=True)

        # downsampling
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), downsampling='st', network='2chreal', complex_input=False)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), downsampling='st', network='2chreal', complex_input=True)

        # normalization
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), normalization='BN', network='2chreal', complex_input=False)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), normalization='IN', network='2chreal', complex_input=False)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), normalization='BN', network='2chreal', complex_input=True)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), normalization='IN', network='2chreal', complex_input=True)

        # upsampling
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), upsampling='us', network='2chreal', complex_input=False)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), upsampling='us', network='2chreal', complex_input=True)

    @unittest.expectedFailure
    def test_UNet_complex_3dt(self):
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), network='complex', complex_input=False)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), network='complex', complex_input=True)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), network='complex', complex_input=True)

        # downsampling
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), downsampling='st', network='complex', complex_input=False)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), downsampling='st', network='complex', complex_input=True)

        # normalization
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), normalization='BN', network='complex', complex_input=False)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), normalization='IN', network='complex', complex_input=False)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), normalization='BN', network='complex', complex_input=True)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), normalization='IN', network='complex', complex_input=True)

        # upsampling
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), upsampling='us', network='complex', complex_input=False)
        self._test_UNet('3Dt', 32, (2, 3, 3, 3), (1, 2, 2, 2), upsampling='us', network='complex', complex_input=True)

    def test_UNet_mag_2d_padding(self):  # padding required
        self._test_UNet('2D', 32, (3, 3), (2, 2), network='mag', complex_input=True, M=28, N=32)
        self._test_UNet('2D', 32, (3, 3), (2, 2), network='mag', complex_input=True, M=28, N=25)

    def test_UNet_mag_3d_padding(self):  # padding required
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='mag', complex_input=True, D=20, M=28, N=32)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='mag', complex_input=True, D=20, M=28, N=25)

    #def test_UNet_complex_3d(self):
    #    self._test_UNet('3D', 32, (3, 3, 3), network='complex', complex_input=False)
    #    self._test_UNet('3D', 32, (3, 3, 3), network='complex', complex_input=True)

    def _test_UNet(self, dim, filters, kernel_size, down_size=(2,2,2), downsampling='mp', upsampling='tc', normalization='none', network='complex', complex_input=True, T=4, M=32, N=32, D=48, num_level=4):

        nBatch = 2

        if network == 'complex':
            model = ComplexUNet(dim, filters, kernel_size, down_size, num_level=num_level, downsampling=downsampling, upsampling=upsampling, normalization=normalization)
        elif network =='2chreal':
            model = Real2chUNet(dim, filters, kernel_size, down_size, num_level=num_level, downsampling=downsampling, upsampling=upsampling, normalization=normalization)
        else:
            model = MagUNet(dim, filters, kernel_size, down_size, num_level=num_level, downsampling=downsampling, upsampling=upsampling, normalization=normalization)

        if dim == '2D':
            if complex_input:
                x = merlintf.random_normal_complex((nBatch, M, N, 1), dtype=tf.float32)
            else:
                x = tf.random.normal((nBatch, M, N, 1), dtype=tf.float32)
        elif dim == '3D' or dim == '2Dt':
            if complex_input:
                x = merlintf.random_normal_complex((nBatch, M, N, D, 1), dtype=tf.float32)
            else:
                x = tf.random.normal((nBatch, M, N, D, 1), dtype=tf.float32)
        elif dim == '3Dt':
            if complex_input:
                x = merlintf.random_normal_complex((nBatch, T, M, N, D, 1), dtype=tf.float32)
            else:
                x = tf.random.normal((nBatch, T, M, N, D, 1), dtype=tf.float32)
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')

        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

if __name__ == "__main__":
    unittest.main()
