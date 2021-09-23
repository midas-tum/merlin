import tensorflow as tf
from merlintf.keras.layers.fft import FFT2, FFT2c, IFFT2, IFFT2c
import merlintf
import unittest
import numpy as np

class Smaps(tf.keras.layers.Layer):
    def __init__(self, coil_axis=-3):
        super().__init__()
        self.coil_axis = coil_axis
        
    def call(self, img, smaps):
        return tf.expand_dims(img, self.coil_axis) * smaps

class SmapsAdj(tf.keras.layers.Layer):
    def __init__(self, coil_axis=-3):
        super().__init__()
        self.coil_axis = coil_axis

    def call(self, coilimg, smaps):
        return tf.reduce_sum(coilimg * tf.math.conj(smaps), self.coil_axis)

class MaskKspace(tf.keras.layers.Layer):
    def call(self, kspace, mask):
        return merlintf.complex_scale(kspace, mask)

class ForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False, channel_dim_defined=True):
        super().__init__()
        if center:
            self.fft2 = FFT2c()
        else:
            self.fft2 = FFT2()
        self.mask = MaskKspace()
        self.channel_dim_defined = channel_dim_defined

    def call(self, image, mask):
        if self.channel_dim_defined:
            kspace = self.fft2(image[...,0])
        else:
            kspace = self.fft2(image)
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace

class AdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False, channel_dim_defined=True):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = IFFT2c()
        else:
            self.ifft2 = IFFT2()
        self.channel_dim_defined = channel_dim_defined

    def call(self, kspace, mask):
        masked_kspace = self.mask(kspace, mask)
        img = self.ifft2(masked_kspace)
        if self.channel_dim_defined:
            return tf.expand_dims(img, -1)
        else:
            return img

class MulticoilForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        if center:
            self.fft2 = FFT2c()
        else:
            self.fft2 = FFT2()
        self.mask = MaskKspace()
        self.smaps = Smaps(coil_axis=coil_axis)
        self.channel_dim_defined = channel_dim_defined

    def call(self, image, mask, smaps):
        if self.channel_dim_defined:
            coilimg = self.smaps(image[...,0], smaps)
        else:
            coilimg = self.smaps(image, smaps)
        kspace = self.fft2(coilimg)
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace

class MulticoilAdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = IFFT2c()
        else:
            self.ifft2 = IFFT2()
        self.adj_smaps = SmapsAdj(coil_axis=coil_axis)
        self.channel_dim_defined = channel_dim_defined

    def call(self, kspace, mask, smaps):
        masked_kspace = self.mask(kspace, mask)
        coilimg = self.ifft2(masked_kspace)
        img = self.adj_smaps(coilimg, smaps)
        if self.channel_dim_defined:
            return tf.expand_dims(img, -1)
        else:
            return img

class TestMulticoilOps(unittest.TestCase):        
    def _get_data(self, shape, channel_dim_defined):
        batch, frames, ncoils, M, N = shape
        img = np.random.randn(*(batch, frames, M, N)) + 1j * np.random.randn(*(batch, frames, M, N))
        mask = np.ones((batch, 1, frames, 1, N))
        smaps = np.random.randn(*(batch, ncoils, 1, M, N)) + 1j * np.random.randn(*(batch, ncoils, 1, M, N))
        kspace = np.random.randn(*(batch, ncoils, frames, M, N)) + 1j * np.random.randn(*(batch, ncoils, frames, M, N))
        if channel_dim_defined:
            img = img[...,None]
        return img, mask, smaps, kspace

    def _test_forward(self, channel_dim_defined):
        shape = (2, 10, 15, 100, 100)
        img, mask, smaps, kspace = self._get_data(shape, channel_dim_defined)
        A = MulticoilForwardOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined)
        Ax = A(img, mask, smaps)
        self.assertTrue(Ax.shape == kspace.shape)

    def test_forward(self):
        self._test_forward(False)

    def test_forward_channel_dim(self):
        self._test_forward(True)

    def _test_backward(self, channel_dim_defined):
        shape = (2, 10, 15, 100, 100)
        img, mask, smaps, kspace = self._get_data(shape, channel_dim_defined)
        AH = MulticoilAdjointOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined)
        AHy = AH(kspace, mask, smaps)
        self.assertTrue(AHy.shape == img.shape)

    def test_backward(self):
        self._test_backward(False)

    def test_backward_channel_dim(self):
        self._test_backward(True)

    def _test_adjointness(self, channel_dim_defined):
        shape = (2, 10, 15, 100, 100)
        img, mask, smaps, kspace = self._get_data(shape, channel_dim_defined)
        A = MulticoilForwardOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined)
        AH = MulticoilAdjointOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined)
        Ax = A(img, mask, smaps,)
        AHy = AH(kspace, mask, smaps)

        lhs = np.sum(Ax * np.conj(kspace))
        rhs = np.sum(img * np.conj(AHy))
        self.assertAlmostEqual(lhs, rhs)

    def test_adjointness(self):
        self._test_adjointness(False)

    def test_adjointness_channel_dim(self):
        self._test_adjointness(True)

if __name__ == "__main__":
    unittest.test()