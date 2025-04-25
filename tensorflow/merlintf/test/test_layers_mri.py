import unittest
import numpy as np
from merlintf.keras.layers.mri import (
    MulticoilForwardOp,
    MulticoilAdjointOp,
    MulticoilMotionForwardOp,
    MulticoilMotionAdjointOp
)

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


class TestMulticoilMotionOps(unittest.TestCase):
    #TODO test warping if features are present
    def _get_data(self, shape):
        batch, frames, frames_all, M, N, ncoils = shape
        img = np.random.randn(*(batch, frames, M, N)) + 1j * np.random.randn(*(batch, frames, M, N))
        uv = np.random.randn(*(batch, frames, frames_all, M, N, 2))
        mask = np.ones((batch, 1, 1, frames_all, 1, N))
        smaps = np.random.randn(*(batch, ncoils, 1, 1, M, N)) + 1j * np.random.randn(*(batch, ncoils, 1, 1, M, N))
        kspace = np.random.randn(*(batch, ncoils, frames, frames_all, M, N)) + 1j * np.random.randn(*(batch, ncoils, frames, frames_all, M, N))
        return img, uv, mask, smaps, kspace

    def _test_forward(self, channel_dim_defined):
        shape = (2, 4, 25, 176, 132, 15)
        img, uv, mask, smaps, kspace = self._get_data(shape)
        if channel_dim_defined:
            img = img[...,None]

        A = MulticoilMotionForwardOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined)
        Ax = A(img, mask, smaps, uv)
        self.assertTrue(Ax.shape == kspace.shape)

    def test_forward(self):
        self._test_forward(False)

    def test_forward_channel_dim(self):
        self._test_forward(True)
    
    def _test_backward(self, channel_dim_defined):
        shape = (2, 4, 25, 176, 132, 15)
        img, uv, mask, smaps, kspace = self._get_data(shape)
        if channel_dim_defined:
            img = img[...,None]
        AH = MulticoilMotionAdjointOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined)
        AHy = AH(kspace, mask, smaps, uv)
        self.assertTrue(AHy.shape == img.shape)

    def test_backward(self):
        self._test_backward(False)

    def test_backward_channel_dim(self):
        self._test_backward(True)

    def _test_adjointness(self, channel_dim_defined):
        shape = (2, 4, 25, 176, 132, 15)
        img, uv, mask, smaps, kspace = self._get_data(shape)
        if channel_dim_defined:
            img = img[...,None]

        A = MulticoilMotionForwardOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined)
        AH = MulticoilMotionAdjointOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined)
        Ax = A(img, mask, smaps, uv)
        AHy = AH(kspace, mask, smaps, uv)

        lhs = np.sum(Ax * np.conj(kspace))
        rhs = np.sum(img * np.conj(AHy))
        self.assertAlmostEqual(lhs, rhs)

    def test_adjointness(self):
        self._test_adjointness(False)

    def test_adjointness_channel_dim(self):
        self._test_adjointness(True)

if __name__ == "__main__":
    unittest.main()