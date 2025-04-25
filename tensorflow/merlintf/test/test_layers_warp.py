import unittest
import numpy as np
import tensorflow as tf
#tf.keras.backend.set_floatx('float64')

from merlintf.keras.layers.warp import (
    WarpForward,
    WarpAdjoint
)
import merlintf

class TestWarping(unittest.TestCase):
    def _get_data(self, shape, is_complex):
        batch, frames, frames_all, M, N = shape
        if is_complex:
            img = merlintf.random_normal_complex((batch, frames, M, N), dtype=tf.keras.backend.floatx())
            imgT = merlintf.random_normal_complex((batch, frames, frames_all, M, N), dtype=tf.keras.backend.floatx())
        else:
            img = tf.random.normal((batch, frames, M, N), dtype=tf.keras.backend.floatx())
            imgT = tf.random.normal((batch, frames, frames_all, M, N),  dtype=tf.keras.backend.floatx())
        uv = tf.random.normal((batch, frames, frames_all, M, N, 2),  dtype=tf.keras.backend.floatx())
        return img, imgT, uv

    def _test_forward(self, is_complex):
        shape = (2, 4, 25, 176, 132)
        img, imgT, uv = self._get_data(shape, is_complex)
        W = WarpForward()
        Wx = W(img, uv)
        self.assertTrue(Wx.shape == imgT.shape)

    def test_forward(self):
        self._test_forward(False)

    def test_forward_complex(self):
        self._test_forward(True)

    def _test_backward(self, is_complex):
        shape = (2, 4, 25, 176, 132)
        img, imgT, uv = self._get_data(shape, is_complex)
        WH = WarpAdjoint()
        WHy = WH(imgT, uv)
        self.assertTrue(WHy.shape == img.shape)

    def test_backward(self):
        self._test_backward(False)

    def test_backward_complex(self):
        self._test_backward(True)

    def _test_adjointness(self, is_complex):
        shape = (2, 4, 25, 176, 132)
        img, imgT, uv = self._get_data(shape, is_complex)
        W = WarpForward()
        WH = WarpAdjoint()
        Wx = W(img, uv)
        WHy = WH(imgT, uv)

        lhs = np.sum(Wx * np.conj(imgT))
        rhs = np.sum(img * np.conj(WHy))
        self.assertTrue(np.sum(np.abs(lhs - rhs)) < 1)
        #self.assertAlmostEqual(lhs, rhs)

    def test_adjointness(self):
        self._test_adjointness(False)
    
    def test_adjointness_complex(self):
        self._test_adjointness(True)

if __name__ == "__main__":
    unittest.main()