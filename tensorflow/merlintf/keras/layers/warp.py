import tensorflow as tf
import optotf.keras.warp
import numpy as np
import unittest

class WarpForward(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.W = optotf.keras.warp.Warp(channel_last=False)

    def call(self, x, u):
        # we assume that the input does not have any channel dimension
        # x [batch, frames, M, N]
        # u [batch, frames, frames_all, M, N, 2]
        out_shape = tf.shape(u)[:-1]
        M, N = tf.shape(u)[-3:-1]
        x = tf.repeat(tf.expand_dims(x, -3), repeats=tf.shape(u)[-4], axis=-3)
        x = tf.reshape(x, (-1, 1, M, N)) # [batch, frames * frames_all, 1, M, N]
        u = tf.reshape(u, (-1, M, N, 2)) # [batch, frames * frames_all, M, N, 2]
        Wx = self.W(x, u)
        return tf.reshape(Wx, out_shape)

class WarpAdjoint(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.WH = optotf.keras.warp.WarpTranspose(channel_last=False)

    def call(self, x, u):
        # we assume that the input does not have any channel dimension
        # x [batch, frames, frames_all, M, N]
        # u [batch, frames, frames_all, M, N, 2]
        out_shape = tf.shape(u)[:-1]
        M, N = tf.shape(u)[-3:-1]
        x = tf.reshape(x, (-1, 1, M, N)) # [batch * frames * frames_all, 1, M, N]
        u = tf.reshape(u, (-1, M, N, 2)) # [batch * frames * frames_all, M, N, 2]
        x_warpT = self.WH(x, u, x)
        x_warpT = tf.reshape(x_warpT, out_shape)
        x_warpT = tf.math.reduce_sum(x_warpT, -3)
        return x_warpT

    
class TestWarping(unittest.TestCase):
    def _get_data(self, shape, is_complex):
        tf.keras.backend.set_floatx('float64')
        batch, frames, frames_all, M, N = shape
        if is_complex:
            img = tf.complex(tf.random.normal((batch, frames, M, N)), tf.random.normal((batch, frames, M, N)))
            imgT = tf.complex(tf.random.normal((batch, frames, frames_all, M, N)), tf.random.normal((batch, frames, frames_all, M, N)))
            img = tf.cast(img, tf.complex128)
            imgT = tf.cast(imgT, tf.complex128)
        else:
            img = tf.random.normal((batch, frames, M, N))
            imgT = tf.random.normal((batch, frames, frames_all, M, N))
            img = tf.cast(img, tf.float64)
            imgT = tf.cast(imgT, tf.float64)
        uv = tf.random.normal((batch, frames, frames_all, M, N, 2))
        uv = tf.cast(uv, tf.float64)
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
        self.assertAlmostEqual(lhs, rhs)

    def test_adjointness(self):
        self._test_adjointness(False)
    
    def test_adjointness_complex(self):
        self._test_adjointness(True)

if __name__ == "__main__":
    unittest.test()