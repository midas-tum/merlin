from unittest.case import TestCase
import numpy as np
import torch
import tensorflow as tf
import unittest
import merlinth
import merlintf

def random_normal_complex(shape, dtype=np.float64):
    return np.random.randn(*shape) + 1j * np.random.randn(*shape)

class TestComplex(unittest.TestCase):
    def _get_complex_input(self, shape):
        x_np = random_normal_complex(shape)
        x_tf = tf.convert_to_tensor(x_np)
        x_th = torch.from_numpy(x_np)
        return x_tf, x_th

    def _get_real_input(self, shape):
        x_np = np.random.randn(*shape)
        x_tf = tf.convert_to_tensor(x_np)
        x_th = torch.from_numpy(x_np)
        return x_tf, x_th

    def test_complex_abs(self):
        x_tf, x_th = self._get_complex_input((1,2))
        y_tf = merlintf.complex_abs(x_tf).numpy()
        y_th = merlinth.complex_abs(x_th).numpy()
        self.assertTrue(np.allclose(y_th, y_tf))

    def test_complex_norm(self):
        x_tf, x_th = self._get_complex_input((1,2))
        y_tf = merlintf.complex_norm(x_tf).numpy()
        y_th = merlinth.complex_norm(x_th).numpy()
        self.assertTrue(np.allclose(y_th, y_tf))

    def test_complex_angle(self):
        x_tf, x_th = self._get_complex_input((1,2))
        y_tf = merlintf.complex_angle(x_tf).numpy()
        y_th = merlinth.complex_angle(x_th).numpy()
        self.assertTrue(np.allclose(y_th, y_tf))

    def test_complex_scale(self):
        x_tf, x_th = self._get_complex_input((1, 2))
        scale_tf, scale_th = self._get_real_input((1, 2))
        y_tf = merlintf.complex_scale(x_tf, scale_tf).numpy()
        y_th = merlinth.complex_scale(x_th, scale_th).numpy()
        self.assertTrue(np.allclose(y_th, y_tf))

    def test_complex_dot(self):
        x_tf, x_th = self._get_complex_input((1, 2))
        x2_tf, x2_th = self._get_complex_input((1, 2))
        y_tf = merlintf.complex_dot(x_tf, x2_tf).numpy()
        y_th = merlinth.complex_dot(x_th, x2_th).numpy()
        self.assertTrue(np.allclose(y_th, y_tf))

    def test_complex_dot_dim(self):
        x_tf, x_th = self._get_complex_input((5, 10, 10))
        x2_tf, x2_th = self._get_complex_input((5, 10, 10))
        y_tf = merlintf.complex_dot(x_tf, x2_tf, axis=(-2,-1)).numpy()
        y_th = merlinth.complex_dot(x_th, x2_th, dim=(-2,-1)).numpy()
        self.assertTrue(np.allclose(y_th, y_tf))

if __name__ == "__main__":
    unittest.test()