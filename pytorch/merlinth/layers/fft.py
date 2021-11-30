import torch
import unittest
import numpy as np

def fft2(x, dim=(-2,-1)):
    return torch.fft.fft2(x, dim=dim, norm='ortho')

def ifft2(X, dim=(-2,-1)):
    return torch.fft.ifft2(X, dim=dim, norm='ortho')

def fft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(fft2(torch.fft.ifftshift(x, dim), dim), dim)

def ifft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(ifft2(torch.fft.ifftshift(x, dim), dim), dim)

class TestFFT(unittest.TestCase):
    def testFFT2(self):
        x = np.random.randn(5, 2, 11, 11) + 1j * np.random.randn(5, 2, 11, 11)
        X_np = np.fft.fft2(x, norm='ortho')
        X_th = fft2(torch.from_numpy(x)).numpy()
        self.assertTrue(np.allclose(X_np, X_th))

    def testFFT2c(self):
        x = np.random.randn(5, 2, 11, 11) + 1j * np.random.randn(5, 2, 11, 11)
        X_np = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, (-2,-1)), norm='ortho'), (-2,-1))
        X_th = fft2c(torch.from_numpy(x)).numpy()
        self.assertTrue(np.allclose(X_np, X_th))

    def testIFFT2(self):
        X = np.random.randn(5, 2, 11, 11) + 1j * np.random.randn(5, 2, 11, 11)
        x_np = np.fft.ifft2(X, norm='ortho')
        x_th = ifft2(torch.from_numpy(X)).numpy()
        self.assertTrue(np.allclose(x_np, x_th))

    def testIFFT2c(self):
        X = np.random.randn(5, 2, 11, 11) + 1j * np.random.randn(5, 2, 11, 11)
        x_np = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X, (-2,-1)), norm='ortho'), (-2,-1))
        x_th = ifft2c(torch.from_numpy(X)).numpy()
        self.assertTrue(np.allclose(x_np, x_th))