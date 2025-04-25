import unittest
import torch
from merlinth.layers.warp import WarpForward, WarpAdjoint
import numpy as np

class TestWarping(unittest.TestCase):
    def _get_data(self, shape, is_complex):
        batch, frames, frames_all, M, N = shape
        cdtype = torch.complex128 if is_complex else torch.double
        img = torch.randn((batch, frames, M, N), dtype=cdtype)
        imgT = torch.randn((batch, frames, frames_all, M, N), dtype=cdtype)
        uv = torch.randn((batch, frames, frames_all, M, N, 2), dtype=torch.double)
        return img.cuda(), imgT.cuda(), uv.cuda()

    def _test_forward(self, is_complex):
        shape = (2, 4, 25, 176, 132)
        img, imgT, uv = self._get_data(shape, is_complex)
        W = WarpForward().cuda()
        Wx = W(img, uv)
        self.assertTrue(Wx.shape == imgT.shape)

    def test_forward(self):
        self._test_forward(False)

    def test_forward_complex(self):
        self._test_forward(True)

    def _test_backward(self, is_complex):
        shape = (2, 4, 25, 176, 132)
        img, imgT, uv = self._get_data(shape, is_complex)
        WH = WarpAdjoint().cuda()
        WHy = WH(imgT, uv)
        self.assertTrue(WHy.shape == img.shape)

    def test_backward(self):
        self._test_backward(False)

    def test_backward_complex(self):
        self._test_backward(True)

    def _test_adjointness(self, is_complex):
        shape = (2, 4, 25, 176, 132)
        img, imgT, uv = self._get_data(shape, is_complex)
        W = WarpForward().cuda()
        WH = WarpAdjoint().cuda()
        Wx = W(img, uv).cpu().numpy()
        WHy = WH(imgT, uv).cpu().numpy()
        imgT = imgT.cpu().numpy()
        img = img.cpu().numpy()
        lhs = np.sum(Wx * np.conj(imgT))
        rhs = np.sum(img * np.conj(WHy))
        self.assertAlmostEqual(lhs, rhs)

    def test_adjointness(self):
        self._test_adjointness(False)
    
    def test_adjointness_complex(self):
        self._test_adjointness(True)

    def _test_forward_grad(self, is_complex):
        shape = (2, 4, 25, 176, 132)
        img, _, uv = self._get_data(shape, is_complex)
        W = WarpForward().cuda()
        img.requires_grad_(True)
        uv.requires_grad_(True)
        Wx = W(img, uv)
        loss = torch.norm(Wx)**2
        loss.backward()

        self.assertTrue(img.grad != None)
        self.assertTrue(uv.grad != None)

    def test_forward_grad(self):
        self._test_forward_grad(False)
    
    def test_forward_grad_complex(self):
        self._test_forward_grad(True)

    def _test_backward_grad(self, is_complex):
        shape = (2, 4, 25, 176, 132)
        _, imgT, uv = self._get_data(shape, is_complex)
        WH = WarpAdjoint().cuda()
        imgT.requires_grad_(True)
        uv.requires_grad_(True)
        WHx = WH(imgT, uv)
        loss = torch.norm(WHx)**2
        loss.backward()

        self.assertTrue(imgT.grad != None)
        self.assertTrue(uv.grad != None)

    def test_backward_grad(self):
        self._test_backward_grad(False)
    
    def test_backward_grad_complex(self):
        self._test_backward_grad(True)

if __name__ == "__main__":
    unittest.main()