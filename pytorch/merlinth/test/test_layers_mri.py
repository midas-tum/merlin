import unittest
import torch
import merlinth
from merlinth.layers.mri import (
    MulticoilForwardOp,
    MulticoilAdjointOp,
    MulticoilMotionForwardOp,
    MulticoilMotionAdjointOp
    )

class TestMulticoilOps(unittest.TestCase):        
    def _get_data(self, shape, channel_dim_defined, dtype=torch.get_default_dtype()):
        batch, frames, ncoils, M, N = shape
        img = merlinth.random_normal_complex((batch, frames, M, N), dtype)
        mask = torch.ones((batch, 1, frames, 1, N), dtype=dtype)
        smaps = merlinth.random_normal_complex((batch, ncoils, 1, M, N), dtype)
        kspace = merlinth.random_normal_complex((batch, ncoils, frames, M, N), dtype)
        if channel_dim_defined:
            img = img[:,None]
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
        img, mask, smaps, kspace = self._get_data(shape, channel_dim_defined, torch.double)
        A = MulticoilForwardOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined).double()
        AH = MulticoilAdjointOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined).double()
        Ax = A(img, mask, smaps,)
        AHy = AH(kspace, mask, smaps)

        lhs = torch.sum(Ax * torch.conj(kspace)).numpy()
        rhs = torch.sum(img * torch.conj(AHy)).numpy()
        self.assertAlmostEqual(lhs, rhs)

    def test_adjointness(self):
        self._test_adjointness(False)

    def test_adjointness_channel_dim(self):
        self._test_adjointness(True)


class TestMulticoilMotionOps(unittest.TestCase):
    #TODO test warping if features are present
    def _get_data(self, shape, channel_dim_defined, dtype=torch.double):
        batch, frames, frames_all, M, N, ncoils = shape
        img = merlinth.random_normal_complex((batch, frames, M, N), dtype=dtype)
        uv = torch.randn((batch, frames, frames_all, M, N, 2), dtype=dtype)
        mask = torch.ones((batch, 1, 1, frames_all, 1, N), dtype=dtype)
        smaps = merlinth.random_normal_complex((batch, ncoils, 1, 1, M, N), dtype=dtype)
        kspace = merlinth.random_normal_complex((batch, ncoils, frames, frames_all, M, N), dtype=dtype)
        if channel_dim_defined:
            img = img[:,None]
        return img.cuda(), uv.cuda(), mask.cuda(), smaps.cuda(), kspace.cuda()

    def _test_forward(self, channel_dim_defined):
        shape = (2, 4, 25, 176, 132, 15)
        img, uv, mask, smaps, kspace = self._get_data(shape, channel_dim_defined)

        A = MulticoilMotionForwardOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined).cuda()
        Ax = A(img, mask, smaps, uv)
        self.assertTrue(Ax.shape == kspace.shape)

    def test_forward(self):
        self._test_forward(False)

    def test_forward_channel_dim(self):
        self._test_forward(True)
    
    def _test_backward(self, channel_dim_defined):
        shape = (2, 4, 25, 176, 132, 15)
        img, uv, mask, smaps, kspace = self._get_data(shape, channel_dim_defined)

        AH = MulticoilMotionAdjointOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined).cuda()
        AHy = AH(kspace, mask, smaps, uv)
        self.assertTrue(AHy.shape == img.shape)

    def test_backward(self):
        self._test_backward(False)

    def test_backward_channel_dim(self):
        self._test_backward(True)

    def _test_adjointness(self, channel_dim_defined):
        shape = (2, 4, 25, 176, 132, 15)
        img, uv, mask, smaps, kspace = self._get_data(shape, channel_dim_defined)

        A = MulticoilMotionForwardOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined).cuda()
        AH = MulticoilMotionAdjointOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined).cuda()
        Ax = A(img, mask, smaps, uv)
        AHy = AH(kspace, mask, smaps, uv)

        lhs = torch.sum(Ax * torch.conj(kspace))
        rhs = torch.sum(img * torch.conj(AHy))
        self.assertAlmostEqual(lhs.cpu().numpy(), rhs.cpu().numpy())

    def test_adjointness(self):
        self._test_adjointness(False)

    def test_adjointness_channel_dim(self):
        self._test_adjointness(True)