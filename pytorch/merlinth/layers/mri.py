import torch
from merlinth.layers.fft import fft2, fft2c, ifft2, ifft2c
from merlinth.layers.warp import WarpForward, WarpAdjoint
import unittest
import merlinth

#TODO add SoftSenseOps

# def adjointSoftSenseOpNoShift(th_kspace, th_smaps, th_mask):
#     th_img = torch.sum(complex_mult_conj(ifft2(th_kspace * th_mask), th_smaps), dim=(-5))
#     return th_img

# def forwardSoftSenseOpNoShift(th_img, th_smaps, th_mask):
#     th_img_pad = th_img.unsqueeze(-5)
#     th_kspace = fft2(complex_mult(th_img_pad.expand_as(th_smaps), th_smaps)) * th_mask
#     th_kspace = torch.sum(th_kspace, dim=-4, keepdim=True)
#     return th_kspace

# def adjointSoftSenseOp(th_kspace, th_smaps, th_mask):
#     th_img = torch.sum(complex_mult_conj(ifft2c(th_kspace * th_mask), th_smaps), dim=(-5))
#     return th_img

# def forwardSoftSenseOp(th_img, th_smaps, th_mask):
#     th_img_pad = th_img.unsqueeze(-5)
#     th_kspace = fft2c(complex_mult(th_img_pad.expand_as(th_smaps), th_smaps)) * th_mask
#     th_kspace = torch.sum(th_kspace, dim=-4, keepdim=True)
#     return th_kspace

class MulticoilForwardOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        if center:
            self.fft2 = fft2c
        else:
            self.fft2 = fft2
        self.coil_axis = coil_axis
        self.channel_dim_defined = channel_dim_defined

    def forward(self, image, mask, smaps):
        if self.channel_dim_defined:
            coilimg = torch.unsqueeze(image[:,0], self.coil_axis) * smaps
        else:
            coilimg = torch.unsqueeze(image, self.coil_axis) * smaps
        kspace = self.fft2(coilimg)
        masked_kspace = kspace * mask
        return masked_kspace

class MulticoilAdjointOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        if center:
            self.ifft2 = ifft2c
        else:
            self.ifft2 = ifft2
        self.coil_axis = coil_axis
        self.channel_dim_defined = channel_dim_defined

    def forward(self, kspace, mask, smaps):
        masked_kspace = kspace * mask
        coilimg = self.ifft2(masked_kspace)
        img = torch.sum(torch.conj(smaps) * coilimg, self.coil_axis)

        if self.channel_dim_defined:
            return torch.unsqueeze(img, 1)
        else:
            return img

class ForwardOp(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = fft2c
        else:
            self.fft2 = fft2

    def forward(self, image, mask):
        kspace = self.fft2(image)
        masked_kspace = kspace * mask
        return masked_kspace


class MulticoilMotionForwardOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        self.W = WarpForward()
        self.A = MulticoilForwardOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.channel_dim_defined = channel_dim_defined

    def forward(self, x, mask, smaps, u):
        if self.channel_dim_defined:
            x = self.W(x[:,0], u)
        else:
            x = self.W(x, u)
        y = self.A(x, mask, smaps)
        return y

class MulticoilMotionAdjointOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        self.AH = MulticoilAdjointOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.WH = WarpAdjoint()
        self.channel_dim_defined = channel_dim_defined

    def forward(self, y, mask, smaps, u):
        x = self.AH(y, mask, smaps)
        x = self.WH(x, u)
        if self.channel_dim_defined:
            return torch.unsqueeze(x, 1)
        else:
            return x

class AdjointOp(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.ifft2 = ifft2c
        else:
            self.ifft2 = ifft2

    def forward(self, kspace, mask):
        masked_kspace = kspace * mask
        img = self.ifft2(masked_kspace)
        return img

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