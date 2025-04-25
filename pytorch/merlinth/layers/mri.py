import torch
from merlinth.layers.fft import fft2, fft2c, ifft2, ifft2c
try:
    from merlinth.layers.warp import WarpForward, WarpAdjoint
except:
    pass

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

