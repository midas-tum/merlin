import torch
import numpy as np
from merlinth.layers.pad import (
    real_pad2d,
    real_pad2d_transpose
)
import unittest

__all__ = ['PadConv2D', 'PadConvScale2D', 'PadConvScaleTranspose2D']

class PadConv2D(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, invariant=False,
                 stride=1, dilation=1, groups=1, bias=False, 
                 zero_mean=False, bound_norm=False, pad=True):
        super(PadConv2D, self).__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.invariant = invariant
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(filters)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.pad = pad

        # add the parameter
        if self.invariant:
            assert self.kernel_size == 3
            self.weight = torch.nn.Parameter(torch.empty(filters, in_channels, 1,  3))
            self.register_buffer('mask', torch.from_numpy(np.asarray([1,4,4], dtype=np.float32)[None, None, None, :]))
        else:
            self.weight = torch.nn.Parameter(torch.empty(filters, in_channels, self.kernel_size,  self.kernel_size))
            self.register_buffer('mask', torch.from_numpy(np.ones((self.kernel_size, self.kernel_size), dtype=np.float32)[None, None, :, :]))
        # insert them using a normal distribution
        torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1/np.prod(in_channels*kernel_size**2)))

        # specify reduction index
        self.weight.L_init = 1e+4
        if zero_mean or bound_norm:
            self.weight.reduction_dim = (1, 2, 3)
    
            # define a projection
            def l2_proj(surface=False):
                # reduce the mean
                if zero_mean:
                    mean = torch.sum(self.weight.data * self.mask, self.weight.reduction_dim, True) / (self.in_channels*self.kernel_size**2)
                    self.weight.data.sub_(mean)
                # normalize by the l2-norm
                if bound_norm:
                    norm = torch.sum(self.weight.data**2 * self.mask, self.weight.reduction_dim, True).sqrt_()
                    if surface:
                        self.weight.data.div_(norm)
                    else:
                        self.weight.data.div_(
                            torch.max(norm, torch.ones_like(norm)))
            self.weight.proj = l2_proj

            # initially call the projection
            self.weight.proj(True)

    def get_weight(self):
        if self.invariant:
            weight = torch.empty(self.filters, self.in_channels, self.kernel_size, self.kernel_size, device=self.weight.device)
            weight[:,:,1,1] = self.weight[:,:,0,0]
            weight[:,:,::2,::2] = self.weight[:,:,0,2].view(self.filters,self.in_channels,1,1)
            weight[:,:,1::2,::2] = self.weight[:,:,0,1].view(self.filters,self.in_channels,1,1)
            weight[:,:,::2,1::2] = self.weight[:,:,0,1].view(self.filters,self.in_channels,1,1)
        else:
            weight = self.weight
        return weight

    def _compute_optox_padding(self):
        pad = []
        for w in self.get_weight().shape[2:4][::-1]:
            pad += [w//2, w//2]
        return pad

    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # then pad
        pad = self._compute_optox_padding()
        if self.pad and any(pad) > 0:
            x = real_pad2d(x, pad)
        # compute the convolution
        x = torch.nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = (
                output_shape[2] - ((x.shape[2]-1)*self.stride+1),
                output_shape[3] - ((x.shape[3]-1)*self.stride+1)
            )
            # output_padding = (output_padding[0]//2 + self.kernel_size // 2, output_padding[1]//2 + self.kernel_size // 2)
        else:
            output_padding = 0

        # compute the convolution
        x = torch.nn.functional.conv_transpose2d(x, weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

        pad = self._compute_optox_padding()
        if self.pad and any(pad) > 0:
            x = real_pad2d_transpose(x, pad)
        return x

    def extra_repr(self):
        s = "({filters}, {in_channels}, {kernel_size}), invariant={invariant}"
        if self.stride != 1:
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.bias is None:
            s += ", bias=True"
        if self.zero_mean:
            s += ", zero_mean={zero_mean}"
        if self.bound_norm:
            s += ", bound_norm={bound_norm}"
        return s.format(**self.__dict__)


class PadConvScale2D(PadConv2D):
    def __init__(self, in_channels, filters, kernel_size=3, invariant=False,
                 groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(PadConvScale2D, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, 
            invariant=invariant, stride=stride, dilation=1, groups=groups, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

        # create the convolution kernel
        if self.stride > 1:
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            np_k = np.reshape(np_k, (1, 1, 5, 5))
            self.register_buffer('blur', torch.from_numpy(np_k))

    def get_weight(self):
        weight = super().get_weight()
        if self.stride > 1:
            weight = weight.reshape(-1, 1, self.kernel_size, self.kernel_size)
            for i in range(self.stride//2): 
                weight = torch.nn.functional.conv2d(weight, self.blur, padding=4)
            weight = weight.reshape(self.filters, self.in_channels, self.kernel_size+2*self.stride, self.kernel_size+2*self.stride)
        return weight


class PadConvScaleTranspose2D(PadConvScale2D):
    def __init__(self, in_channels, filters, kernel_size=3, invariant=False,
                 groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(PadConvScaleTranspose2D, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, 
            invariant=invariant, groups=groups, stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)

class PadConvScaleTranspose2dTest(unittest.TestCase):
    def test_conv_transpose2d_complex(self):
        nBatch = 5
        M = 320
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = PadConvScaleTranspose2D(nf_in, nf_out, kernel_size=3).cuda()
        
        x = torch.randn(nBatch, nf_in, M, N).cuda()
        Kx = model.backward(x)
        
        y = torch.randn(*Kx.shape).cuda()
        KHy = model.forward(y, output_shape=x.shape)

        rhs = torch.sum(Kx * y).detach().cpu().numpy()
        lhs = torch.sum(x * KHy).detach().cpu().numpy()

        self.assertTrue(rhs, lhs)

class PadConvScale2dTest(unittest.TestCase):
    def test_conv2d_complex(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32

        model = PadConvScale2D(nf_in, nf_out, kernel_size=3, stride=2).cuda()

        x = torch.randn(nBatch, nf_in, M, N).cuda()
        Kx = model(x)

        y = torch.randn(*Kx.shape).cuda()
        KHy = model.backward(y, output_shape=x.shape)

        rhs = torch.sum(Kx * y).detach().cpu().numpy()
        lhs = torch.sum(x * KHy).detach().cpu().numpy()

        self.assertTrue(rhs, lhs)