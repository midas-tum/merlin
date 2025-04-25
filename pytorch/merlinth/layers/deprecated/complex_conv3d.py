
import torch
import torch.nn.functional as F

import numpy as np

import unittest
from merlinth.layers.complex_init import *
from merlinth.layers import PadConv3D
from merlinth.utils import validate_input_dimension
from merlinth.layers.pad import (
    complex_pad3d,
    complex_pad3d_transpose
)
from merlinth.complex import complex_mult_conj

__all__ = [ 'ComplexPadConv3D',
            'ComplexPadConvScale3D',
            'ComplexPadConvScaleTranspose3D',
            'ComplexPadConvRealWeight3D',
            'ComplexPadConv2Dt']

class ComplexPadConvRealWeight3D(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3,
                 stride=1, dilation=1, groups=1, bias=False, 
                 zero_mean=False, bound_norm=False, pad=True):
        super(ComplexPadConvRealWeight3D, self).__init__()

        self.conv = Conv3D(in_channels, filters, kernel_size,
                 stride, dilation, groups, bias, zero_mean, bound_norm, pad)

    def forward(self, x):
        Kx_re = self.conv(x[...,0].contiguous())
        Kx_im = self.conv(x[...,1].contiguous())
        return torch.cat([Kx_re.unsqueeze_(-1), Kx_im.unsqueeze_(-1)], dim=-1)

    def backward(self, x, output_shape=None):
        KTx_re = self.conv.backward(x[...,0].contiguous(), output_shape=output_shape)
        KTx_im = self.conv.backward(x[...,1].contiguous(), output_shape=output_shape)
        return torch.cat([KTx_re.unsqueeze_(-1), KTx_im.unsqueeze_(-1)], dim=-1)

class ComplexPadConv3D(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 zero_mean=False, bound_norm=False, pad=True):
        super(ComplexPadConv3D, self).__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = validate_input_dimension('3D', kernel_size)
        self.stride = validate_input_dimension('3D', stride)
        self.dilation = validate_input_dimension('3D', dilation)
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(2, filters)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.pad = pad

        # add the parameter
        self.weight = torch.nn.Parameter(torch.empty(self.filters,
                                                     self.in_channels,
                                                     *kernel_size,
                                                     2))

        # insert them using a normal distribution
        #torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1/np.prod(in_channels*kernel_size_t*kernel_size_sp_y* kernel_size_sp_x)))
        complex_independent_filters_init(self.weight, 'glorot')

        # specify reduction index
        self.weight.L_init = 1e+4
        if bound_norm:
            self.weight.reduction_dim = (1, 2, 3, 4, 5)
        if zero_mean:
            self.weight.reduction_dim_mean = (1, 2, 3, 4)
        if zero_mean or bound_norm:    
            # define a projection
            def l2_proj(surface=False):
                # reduce the mean
                if zero_mean:
                    mean = torch.mean(self.weight.data, self.weight.reduction_dim_mean, True)
                    self.weight.data.sub_(mean)
                # normalize by the l2-norm
                if bound_norm:
                    norm = torch.sum(self.weight.data**2, self.weight.reduction_dim, True).sqrt_()
                    if surface:
                        self.weight.data.div_(norm)
                    else:
                        self.weight.data.div_(
                            torch.max(norm, torch.ones_like(norm)))
            self.weight.proj = l2_proj

            # initially call the projection
            self.weight.proj(True)
        # else:
        #     self.weight.reduction_dim = (5)

    def get_weight(self):
        weight = self.weight
        return weight

    def _compute_optox_padding(self):
        pad = []
        for w in self.get_weight().shape[2:5][::-1]:
            pad += [w//2, w//2]
        return pad

    def conv3d_forward(self, input, weight, bias):
        return F.conv3d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def conv3d_transpose(self, input, weight, bias, output_padding):
        return F.conv_transpose3d(input, weight, bias, self.stride,
                        self.padding, output_padding, self.groups, self.dilation)
                        
    def complex_conv3d_forward(self, input, weight, bias):
        x_re = input[...,0]
        x_im = input[...,1]


        k_re = weight[...,0]
        k_im = weight[...,1]

        if bias is not None:
            bias_re = bias[0]
            bias_im = bias[1]
        else:
            bias_re = None
            bias_im = None

        conv_rr = self.conv3d_forward(x_re, k_re, bias_re)
        conv_ii = self.conv3d_forward(x_im, k_im, bias_im)
        conv_ri = self.conv3d_forward(x_re, k_im, bias_im)
        conv_ir = self.conv3d_forward(x_im, k_re, bias_re)

        conv_re = conv_rr - conv_ii
        conv_im = conv_ir + conv_ri

        return torch.cat([conv_re.unsqueeze_(-1), conv_im.unsqueeze_(-1)], -1)

    def complex_conv3d_transpose(self, input, weight, bias, output_padding):
        x_re = input[...,0]
        x_im = input[...,1]

        k_re = weight[...,0]
        k_im = weight[...,1]

        if bias is not None:
            bias_re = bias[0]
            bias_im = bias[1]
        else:
            bias_re = None
            bias_im = None

        convT_rr = self.conv3d_transpose(x_re, k_re, bias_re, output_padding)
        convT_ii = self.conv3d_transpose(x_im, k_im, bias_im,output_padding)
        convT_ri = self.conv3d_transpose(x_re, k_im, bias_im, output_padding)
        convT_ir = self.conv3d_transpose(x_im, k_re, bias_re, output_padding)

        convT_re = convT_rr + convT_ii
        convT_im = convT_ir - convT_ri

        return torch.cat([convT_re.unsqueeze_(-1), convT_im.unsqueeze_(-1)], -1)

    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # then pad
        pad = self._compute_optox_padding()

        if self.pad and any(pad):
            x = complex_pad3d(x, pad)

        # compute the convolution
        x = self.complex_conv3d_forward(x, weight, self.bias)

        return x

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = (
                output_shape[-4] - ((x.shape[-4]-1)*self.stride[0]+1),
                output_shape[-3] - ((x.shape[-3]-1)*self.stride[1]+1),
                output_shape[-2] - ((x.shape[-2]-1)*self.stride[2]+1)
            )
        else:
            output_padding = 0

        # compute the transpose convolution
        x = self.complex_conv3d_transpose(x, weight, self.bias, output_padding)
        # transpose padding
        pad = self._compute_optox_padding()
        if self.pad and any(pad):
            x = complex_pad3d_transpose(x, pad)

        return x

    def extra_repr(self):
        s = "({filters}, {in_channels}, {kernel_size})"
        if any(self.stride) != 1:
            s += ", stride={stride}"
        if any(self.dilation) != 1:
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


class ComplexPadConvScale3D(ComplexPadConv3D):
    def __init__(self, in_channels, filters, kernel_size=3, groups=1, stride=(1, 2, 2), bias=False, zero_mean=False, bound_norm=False, pad=True):
        super(ComplexPadConvScale3D, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, stride=stride, dilation=1, groups=groups, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm, pad=pad)

        assert self.kernel_size[1] == self.kernel_size[2]
        assert self.kernel_size[1] > 1
        assert self.stride[1] == self.stride[2]
        assert self.stride[1] > 1

        # create the convolution kernel
        if self.stride[1]  > 1:
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            np_k = np.reshape(np_k, (1, 1, 5, 5))
            self.register_buffer('blur', torch.from_numpy(np_k))

    def get_weight(self):
        weight = super().get_weight()
        if self.stride[1] > 1:
            weight = weight.permute(0, 1, 5, 2, 3, 4)
            weight = weight.reshape(-1, 1, self.kernel_size[1], self.kernel_size[1])
            for i in range(self.stride[1]//2): 
                weight = torch.nn.functional.conv2d(weight, self.blur, padding=4)
            weight = weight.reshape(self.filters, self.in_channels, 2, self.kernel_size[0], self.kernel_size[1]+2*self.stride[1], self.kernel_size[1]+2*self.stride[1])
            weight = weight.permute(0, 1, 3, 4, 5, 2)
        return weight


class ComplexPadConvScaleTranspose3D(ComplexPadConvScale3D):
    def __init__(self, in_channels, filters, kernel_size=3, 
                 groups=1, stride=(1, 2, 2), bias=False, zero_mean=False, bound_norm=False, pad=True):
        super(ComplexPadConvScaleTranspose3D, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, groups=groups, stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm, pad=pad)

        self.bias = torch.nn.Parameter(torch.zeros(2, in_channels)) if bias else None

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)

class ComplexPadConv2Dt(torch.nn.Module):
    def __init__(self, in_channels, intermediate_filters, filters, kernel_size=3,
                 stride=1, dilation=1, groups=1, bias=False, zero_mean=True, bound_norm=True, pad=True):
        super(ComplexPadConv2Dt, self).__init__()

        if stride > 2:
            conv_module = ComplexPadConvScale3D
        else:
            conv_module = ComplexPadConv3D

        self.conv_xy = conv_module(in_channels,
                    intermediate_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    zero_mean=zero_mean,
                    bound_norm=bound_norm,
                    pad=pad)

        self.conv_t = ComplexPadConv3D(intermediate_filters,
                 filters,
                 kernel_size=kernel_size,
                 bias=bias,
                 zero_mean=False,
                 bound_norm=bound_norm,
                 pad=pad)

    def forward(self, x):
        x_sp = self.conv_xy(x)
        x_t = self.conv_t(x_sp)
        return x_t  

    def backward(self, x, output_shape=None):
        xT_t = self.conv_t.backward(x, output_shape)
        xT_sp = self.conv_xy.backward(xT_t, output_shape)
        return xT_sp

class ComplexPadConv3DTest(unittest.TestCase):
    def test_constraints(self):
        nBatch = 5
        M = 120
        N = 120
        D = 16
        nf_in = 10
        nf_out = 32

        model = ComplexPadConv3D(nf_in,
                            nf_out,
                            kernel_size=[3,5,5],
                            zero_mean=True,
                            bound_norm=True).cuda()

        np_weight = model.weight.data.detach().cpu().numpy()
        np_weight = np_weight[...,0] + 1j * np_weight[...,1]

        self.assertTrue(np.max(np.abs(np.mean(np_weight, axis=(1,2,3,4)))) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=(1,2,3,4)))
        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test(self, kernel_size):
        nBatch = 5
        M = 120
        N = 120
        D = 16
        nf_in = 10
        nf_out = 32
        
        model = ComplexPadConv3D(nf_in, nf_out, kernel_size=kernel_size).cuda()
        
        x = torch.randn(nBatch, nf_in, D, M, N, 2).cuda()
        Kx = model(x)
        
        y = torch.randn(*Kx.shape).cuda()
        KHy = model.backward(y)

        rhs = complex_mult_conj(Kx, y).view(-1, 2).sum(0).detach().cpu().numpy()
        lhs = complex_mult_conj(x, KHy).view(-1, 2).sum(0).detach().cpu().numpy()

        self.assertTrue(rhs[0]+1j*rhs[1], lhs[0] + 1j*lhs[1])

    def test1(self):
        self._test([3, 5, 5])

    def test2(self):
        self._test([5, 1, 3])

    def test3(self):
        self._test([1, 5, 3])

    def test4(self):
        self._test([1, 5, 5])

    def test5(self):
        self._test([3, 1, 1])

class ComplexPadConvScale3DTest(unittest.TestCase):
    def _test(self, kernel_size):
        nBatch = 5
        D = 16
        M = 100
        N = 100
        nf_in = 10
        nf_out = 32

        model = ComplexPadConvScale3D(nf_in, nf_out, kernel_size=kernel_size, stride=2).cuda()

        x = torch.randn(nBatch, nf_in, D, M, N, 2).cuda()
        Kx = model(x)

        y = torch.randn(*Kx.shape).cuda()
        KHy = model.backward(y, output_shape=x.shape)

        rhs = complex_mult_conj(Kx, y).view(-1, 2).sum(0).detach().cpu().numpy()
        lhs = complex_mult_conj(x, KHy).view(-1, 2).sum(0).detach().cpu().numpy()
        self.assertTrue(rhs[0]+1j*rhs[1], lhs[0] + 1j*lhs[1])
    
    def test1(self):
        self._test([3, 5, 5])

    def test2(self):
        self._test([1, 5, 5,])

class ComplexPadConv3DGradientTest(unittest.TestCase):
    def _test(self, kernel_size):
        nBatch = 1
        M = 10
        N = 10
        D = 5
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConv3D(nf_in, nf_out, kernel_size=kernel_size).cuda()
        
        npth_img = np.arange(0,nBatch*M*N*D*nf_in).reshape(nBatch,nf_in,D,M,N,1) + 1j * np.arange(0,nBatch*M*N*D*nf_in).reshape(nBatch,nf_in,D,M,N,1)

        x = torch.from_numpy(np.concatenate([np.real(npth_img), np.imag(npth_img)], -1)).to(torch.float32)

        #x = torch.randn(nBatch, nf_in, M, N, 2).cuda()

        x.requires_grad_(True)
        Kx = model.forward(x.cuda())

        loss = 0.5 * torch.sum(Kx ** 2)
        loss.backward()
        
        x_autograd = x.grad.detach().cpu().numpy()
        x_autograd = x_autograd[...,0] + 1j * x_autograd[...,1] 

        KHKx = model.backward(Kx, output_shape=x.shape)

        x_bwd = KHKx.detach().cpu().numpy()
        x_bwd = x_bwd[...,0] + 1j * x_bwd[...,1]

        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-3)

    def test1(self):
        self._test([5, 5, 3])

    def test2(self):
        self._test([5, 1, 3])

    def test3(self):
        self._test([1, 5, 3])

    def test4(self):
        self._test([5, 5, 1])

    def test5(self):
        self._test([1, 1, 3])

if __name__ == "__main__":
    unittest.main()

