
import torch
import torch.nn.functional as F
import optoth.pad3d

import numpy as np

import unittest
from merlinth import mytorch
from .complex_init import *

__all__ = [ 'ComplexConv3d',
            'ComplexConvScale3d',
            'ComplexConvScaleTranspose3d',
            'ComplexConvRealWeight3d',
            'ComplexConv2dt']

class ComplexConvRealWeight3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_sp_x=3, kernel_size_sp_y=3, kernel_size_t=3,
                 stride=1, dilation=1, groups=1, bias=False, 
                 zero_mean=False, bound_norm=False):
        super(ComplexConvRealWeight3d, self).__init__()

        self.conv = Conv3d(in_channels, out_channels, kernel_size_sp_x, kernel_size_sp_y, kernel_size_t,
                 stride, dilation, groups, bias, zero_mean, bound_norm)

    def forward(self, x):
        Kx_re = self.conv(x[...,0].contiguous())
        Kx_im = self.conv(x[...,1].contiguous())
        return torch.cat([Kx_re.unsqueeze_(-1), Kx_im.unsqueeze_(-1)], dim=-1)

    def backward(self, x, output_shape=None):
        KTx_re = self.conv.backward(x[...,0].contiguous(), output_shape=output_shape)
        KTx_im = self.conv.backward(x[...,1].contiguous(), output_shape=output_shape)
        return torch.cat([KTx_re.unsqueeze_(-1), KTx_im.unsqueeze_(-1)], dim=-1)

class Conv3d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size_sp_x=3,
                 kernel_size_sp_y=3,
                 kernel_size_t=3,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 zero_mean=False, bound_norm=False):
        super(Conv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_sp_x = kernel_size_sp_x
        self.kernel_size_sp_y = kernel_size_sp_y
        self.kernel_size_t = kernel_size_t
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0

        # add the parameter
        self.weight = torch.nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.kernel_size_t, self.kernel_size_sp_y,  self.kernel_size_sp_x))

        # insert them using a normal distribution
        torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1/np.prod(in_channels*kernel_size_t*kernel_size_sp_y* kernel_size_sp_x)))

        # specify reduction index
        self.weight.L_init = 1e+4
        if zero_mean or bound_norm:    
            self.weight.reduction_dim = (1, 2, 3, 4)
            self.weight.reduction_dim_mean = (1, 2, 3, 4)
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
    
    def forward(self, x):
        # construct the kernel
        weight = self.weight

        pad_sp_x = weight.shape[-1]//2
        pad_sp_y = weight.shape[-2]//2
        pad_t = weight.shape[-3]//2

        if pad_sp_x > 0 or pad_sp_y > 0 or pad_t > 0:
            x = optoth.pad3d.pad3d(x, (pad_sp_x,pad_sp_x,pad_sp_y,pad_sp_y,pad_t,pad_t), mode='symmetric')

        # compute the convolution
        return torch.nn.functional.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.weight

        # determine the output padding
        if not output_shape is None:
            output_padding = (
                output_shape[-4] - ((x.shape[-4]-1)*1+1),
                output_shape[-3] - ((x.shape[-3]-1)*self.stride+1),
                output_shape[-2] - ((x.shape[-2]-1)*self.stride+1)
            )
        else:
            output_padding = 0

        # compute the transpose convolution
        x = torch.nn.functional.conv_transpose3d(x, weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        
        # transpose padding
        pad_sp_x = weight.shape[-1]//2
        pad_sp_y = weight.shape[-2]//2
        pad_t = weight.shape[-3]//2

        # compute the convolution
        if pad_sp_x > 0 or pad_sp_y > 0 or pad_t > 0:
            x = optoth.pad3d.pad3d_transpose(x, (pad_sp_x,pad_sp_x,pad_sp_y,pad_sp_y,pad_t,pad_t), mode='symmetric')
        return x

    def extra_repr(self):
        s = "({out_channels}, {in_channels}, {kernel_size_t},  {kernel_size_sp_y},  {kernel_size_sp_x}),"
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

class ComplexConv3d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size_sp_x=3,
                 kernel_size_sp_y=3,
                 kernel_size_t=3,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 zero_mean=False, bound_norm=False):
        super(ComplexConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_sp_x = kernel_size_sp_x
        self.kernel_size_sp_y = kernel_size_sp_y
        self.kernel_size_t = kernel_size_t
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(2, out_channels)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0

        # add the parameter
        self.weight = torch.nn.Parameter(torch.empty(self.out_channels,
                                                     self.in_channels,
                                                     self.kernel_size_t,
                                                     self.kernel_size_sp_y,
                                                     self.kernel_size_sp_x,
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

    def conv3d_forward(self, input, weight, bias):
        return F.conv3d(input, weight, bias, (1, self.stride, self.stride),
                        self.padding, self.dilation, self.groups)

    def conv3d_transpose(self, input, weight, bias, output_padding):
        return F.conv_transpose3d(input, weight, bias, (1, self.stride, self.stride),
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

    def complex_pad3d(self, x, pad_sp_x, pad_sp_y, pad_t):
        xp_re = optoth.pad3d.pad3d(x[...,0].contiguous(), (pad_sp_x,pad_sp_x,pad_sp_y,pad_sp_y,pad_t,pad_t), mode='symmetric')
        xp_im = optoth.pad3d.pad3d(x[...,1].contiguous(), (pad_sp_x,pad_sp_x,pad_sp_y,pad_sp_y,pad_t,pad_t), mode='symmetric')

        new_shape = list(xp_re.shape)
        new_shape.append(2)
        xp = torch.zeros(*new_shape, device=x.device, dtype=x.dtype)
        xp[...,0] = xp_re
        xp[...,1] = xp_im
    
        return xp

    def complex_pad3d_transpose(self, x, pad_sp_x, pad_sp_y, pad_t):
        xp_re = optoth.pad3d.pad3d_transpose(x[...,0].contiguous(), (pad_sp_x,pad_sp_x,pad_sp_y,pad_sp_y,pad_t,pad_t), mode='symmetric')
        xp_im = optoth.pad3d.pad3d_transpose(x[...,1].contiguous(), (pad_sp_x,pad_sp_x,pad_sp_y,pad_sp_y,pad_t,pad_t), mode='symmetric')

        new_shape = list(xp_re.shape)
        new_shape.append(2)
        xp = torch.zeros(*new_shape, device=x.device, dtype=x.dtype)
        xp[...,0] = xp_re
        xp[...,1] = xp_im

        return xp

    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # then pad
        pad_sp_x = weight.shape[-2]//2
        pad_sp_y = weight.shape[-3]//2
        pad_t = weight.shape[-4]//2

        if pad_sp_x > 0 or pad_sp_y > 0 or pad_t > 0:
            x = self.complex_pad3d(x, pad_sp_x, pad_sp_y, pad_t)

        # compute the convolution
        x = self.complex_conv3d_forward(x, weight, self.bias)

        return x

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = (
                output_shape[-4] - ((x.shape[-4]-1)*1+1),
                output_shape[-3] - ((x.shape[-3]-1)*self.stride+1),
                output_shape[-2] - ((x.shape[-2]-1)*self.stride+1)
            )
        else:
            output_padding = 0

        # compute the transpose convolution
        x = self.complex_conv3d_transpose(x, weight, self.bias, output_padding)
        # transpose padding
        pad_sp_x = weight.shape[-2]//2
        pad_sp_y = weight.shape[-3]//2
        pad_t = weight.shape[-4]//2
        if pad_sp_x > 0 or pad_sp_y > 0 or pad_t > 0:
            x = self.complex_pad3d_transpose(x, pad_sp_x, pad_sp_y, pad_t)

        return x

    def extra_repr(self):
        s = "({out_channels}, {in_channels}, {kernel_size_t}, {kernel_size_sp_y}, {kernel_size_sp_x})"
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


class ComplexConvScale3d(ComplexConv3d):
    def __init__(self, in_channels, out_channels, kernel_size_sp_x=3, kernel_size_sp_y=3, kernel_size_t=3,
                 groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(ComplexConvScale3d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size_sp_x=kernel_size_sp_x, kernel_size_sp_y=kernel_size_sp_y,
            kernel_size_t=kernel_size_t, stride=stride, dilation=1, groups=groups, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)
        assert kernel_size_sp_x == kernel_size_sp_y
        assert kernel_size_sp_x > 1
        self.kernel_size_sp = kernel_size_sp_x

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
            weight = weight.permute(0, 1, 5, 2, 3, 4)
            weight = weight.reshape(-1, 1, self.kernel_size_sp, self.kernel_size_sp)
            for i in range(self.stride//2): 
                weight = torch.nn.functional.conv2d(weight, self.blur, padding=4)
            weight = weight.reshape(self.out_channels, self.in_channels, 2, self.kernel_size_t, self.kernel_size_sp+2*self.stride, self.kernel_size_sp+2*self.stride)
            weight = weight.permute(0, 1, 3, 4, 5, 2)
        return weight


class ComplexConvScaleTranspose3d(ComplexConvScale3d):
    def __init__(self, in_channels, out_channels, kernel_size_sp_x=3, kernel_size_sp_y=3, kernel_size_t=3, 
                 groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(ComplexConvScaleTranspose3d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size_sp_x=kernel_size_sp_x, kernel_size_sp_y=kernel_size_sp_y,
            kernel_size_t=kernel_size_t, groups=groups, stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)
        assert kernel_size_sp_x == kernel_size_sp_y
        assert kernel_size_sp_x > 1
        self.kernel_size_sp = kernel_size_sp_x
        self.bias = torch.nn.Parameter(torch.zeros(2, in_channels)) if bias else None

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)

class ComplexConv2dt(torch.nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, kernel_size_sp_x=3, kernel_size_sp_y=3, kernel_size_t=3,
                 stride=1, dilation=1, groups=1, bias=False, zero_mean=True, bound_norm=True):
        super(ComplexConv2dt, self).__init__()

        if stride > 2:
            conv_module = ComplexConvScale3d
        else:
            conv_module = ComplexConv3d

        self.conv_xy = conv_module(in_channels,
                    inter_channels,
                    kernel_size_sp_x=kernel_size_sp_x,
                    kernel_size_sp_y=kernel_size_sp_y,
                    kernel_size_t=1,
                    stride=stride,
                    bias=bias,
                    zero_mean=zero_mean,
                    bound_norm=bound_norm)

        self.conv_t = ComplexConv3d(inter_channels,
                 out_channels,
                 kernel_size_sp_x=1,
                 kernel_size_sp_y=1,
                 kernel_size_t=kernel_size_t,
                 bias=bias,
                 zero_mean=False,
                 bound_norm=bound_norm)

    def forward(self, x):
        x_sp = self.conv_xy(x)
        x_t = self.conv_t(x_sp)
        return x_t  

    def backward(self, x, output_shape=None):
        xT_t = self.conv_t.backward(x, output_shape)
        xT_sp = self.conv_xy.backward(xT_t, output_shape)
        return xT_sp

class ComplexConv3dTest(unittest.TestCase):
    def test_constraints(self):
        nBatch = 5
        M = 120
        N = 120
        D = 16
        nf_in = 10
        nf_out = 32

        model = ComplexConv3d(nf_in,
                            nf_out,
                            kernel_size_sp_x=5,
                            kernel_size_sp_y=5,
                            kernel_size_t=3,
                            zero_mean=True,
                            bound_norm=True).cuda()

        np_weight = model.weight.data.detach().cpu().numpy()
        np_weight = np_weight[...,0] + 1j * np_weight[...,1]

        self.assertTrue(np.max(np.abs(np.mean(np_weight, axis=(1,2,3,4)))) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=(1,2,3,4)))
        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test(self, ksx, ksy, kst):
        nBatch = 5
        M = 120
        N = 120
        D = 16
        nf_in = 10
        nf_out = 32
        
        model = ComplexConv3d(nf_in, nf_out, kernel_size_sp_x=ksx, kernel_size_sp_y=ksy, kernel_size_t=kst).cuda()
        
        x = torch.randn(nBatch, nf_in, D, M, N, 2).cuda()
        Kx = model(x)
        
        y = torch.randn(*Kx.shape).cuda()
        KHy = model.backward(y)

        rhs = mytorch.complex.complex_mult_conj(Kx, y).view(-1, 2).sum(0).detach().cpu().numpy()
        lhs = mytorch.complex.complex_mult_conj(x, KHy).view(-1, 2).sum(0).detach().cpu().numpy()

        self.assertTrue(rhs[0]+1j*rhs[1], lhs[0] + 1j*lhs[1])

    def test1(self):
        self._test(5, 5, 3)

    def test2(self):
        self._test(5, 1, 3)

    def test3(self):
        self._test(1, 5, 3)

    def test4(self):
        self._test(5, 5, 1)

    def test5(self):
        self._test(1, 1, 3)

class ComplexConvScale3dTest(unittest.TestCase):
    def _test(self, ksx, ksy, kst):
        nBatch = 5
        D = 16
        M = 100
        N = 100
        nf_in = 10
        nf_out = 32

        model = ComplexConvScale3d(nf_in, nf_out, kernel_size_sp_x=ksx, kernel_size_sp_y=ksy, kernel_size_t=kst, stride=2).cuda()

        x = torch.randn(nBatch, nf_in, D, M, N, 2).cuda()
        Kx = model(x)

        y = torch.randn(*Kx.shape).cuda()
        KHy = model.backward(y, output_shape=x.shape)

        rhs = mytorch.complex.complex_mult_conj(Kx, y).view(-1, 2).sum(0).detach().cpu().numpy()
        lhs = mytorch.complex.complex_mult_conj(x, KHy).view(-1, 2).sum(0).detach().cpu().numpy()
        self.assertTrue(rhs[0]+1j*rhs[1], lhs[0] + 1j*lhs[1])
    
    def test1(self):
        self._test(5, 5, 3)

    def test2(self):
        self._test(5, 5, 1)

class ComplexConv3dGradientTest(unittest.TestCase):
    def _test(self, ksx, ksy, kst):
        nBatch = 1
        M = 10
        N = 10
        D = 5
        nf_in = 1
        nf_out = 32
        
        model = ComplexConv3d(nf_in, nf_out, kernel_size_sp_x=ksx, kernel_size_sp_y=ksy, kernel_size_t=kst).cuda()
        
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
        self._test(5, 5, 3)

    def test2(self):
        self._test(5, 1, 3)

    def test3(self):
        self._test(1, 5, 3)

    def test4(self):
        self._test(5, 5, 1)

    def test5(self):
        self._test(1, 1, 3)

if __name__ == "__main__":
    unittest.test()

