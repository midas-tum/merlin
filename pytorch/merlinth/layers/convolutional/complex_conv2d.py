
import torch
import torch.nn.functional as F
from merlinth.layers import PadConv2D
from merlinth.layers.complex_init import *
from merlinth.layers.pad import (
    real_pad2d,
    real_pad2d_transpose,
    complex_pad2d,
    complex_pad2d_transpose
)
from merlinth.complex import complex_mult_conj
import numpy as np

import unittest
import sys

__all__ = ['ComplexConv2D',
           'ComplexPadConv2D',
           'ComplexPadConvRealWeight2D',
           'ComplexPadConvScale2D',
           'ComplexPadConvScaleTranspose2D',
           'PseudoComplexPadConv2D']

class ComplexConv2D(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode='zeros'):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.filters = filters
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.bias = torch.nn.Parameter(torch.zeros(2, filters)) if bias else None
        self.weight = torch.nn.Parameter(torch.empty(self.filters, self.in_channels, self.kernel_size,  self.kernel_size, 2))
        # insert them using a normal distribution
        #torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1/np.prod(in_channels*kernel_size**2)))
        complex_independent_filters_init(self.weight, 'glorot')
        #torch.nn.init.xavier_uniform_(self.weight, gain=1.0)

    def forward(self, x):
        # compute the convolution
        x = self.complex_conv2d_forward(x, self.weight, self.bias)
        return x
    
    def conv2d_forward(self, input, weight, bias):
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def conv2d_transpose(self, input, weight, bias, output_padding):
        return F.conv_transpose2d(input, weight, bias, self.stride,
                        self.padding, output_padding, self.groups, self.dilation)
                        
    def complex_conv2d_forward(self, input, weight, bias):        
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

        conv_rr = self.conv2d_forward(x_re, k_re, bias_re)
        conv_ii = self.conv2d_forward(x_im, k_im, bias_im)
        conv_ri = self.conv2d_forward(x_re, k_im, bias_im)
        conv_ir = self.conv2d_forward(x_im, k_re, bias_re)

        conv_re = conv_rr - conv_ii
        conv_im = conv_ir + conv_ri

        return torch.cat([conv_re.unsqueeze_(-1), conv_im.unsqueeze_(-1)], -1)

    def complex_conv2d_transpose(self, input, weight, bias, output_padding):
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

        convT_rr = self.conv2d_transpose(x_re, k_re, bias_re, output_padding)
        convT_ii = self.conv2d_transpose(x_im, k_im, bias_im, output_padding)
        convT_ri = self.conv2d_transpose(x_re, k_im, bias_im, output_padding)
        convT_ir = self.conv2d_transpose(x_im, k_re, bias_re, output_padding)

        convT_re = convT_rr + convT_ii
        convT_im = convT_ir - convT_ri

        return torch.cat([convT_re.unsqueeze_(-1), convT_im.unsqueeze_(-1)], -1)

class ComplexPadConvRealWeight2D(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, invariant=False,
                 stride=1, dilation=1, groups=1, bias=False, 
                 zero_mean=False, bound_norm=False):
        super().__init__()

        self.conv = PadConv2d(in_channels, filters, kernel_size, invariant,
                 stride, dilation, groups, bias, zero_mean, bound_norm)

    def forward(self, x):
        Kx_re = self.conv(x[...,0].contiguous())
        Kx_im = self.conv(x[...,1].contiguous())
        return torch.cat([Kx_re.unsqueeze_(-1), Kx_im.unsqueeze_(-1)], dim=-1)

    def backward(self, x, output_shape=None):
        KTx_re = self.conv.backward(x[...,0].contiguous(), output_shape=output_shape)
        KTx_im = self.conv.backward(x[...,1].contiguous(), output_shape=output_shape)
        return torch.cat([KTx_re.unsqueeze_(-1), KTx_im.unsqueeze_(-1)], dim=-1)

class ComplexPadConv2D(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, invariant=False,
                 stride=1, dilation=1, groups=1, bias=False, 
                 zero_mean=False, bound_norm=False, pad=True):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.invariant = invariant
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(2, filters)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.pad = pad

        # add the parameter
        if self.invariant:
            assert self.kernel_size == 3
            self.weight = torch.nn.Parameter(torch.empty(self.filters, self.in_channels, 1,  3, 2))
            self.register_buffer('mask', torch.from_numpy(np.asarray([1,4,4], dtype=np.float32)[None, None, None, :, None]))
        else:
            self.weight = torch.nn.Parameter(torch.empty(self.filters, self.in_channels, self.kernel_size,  self.kernel_size, 2))
            self.register_buffer('mask', torch.from_numpy(np.ones((self.kernel_size, self.kernel_size), dtype=np.float32)[None, None, :, :, None]))
        # insert them using a normal distribution
        #torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1/np.prod(in_channels*kernel_size**2)))
        complex_independent_filters_init(self.weight, 'glorot')

        # specify reduction index
        self.weight.L_init = 1e+4
        if zero_mean or bound_norm:
            self.weight.reduction_dim = (1, 2, 3, 4)
            self.weight.reduction_dim_mean = (1, 2, 3)
    
            # define a projection
            def l2_proj(surface=False):
                # reduce the mean
                if zero_mean:
                    mean = torch.sum(self.weight.data * self.mask, self.weight.reduction_dim_mean, True) / (self.kernel_size**2)
                    self.weight.data.sub_(mean)
                # normalize by the l2-norm
                if bound_norm:
                    norm = torch.sum(self.weight.data**2 * self.mask, self.weight.reduction_dim, True).sqrt_()
                    if surface:
                        self.weight.data.div_(
                            torch.max(norm, torch.ones_like(norm)*1e-9))
                    else:
                        self.weight.data.div_(
                            torch.max(norm, torch.ones_like(norm)))
            self.weight.proj = l2_proj

            # initially call the projection
            self.weight.proj(True)
        else:
            self.weight.reduction_dim = (4)

    def get_weight(self):
        if self.invariant:
            weight = torch.empty(self.filters, self.in_channels, self.kernel_size, self.kernel_size, 2, device=self.weight.device)
            weight[:,:,1,1,:] = self.weight[:,:,0,0,:]
            weight[:,:,::2,::2,:] = self.weight[:,:,0,2,:].view(self.filters,self.in_channels,1,1,2)
            weight[:,:,1::2,::2,:] = self.weight[:,:,0,1,:].view(self.filters,self.in_channels,1,1,2)
            weight[:,:,::2,1::2,:] = self.weight[:,:,0,1,:].view(self.filters,self.in_channels,1,1,2)
        else:
            weight = self.weight
        return weight

    def _compute_optox_padding(self):
        pad = []
        for w in self.get_weight().shape[2:4][::-1]:
            pad += [w//2, w//2]
        return pad

    def conv2d_forward(self, input, weight, bias):
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def conv2d_transpose(self, input, weight, bias, output_padding):
        return F.conv_transpose2d(input, weight, bias, self.stride,
                        self.padding, output_padding, self.groups, self.dilation)
                        
    def complex_conv2d_forward(self, input, weight, bias):        
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

        conv_rr = self.conv2d_forward(x_re, k_re, bias_re)
        conv_ii = self.conv2d_forward(x_im, k_im, bias_im)
        conv_ri = self.conv2d_forward(x_re, k_im, bias_im)
        conv_ir = self.conv2d_forward(x_im, k_re, bias_re)

        conv_re = conv_rr - conv_ii
        conv_im = conv_ir + conv_ri

        return torch.cat([conv_re.unsqueeze_(-1), conv_im.unsqueeze_(-1)], -1)

    def complex_conv2d_transpose(self, input, weight, bias, output_padding):
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

        convT_rr = self.conv2d_transpose(x_re, k_re, bias_re, output_padding)
        convT_ii = self.conv2d_transpose(x_im, k_im, bias_im, output_padding)
        convT_ri = self.conv2d_transpose(x_re, k_im, bias_im, output_padding)
        convT_ir = self.conv2d_transpose(x_im, k_re, bias_re, output_padding)

        convT_re = convT_rr + convT_ii
        convT_im = convT_ir - convT_ri

        return torch.cat([convT_re.unsqueeze_(-1), convT_im.unsqueeze_(-1)], -1)



    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # then pad
        pad = self._compute_optox_padding()
        if any(pad) > 0:
            x = complex_pad2d(x, pad)
        # compute the convolution
        x = self.complex_conv2d_forward(x, weight, self.bias)
        return x

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = (
                (output_shape[2] - ((x.shape[2]-1)*self.stride+1)),
                (output_shape[3] - ((x.shape[3]-1)*self.stride+1))
            )
        else:
            output_padding = 0

        # compute the convolution
        x = self.complex_conv2d_transpose(x, weight, self.bias, output_padding)
        pad = self._compute_optox_padding()

        if self.pad and any(pad) > 0:
            x = complex_pad2d_transpose(x, pad)

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

class PseudoComplexPadConv2D(ComplexPadConv2D):
    def get_weight(self):
        if self.invariant:
            weight = torch.empty(self.filters, self.in_channels, self.kernel_size, self.kernel_size, 2, device=self.weight.device)
            weight[:,:,1,1,:] = self.weight[:,:,0,0,:]
            weight[:,:,::2,::2,:] = self.weight[:,:,0,2,:].view(self.filters,self.in_channels,1,1,2)
            weight[:,:,1::2,::2,:] = self.weight[:,:,0,1,:].view(self.filters,self.in_channels,1,1,2)
            weight[:,:,::2,1::2,:] = self.weight[:,:,0,1,:].view(self.filters,self.in_channels,1,1,2)
        else:
            weight = self.weight
        weight = weight.permute(0,1,4,2,3)
        weight = weight.view(weight.shape[0], weight.shape[1]*weight.shape[2], weight.shape[3], weight.shape[4])
        return weight

    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # then pad
        pad = self._compute_optox_padding()
        if any(pad) > 0:
            x = real_pad2d(x, pad)

        # compute the convolution
        return torch.nn.functional.conv2d(x,
               weight,
               self.bias,
               self.stride,
               self.padding,
               self.dilation,
               self.groups)

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = (
                output_shape[2] - ((x.shape[2]-1)*self.stride+1),
                output_shape[3] - ((x.shape[3]-1)*self.stride+1)
            )
        else:
            output_padding = 0

        # compute the convolution
        x = torch.nn.functional.conv_transpose2d(x, 
                            weight,
                            self.bias,
                            self.stride,
                            self.padding,
                            output_padding,
                            self.groups,
                            self.dilation)

        pad = self._compute_optox_padding()
        if any(pad) > 0:
            x = real_pad2d_transpose(x, pad)
        return x

class ComplexPadConvScale2D(ComplexPadConv2D):
    def __init__(self, in_channels, filters, kernel_size=3, invariant=False,
                 groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False):
        super().__init__(
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
            weight = weight.permute(0, 1, 4, 2, 3)
            weight = weight.reshape(-1, 1, self.kernel_size, self.kernel_size)
            for i in range(self.stride//2): 
                weight = torch.nn.functional.conv2d(weight, self.blur, padding=4)
            weight = weight.reshape(self.filters, self.in_channels, 2, self.kernel_size+2*self.stride, self.kernel_size+2*self.stride)
            weight = weight.permute(0, 1, 3, 4, 2)
        return weight


class ComplexPadConvScaleTranspose2D(ComplexPadConvScale2D):
    def __init__(self, in_channels, filters, kernel_size=3, invariant=False,
                 groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False):
        super().__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, 
            invariant=invariant, groups=groups, stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)
        
        self.bias = torch.nn.Parameter(torch.zeros(2, in_channels)) if bias else None
        
    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)


class ComplexPadConv2dTest(unittest.TestCase):
    def test_conv2d_complex(self):
        nBatch = 5
        M = 320
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConv2D(nf_in, nf_out, kernel_size=3).cuda()
        
        x = torch.randn(nBatch, nf_in, M, N, 2).cuda()
        Kx = model(x)
        
        y = torch.randn(*Kx.shape).cuda()
        KHy = model.backward(y, output_shape=x.shape)

        rhs = complex_mult_conj(Kx, y).view(-1, 2).sum(0).detach().cpu().numpy()
        lhs = complex_mult_conj(x, KHy).view(-1, 2).sum(0).detach().cpu().numpy()

        self.assertTrue(rhs[0] + 1j*rhs[1], lhs[0] + 1j*lhs[1])

    def test_constraints(self):
        nBatch = 5
        M = 320
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConv2D(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True).double().cuda()

        np_weight = model.weight.data.detach().cpu().numpy()
        np_weight = np_weight[...,0] + 1j * np_weight[...,1]

        self.assertTrue(np.max(np.abs(np.mean(np_weight, axis=(1,2,3)))) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=(1,2,3)))
        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

class ComplexPadConv2dGradientTest(unittest.TestCase):
    def test_conv2d_complex_gradient(self):
        nBatch = 1
        M = 10
        N = 10
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConv2D(nf_in, nf_out, kernel_size=3).cuda()
        
        npth_img = np.arange(0,nBatch*M*N*nf_in).reshape(nBatch,nf_in,M,N,1) + 1j * np.arange(0,nBatch*M*N*nf_in).reshape(nBatch,nf_in,M,N,1)

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
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

class ComplexPadConvScaleTranspose2dTest(unittest.TestCase):
    def test_conv_transpose2d_complex(self):
        nBatch = 5
        M = 320
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = ComplexPadConvScaleTranspose2D(nf_in, nf_out, kernel_size=3).cuda()
        
        x = torch.randn(nBatch, nf_in, M, N, 2).cuda()
        Kx = model.backward(x)
        
        y = torch.randn(*Kx.shape).cuda()
        KHy = model.forward(y, output_shape=x.shape)

        rhs = complex_mult_conj(Kx, y).view(-1, 2).sum(0).detach().cpu().numpy()
        lhs = complex_mult_conj(x, KHy).view(-1, 2).sum(0).detach().cpu().numpy()

        self.assertTrue(rhs[0] + 1j*rhs[1], lhs[0] + 1j*lhs[1])

class ComplexPadConvScale2dTest(unittest.TestCase):
    def test_conv2d_complex(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32

        model = ComplexPadConvScale2D(nf_in, nf_out, kernel_size=3, stride=2).cuda()

        x = torch.randn(nBatch, nf_in, M, N, 2).cuda()
        Kx = model(x)

        y = torch.randn(*Kx.shape).cuda()
        KHy = model.backward(y, output_shape=x.shape)

        rhs = complex_mult_conj(Kx, y).view(-1, 2).sum(0).detach().cpu().numpy()
        lhs = complex_mult_conj(x, KHy).view(-1, 2).sum(0).detach().cpu().numpy()
        self.assertTrue(rhs[0] + 1j*rhs[1], lhs[0] + 1j*lhs[1])

class PseudoComplexPadConvScale2d(unittest.TestCase):
    def test_conv2d_complex(self):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 1
        nf_out = 32

        model = PseudoComplexPadConv2D(nf_in, nf_out, kernel_size=3).cuda()

        x = torch.randn(nBatch, nf_in*2, M, N).cuda()
        Kx = model(x)

        y = torch.randn(*Kx.shape).cuda()
        KHy = model.backward(y, output_shape=x.shape)

        rhs = torch.sum(Kx * y).detach().cpu().numpy()
        lhs = torch.sum(x * KHy).detach().cpu().numpy()

        self.assertTrue(rhs, lhs)

    def test_constraints(self):
        nBatch = 5
        M = 320
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = PseudoComplexPadConv2D(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True).double().cuda()

        np_weight = model.weight.data.detach().cpu().numpy()
        np_weight = np_weight[...,0] + 1j * np_weight[...,1]

        self.assertTrue(np.max(np.abs(np.mean(np_weight, axis=(1,2,3)))) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=(1,2,3)))

        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

if __name__ == "__main__":
    unittest.test()

