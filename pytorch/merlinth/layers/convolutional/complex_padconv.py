import torch
import numpy as np
import optoth.pad
import unittest
import merlinth.utils
from merlinth.layers.convolutional.complex_conv import complex_conv, complex_conv_transpose
from merlinth.layers.convolutional.padconv import PadConv
from merlinth.layers.module import ComplexModule

__all__ = ['ComplexPadConv1d',
           'ComplexPadConv2d',
           'ComplexPadConv3d',
           'ComplexPadConvScale2d',
           'ComplexPadConvScale3d',
           'ComplexPadConvScaleTranspose2d',
           'ComplexPadConvScaleTranspose3d',
           'ComplexPadConvRealWeight1d',
           'ComplexPadConvRealWeight2d',
           'ComplexPadConvRealWeight3d',
           ]

class ComplexPadConv(ComplexModule):
    def __init__(self, 
                 rank,
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = merlinth.utils.validate_input_dimension(rank, kernel_size)
        self.stride = merlinth.utils.validate_input_dimension(rank, stride)
        self.dilation = merlinth.utils.validate_input_dimension(rank, dilation)
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(filters, dtype=merlinth.get_default_cdtype())) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.pad = pad
        self.padding_mode = padding
        self.rank = rank

        # add the parameter
        self.weight = torch.nn.Parameter(torch.empty(filters, in_channels, *self.kernel_size, dtype=merlinth.get_default_cdtype()))
        # insert them using a normal distribution
        # TODO complex
        torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1/np.prod(in_channels*np.prod(kernel_size))))

        # specify reduction index
        self.weight.L_init = 1e+4
        if zero_mean or bound_norm:
            self.weight.reduction_dim = tuple([i+1 for i in range(rank+1)])
    
            # define a projection
            def l2_proj(surface=False):
                # reduce the mean
                if zero_mean:
                    mean = torch.sum(self.weight.data, self.weight.reduction_dim, True) / (self.in_channels*np.prod(self.kernel_size))
                    self.weight.data.sub_(mean)
                # normalize by the l2-norm
                if bound_norm:
                    norm = torch.sum(torch.real(torch.conj(self.weight.data)*self.weight.data), self.weight.reduction_dim, True).sqrt_()
                    if surface:
                        self.weight.data.div_(norm)
                    else:
                        self.weight.data.div_(
                            torch.max(norm, torch.ones_like(norm)))
            self.weight.proj = l2_proj

            # initially call the projection
            self.weight.proj(True)

        if rank == 1:
            self._convolution_op = torch.nn.functional.conv1d
            self._padding_op = optoth.pad.pad1d
            self._convolutionT_op = torch.nn.functional.conv_transpose1d
            self._paddingT_op = optoth.pad.pad1d_transpose
        elif rank == 2:
            self._convolution_op = torch.nn.functional.conv2d
            self._padding_op = optoth.pad.pad2d
            self._convolutionT_op = torch.nn.functional.conv_transpose2d
            self._paddingT_op = optoth.pad.pad2d_transpose
        elif rank == 3:
            self._convolution_op = torch.nn.functional.conv3d
            self._padding_op = optoth.pad.pad3d
            self._convolutionT_op = torch.nn.functional.conv_transpose3d
            self._paddingT_op = optoth.pad.pad3d_transpose

    def get_weight(self):
        weight = self.weight
        return weight

    def _compute_optox_padding(self):
        pad = []
        for w in self.get_weight().shape[2:][::-1]:
            pad += [w//2, w//2]
        return pad

    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # then pad
        pad = self._compute_optox_padding()
        if self.pad and any(pad) > 0:
            x = self._padding_op(x, pad, mode=self.padding_mode)
        # compute the convolution
        x = complex_conv(self._convolution_op, x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = [output_shape[-self.rank+i] - ((x.shape[-self.rank+i]-1)*self.stride[i]+1) for i in range(self.rank)]
        else:
            output_padding = 0

        # compute the convolution
        x = complex_conv_transpose(self._convolutionT_op, x, weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        pad = self._compute_optox_padding()
        if self.pad and any(pad) > 0:
            x = self._paddingT_op(x, pad, mode=self.padding_mode)
        return x

    def extra_repr(self):
        s = "({filters}, {in_channels}, {kernel_size})"
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

class ComplexPadConvRealWeight(PadConv):
    def __init__(self, 
                 rank,
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__(rank, in_channels, filters, kernel_size, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias, zero_mean=zero_mean, bound_norm=bound_norm, pad=pad)
    def forward(self, x):
        Kx_re = super().forward(torch.real(x).contiguous())
        Kx_im = super().forward(torch.imag(x).contiguous())
        return torch.complex(Kx_re, Kx_im)
    
    def backward(self, x, output_shape=None):
        KTx_re = super().backward(torch.real(x).contiguous(), output_shape)
        KTx_im = super().backward(torch.imag(x).contiguous(), output_shape)
        return torch.complex(KTx_re, KTx_im)

class ComplexPadConv1d(ComplexPadConv):
    def __init__(self, 
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__(1,
                        in_channels=in_channels,
                        filters=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias, 
                        zero_mean=zero_mean,
                        bound_norm=bound_norm,
                        pad=pad)

class ComplexPadConvRealWeight1d(ComplexPadConvRealWeight):
    def __init__(self, 
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__(1,
                        in_channels=in_channels,
                        filters=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias, 
                        zero_mean=zero_mean,
                        bound_norm=bound_norm,
                        pad=pad)

class ComplexPadConv2d(ComplexPadConv):
    def __init__(self, 
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__(2,
                        in_channels=in_channels,
                        filters=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias, 
                        zero_mean=zero_mean,
                        bound_norm=bound_norm,
                        pad=pad)

class ComplexPadConvRealWeight2d(ComplexPadConvRealWeight):
    def __init__(self, 
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__(2,
                        in_channels=in_channels,
                        filters=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias, 
                        zero_mean=zero_mean,
                        bound_norm=bound_norm,
                        pad=pad)

class ComplexPadConvScale2d(ComplexPadConv):
    def __init__(self, in_channels, filters, kernel_size=3, groups=1, stride=2, bias=False, padding='symmetric', zero_mean=False, bound_norm=False):
        super(ComplexPadConvScale2d, self).__init__(
            2, 
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, padding=padding, 
            stride=stride, dilation=1, groups=groups, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)
        assert self.kernel_size[0] == self.kernel_size[1]
        assert self.stride[0] == self.stride[1]

        # create the convolution kernel
        if any(self.stride) > 1:
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            np_k = np.reshape(np_k, (1, 1, 5, 5))
            self.register_buffer('blur', torch.from_numpy(np_k))

    def get_weight(self):
        weight = super().get_weight()
        if any(self.stride) > 1:
            weight = weight.reshape(-1, 1, *self.kernel_size)
            for i in range(self.stride//2): 
                weight_re = torch.nn.functional.conv2d(torch.real(weight), self.blur, padding=4)
                weight_im = torch.nn.functional.conv2d(torch.imag(weight), self.blur, padding=4)
                weight = torch.complex(weight_re, weight_im)
            weight = weight.reshape(self.filters,
                                    self.in_channels,
                                    self.kernel_size[0]+2*self.stride[0],
                                    self.kernel_size[1]+2*self.stride[1])
        return weight

class ComplexPadConvScaleTranspose2d(ComplexPadConvScale2d):
    def __init__(self, in_channels, filters, kernel_size=3, padding='symmetric',
                 groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(ComplexPadConvScaleTranspose2d, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, 
            padding=padding, groups=groups, stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)

class ComplexPadConv3d(ComplexPadConv):
    def __init__(self, 
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__(3,
                        in_channels=in_channels,
                        filters=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias, 
                        zero_mean=zero_mean,
                        bound_norm=bound_norm,
                        pad=pad)

class ComplexPadConvRealWeight3d(ComplexPadConvRealWeight):
    def __init__(self, 
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__(3,
                        in_channels=in_channels,
                        filters=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias, 
                        zero_mean=zero_mean,
                        bound_norm=bound_norm,
                        pad=pad)

class ComplexPadConvScale3d(ComplexPadConv3d):
    def __init__(self, in_channels, filters, kernel_size=3, padding='symmetric',
                 groups=1, stride=(1,2,2), bias=False, zero_mean=False, bound_norm=False):
        super(ComplexPadConvScale3d, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, padding=padding,
            stride=stride, dilation=1, groups=groups, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)
        assert self.kernel_size[1] == self.kernel_size[2]
        assert self.kernel_size[1] > 1
        assert self.stride[1] == self.stride[2]
        assert self.stride[1] > 1

        # create the convolution kernel
        if self.stride[1] > 1:
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            np_k = np.reshape(np_k, (1, 1, 5, 5))
            self.register_buffer('blur', torch.from_numpy(np_k))

    def get_weight(self):
        weight = super().get_weight()
        if self.stride[1] > 1:
            weight = weight.reshape(-1, 1, self.kernel_size[1], self.kernel_size[2])
            for i in range(self.stride[1]//2): 
                weight_re = torch.nn.functional.conv2d(torch.real(weight), self.blur, padding=4)
                weight_im = torch.nn.functional.conv2d(torch.imag(weight), self.blur, padding=4)
                weight = torch.complex(weight_re, weight_im)

            weight = weight.reshape(self.filters, self.in_channels, self.kernel_size[0], self.kernel_size[1]+2*self.stride[1], self.kernel_size[1]+2*self.stride[1])
        return weight

class ComplexPadConvScaleTranspose3d(ComplexPadConvScale3d):
    def __init__(self, in_channels, filters, kernel_size=3, groups=1, stride=(1,2,2), bias=False, zero_mean=False, bound_norm=False):
        super(ComplexPadConvScaleTranspose3d, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, 
            groups=groups, stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)

class ComplexPadConv2Dt(torch.nn.Module):
    def __init__(self, in_channels, intermediate_filters, filters, kernel_size=3,
                 stride=1, bias=False, zero_mean=True, bound_norm=True, pad=True):
        super(ComplexPadConv2Dt, self).__init__()

        if stride > 2:
            conv_module = ComplexPadConvScale3d
        else:
            conv_module = ComplexPadConv3d

        self.conv_xy = conv_module(in_channels,
                    intermediate_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    zero_mean=zero_mean,
                    bound_norm=bound_norm,
                    pad=pad)

        self.conv_t = ComplexPadConv3d(intermediate_filters,
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

# Aliases
ComplexPadConv1D = ComplexPadConv1d
ComplexPadConvRealWeight1D = ComplexPadConvRealWeight1d
ComplexPadConv2D = ComplexPadConv2d
ComplexPadConvRealWeight2D = ComplexPadConvRealWeight2d
ComplexPadConvScale2D = ComplexPadConvScale2d
ComplexPadConvScaleTranspose2D = ComplexPadConvScaleTranspose2d
ComplexPadConv3D = ComplexPadConv3d
ComplexPadConvRealWeight3D = ComplexPadConvRealWeight3d
ComplexPadConvScale3D = ComplexPadConvScale3d
ComplexPadConvScaleTranspose3D = ComplexPadConvScaleTranspose3d
ComplexPadConv2dt = ComplexPadConv2Dt
