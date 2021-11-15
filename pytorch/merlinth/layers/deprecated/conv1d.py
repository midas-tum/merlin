import torch
import numpy as np
from merlinth.layers.pad import (
    real_pad1d,
    real_pad1d_transpose
)
import unittest

__all__ = ['PadConv1D', 'PadConvScale1D', 'PadConvScaleTranspose1D']

class PadConv1D(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3,
                 stride=1, dilation=1, groups=1, bias=False, 
                 zero_mean=False, bound_norm=False, pad=True):
        super(PadConv1D, self).__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(filters)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.pad = pad

        # add the parameter
        self.weight = torch.nn.Parameter(torch.empty(filters, in_channels, self.kernel_size))
        # insert them using a normal distribution
        torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1/np.prod(in_channels*kernel_size)))

        # specify reduction index
        self.weight.L_init = 1e+4
        if zero_mean or bound_norm:
            self.weight.reduction_dim = (1, 2, )
    
            # define a projection
            def l2_proj(surface=False):
                # reduce the mean
                if zero_mean:
                    mean = torch.sum(self.weight.data, self.weight.reduction_dim, True) / (self.in_channels*self.kernel_size)
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
            x = real_pad1d(x, pad)
        # compute the convolution
        x = torch.nn.functional.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = (
                output_shape[2] - ((x.shape[2]-1)*self.stride+1),
            )
            # output_padding = (output_padding[0]//2 + self.kernel_size // 2, output_padding[1]//2 + self.kernel_size // 2)
        else:
            output_padding = 0

        # compute the convolution
        x = torch.nn.functional.conv_transpose1d(x, weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

        pad = self._compute_optox_padding()
        if self.pad and any(pad) > 0:
            x = real_pad1d_transpose(x, pad)
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


