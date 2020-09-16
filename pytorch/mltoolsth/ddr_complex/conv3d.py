import torch
import optoth.pad3d

import numpy as np

__all__ = ['Conv3d', 'ConvScale3d', 'ConvScaleTranspose3d']

class Conv3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, groups=1, bias=False, 
                 zero_mean=False, bound_norm=False, pad=True):
        super(Conv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.pad = pad

        # add the parameter
        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, self.kernel_size,  self.kernel_size, self.kernel_size))

        # insert them using a normal distribution
        torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1/np.prod(in_channels*kernel_size**3)))

        # specify reduction index
        self.weight.L_init = 1e+4
        if zero_mean or bound_norm:
            self.weight.reduction_dim = (1, 2, 3, 4)
    
            # define a projection
            def l2_proj(surface=False):
                # reduce the mean
                if zero_mean:
                    mean = torch.sum(self.weight.data, self.weight.reduction_dim, True) / (self.in_channels*self.kernel_size**2)
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
        return self.weight

    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # then pad
        pad = (weight.shape[-3]//2, weight.shape[-2]//2, weight.shape[-1]//2)
        if self.pad and any(pad) > 0:
            x = optoth.pad3d.pad3d(x, (pad[-1],pad[-1],pad[-2],pad[-2],pad[-3],pad[-3]), mode='symmetric')
        # compute the convolution
        return torch.nn.functional.conv3d(x, weight, self.bias, (1, self.stride, self.stride), self.padding, self.dilation, self.groups)

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = (
                output_shape[2] - ((x.shape[2]-1)+1),
                output_shape[3] - ((x.shape[3]-1)*self.stride+1),
                output_shape[4] - ((x.shape[4]-1)*self.stride+1)
            )
        else:
            output_padding = 0

        # compute the convolution
        x = torch.nn.functional.conv_transpose3d(x, weight, self.bias, (1, self.stride, self.stride), self.padding, output_padding, self.groups, self.dilation)
        pad = (weight.shape[-3]//2, weight.shape[-2]//2, weight.shape[-1]//2)
        if self.pad and any(pad) > 0:
            x = optoth.pad3d.pad3d_transpose(x, (pad[-1],pad[-1],pad[-2],pad[-2],pad[-3],pad[-3]), mode='symmetric')
        return x

    def extra_repr(self):
        s = "({out_channels}, {in_channels}, {kernel_size}), invariant={invariant}"
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


class ConvScale3d(Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size=3, invariant=False,
                 groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(ConvScale3d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            stride=stride, dilation=1, groups=groups, bias=bias, 
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
            weight = weight.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size+2*self.stride, self.kernel_size+2*self.stride)
        return weight


class ConvScaleTranspose3d(ConvScale3d):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(ConvScaleTranspose3d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            groups=groups, stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)