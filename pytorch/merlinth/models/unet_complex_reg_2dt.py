
import torch
import torch.nn.functional as F
from merlinth.layers.convolutional.complex_padconv import (
    ComplexPadConv3D,
    ComplexPadConvScale3D,
    ComplexPadConvScaleTranspose3D
)
from merlinth.layers.complex_init import *
from merlinth.layers.complex_norm import get_normalization

import numpy as np
from merlinth.layers.complex_act import *

__all__ = ['ComplexSplitFast',
           'ComplexSplitFastUpsampling',
           'ComplexConvBlock3d',
           'ComplexConvBlock3dUpsampling',
           'ComplexConvBlock2dt',
           'ComplexConvBlock2dtUpsampling']

def get_activation(activation, num_parameters):
    if activation == 'cReLU':
        return cReLU()
    elif activation == 'cPReLU':
        return cPReLU(num_parameters)
    elif activation == 'ModReLU':
        return ModReLU(num_parameters)
    elif activation == 'ModPReLU':
        return ModPReLU(num_parameters)
    elif activation == 'identity':
        return Identity(num_parameters)
    else:
        raise ValueError("Options for activation: ['cReLU', 'cPReLU', 'ModPReLU', 'ModReLU', 'ComplexTrainablePolarActivation', 'ComplexTrainableMagnitudeActivation', 'ComplexStudentT']")

class ComplexConvBlock3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_sp=3, kernel_size_t=3,
                 stride=1, dilation=1, groups=1, bias=False, activation='cPReLU',
                 zero_mean=False, bound_norm=False, normalization='no'):
        super(ComplexConvBlock3d, self).__init__()
        if stride > 2:
            conv_module = ComplexPadConvScale3D
        else:
            conv_module = ComplexPadConv3D

        self.conv = conv_module(in_channels,
                    out_channels,
                    kernel_size = (kernel_size_t, kernel_size_sp, kernel_size_sp),
                    stride=stride,
                    bias=bias,
                    zero_mean=zero_mean,
                    bound_norm=bound_norm)

        self.norm = get_normalization(normalization)
        self.act = get_activation(activation, out_channels)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ComplexConvBlock3dUpsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_sp=3, kernel_size_t=3,
                 stride=1, dilation=1, groups=1, bias=False, activation = 'cPReLU', normalization='no'):
        super(ComplexConvBlock3dUpsampling, self).__init__()

        self.conv = ComplexPadConvScaleTranspose3D(out_channels,
                    in_channels,
                    kernel_size = (kernel_size_t, kernel_size_sp, kernel_size_sp),
                    stride=stride,
                    bias=bias)
        self.norm = get_normalization(normalization)
        self.act = get_activation(activation, out_channels)

    def forward(self, x, output_shape):
        return self.act(self.norm(self.conv(x, output_shape)))

class ComplexSplitFast(torch.nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, kernel_size_sp=3, kernel_size_t=3,
                 stride=1, dilation=1, groups=1, bias=False, activation='cPReLU', normalization='no'):
        super(ComplexSplitFast, self).__init__()

        nom = kernel_size_sp ** 2 * kernel_size_t * in_channels * out_channels
        denom = kernel_size_sp ** 2 * in_channels + 2 * kernel_size_sp * kernel_size_t * out_channels
        inter_channels = int(np.floor(nom/denom))

        if stride > 2:
            conv_module = ComplexPadConvScale3D
        else:
            conv_module = ComplexPadConv3D

        self.conv_xy = conv_module(in_channels,
                    inter_channels,
                    kernel_size = (1, kernel_size_sp, kernel_size_sp),
                    stride=stride,
                    bias=bias)

        self.conv_xt = ComplexPadConv3D(inter_channels,
                 out_channels,
                 kernel_size = (kernel_size_t, 1, kernel_size_sp),
                 bias=bias)

        self.conv_yt = ComplexPadConv3D(inter_channels,
                 out_channels,
                 kernel_size = (kernel_size_t, kernel_size_sp, 1),
                 bias=bias)

        # self.conv_t = ComplexConv3d(out_channels*2,
        #          out_channels,
        #          kernel_size_sp_x=1,
        #          kernel_size_sp_y=1,
        #          kernel_size_t=1,
        #          bias=bias)

        # self.norm_xy = get_normalization(normalization)
        # self.norm_xt = get_normalization(normalization)
        # self.norm_yt = get_normalization(normalization)

        # self.act_xy = get_activation(activation, inter_channels)
        # self.act_xt = get_activation(activation, out_channels)
        # self.act_yt = get_activation(activation, out_channels)
        self.norm = get_normalization(normalization)
        self.act = get_activation(activation, out_channels)

    def forward(self, x):
        # x_sp = self.act_xy(self.norm_xy(self.conv_xy(x)))
        # x_xt = self.act_xt(self.norm_xt(self.conv_xt(x_sp)))
        # x_yt = self.act_yt(self.norm_yt(self.conv_yt(x_sp)))
        # return 0.5*(x_xt + x_yt)
        #x_sp = self.act_xy(self.norm_xy(self.conv_xy(x)))
        x_sp = self.conv_xy(x)
        x_xt = self.conv_xt(x_sp)
        x_yt = self.conv_yt(x_sp)
        return self.act(self.norm(x_xt + x_yt))

class ComplexSplitFastUpsampling(torch.nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, kernel_size_sp=3, kernel_size_t=3,
                 stride=1, dilation=1, groups=1, bias=False, activation='cPReLU', normalization='no'):
        super(ComplexSplitFastUpsampling, self).__init__()

        nom = kernel_size_sp ** 2 * kernel_size_t * in_channels * out_channels
        denom = kernel_size_sp ** 2 * in_channels + 2 * kernel_size_sp * kernel_size_t * out_channels
        inter_channels = int(np.floor(nom/denom))
        
        self.conv_xy = ComplexPadConvScaleTranspose3D(inter_channels,
                    in_channels,
                    kernel_size = (1, kernel_size_sp, kernel_size_sp),
                    stride=stride,
                    bias=bias)

        self.conv_xt = ComplexPadConv3D(inter_channels,
                 out_channels,
                 kernel_size = (kernel_size_t, 1, kernel_size_sp),
                 bias=bias)

        self.conv_yt = ComplexPadConv3D(inter_channels,
                 out_channels,
                 kernel_size = (kernel_size_t, kernel_size_sp, 1),
                 bias=bias)

        # self.norm_xy = get_normalization(normalization)
        # self.norm_xt = get_normalization(normalization)
        # self.norm_yt = get_normalization(normalization)

        # self.act_xy = get_activation(activation, inter_channels)
        # self.act_xt = get_activation(activation, out_channels)
        # self.act_yt = get_activation(activation, out_channels)
        self.norm = get_normalization(normalization)
        self.act = get_activation(activation, out_channels)

    def forward(self, x, output_shape):
        # x_sp = self.act_xy(self.norm_xy(self.conv_xy(x, output_shape)))
        # x_xt = self.act_xt(self.norm_xt(self.conv_xt(x_sp)))
        # x_yt = self.act_yt(self.norm_yt(self.conv_yt(x_sp)))
        # return 0.5*(x_xt + x_yt)
        x_sp = self.conv_xy(x, output_shape)
        x_xt = self.conv_xt(x_sp)
        x_yt = self.conv_yt(x_sp)
        return self.act(self.norm(x_xt + x_yt))
        # x_sp = self.act_xy(self.norm_xy(self.conv_xy(x, output_shape)))
        # x_xt = self.conv_xt(x_sp)
        # x_yt = self.conv_yt(x_sp)
        # return self.act_xt(self.norm_xt(x_xt + x_yt))

class ComplexConvBlock2dt(torch.nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, kernel_size_sp=3, kernel_size_t=3,
                 stride=1, dilation=1, groups=1, bias=False, activation='cPReLU', activation_xy = False, normalization='no'):
        super(ComplexConvBlock2dt, self).__init__()

        if stride > 2:
            conv_module = ComplexPadConvScale3D
        else:
            conv_module = ComplexPadConv3D

        nom = kernel_size_sp ** 2 * kernel_size_t * in_channels * out_channels
        denom = kernel_size_sp ** 2 * in_channels + kernel_size_t * out_channels
        inter_channels = int(np.floor(nom/denom))

        self.conv_xy = conv_module(in_channels,
                    inter_channels,
                    kernel_size = (1, kernel_size_sp, kernel_size_sp),
                    stride=stride,
                    bias=bias)

        self.conv_t = ComplexPadConv3D(inter_channels,
                 out_channels,
                 kernel_size = (kernel_size_t, 1, 1),
                 bias=bias)
        #self.conv_t.weight.lrscale = 2

        self.activation_xy = activation_xy

        if activation_xy:
            self.act_xy = get_activation(activation, inter_channels)

        self.norm = get_normalization(normalization)
        self.act = get_activation(activation, out_channels)

    def forward(self, x):
        x_sp = self.conv_xy(x)
        if self.activation_xy:
            x_sp = self.act_xy(self.norm(x_sp))

        x_t = self.conv_t(x_sp)
        return self.act(self.norm(x_t))

class ComplexConvBlock2dtUpsampling(torch.nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, kernel_size_sp=3, kernel_size_t=3,
                 stride=1, dilation=1, groups=1, bias=False, activation='cPReLU', activation_xy = False, normalization='no'):
        super(ComplexConvBlock2dtUpsampling, self).__init__()

        nom = kernel_size_sp ** 2 * kernel_size_t * in_channels * out_channels
        denom = kernel_size_sp ** 2 * in_channels + kernel_size_t * out_channels
        inter_channels = int(np.floor(nom/denom))

        self.conv_xy = ComplexPadConvScaleTranspose3D(inter_channels,
                    in_channels,
                    kernel_size = (1, kernel_size_sp, kernel_size_sp),
                    stride=stride,
                    bias=bias)

        self.conv_t = ComplexPadConv3D(inter_channels,
                 out_channels,
                 kernel_size = (kernel_size_t, 1, 1),
                 bias=bias)
        #self.conv_t.weight.lrscale = 2

        self.activation_xy = activation_xy
        
        if activation_xy:
            self.act_xy = get_activation(activation, inter_channels)

        self.norm = get_normalization(normalization)
        self.act = get_activation(activation, out_channels)
    
    def forward(self, x, output_shape):
        x_sp = self.conv_xy(x, output_shape)
        if self.activation_xy:
            x_sp = self.act_xy(self.norm(x_sp))
        x_t = self.conv_t(x_sp)
        return self.act(self.norm(x_t))

