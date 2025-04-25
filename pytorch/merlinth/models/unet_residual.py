from functools import total_ordering
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest
from merlinth.layers.convolutional.complex_padconv import (
    ComplexPadConv3D,
    ComplexPadConvScale3D,
    ComplexPadConvScaleTranspose3D
)
from merlinth.layers.complex_init import *
from merlinth.layers.complex_norm import get_normalization
from merlinth.layers.complex_maxpool import MagnitudeMaxPool3D
from merlinth.models.unet_complex_residual_block import *
from merlinth.models.unet_complex_reg_2dt import *
import optoth.pad

__all__ = ['ResidualUnetModel3d', 'ResidualUnetModel2dt', 'ResidualUnetModelFast']

class ResidualUnetModel(nn.Module):
    """
    Residual Unet
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            channels,
            num_pool_layers,
            local_residual=False,
            bias=True,
            multiplier=1,
            **kwargs,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input to the U-Net model.
            out_channels (int): Number of channels in the output to the U-Net model.
            channels (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            local_residual (bool): whether to add local residual or not
            use_instance_norm (bool): whether to use instance norm or not
            use_dropout (bool): use drop out
            drop_prob (float): Dropout probability.
        """
        super(ResidualUnetModel, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_pool_layers = num_pool_layers
        self.local_residual = local_residual
        # self.residual_scale = kwargs.pop('residual_scale', 1.0)
        # self.norm = get_normalization(kwargs.pop('normalization', 'None'))
               
        self.gain = kwargs.pop('gain', 1.0)
        self.mode = kwargs.pop('mode', 'glorot')
        self.ortho = kwargs.pop('ortho', False)
        #print('mode=', self.mode, 'ortho=', self.ortho)

    def weight_init(self, module):
        pass
#         TODO
#         if isinstance(module, ComplexPadConv3D) \
#             or isinstance(module, ComplexPadConvScale3D) \
#             or isinstance(module, ComplexPadConvScaleTranspose3D):

#             if self.ortho:
#                 complex_independent_filters_init(module.weight, mode=self.mode)
#             else:
#                 complex_init(module.weight, mode=self.mode)

#             if hasattr(module.weight, 'proj'):
#                 # initially call the projection
#                 module.weight.proj(True)

#             if module.bias is not None:
#                 module.bias.data.fill_(0)

#         if isinstance(module, torch.nn.Conv3d) \
#            or isinstance(module, torch.nn.Linear):
#             # torch.nn.init.kaiming_normal_(
#             #     module.weight,
#             #     mode='fan_in',
#             #     nonlinearity='relu',
#             # )
# #             torch.nn.init.kaiming_uniform_(
# #                 module.weight, a=self.gain,
# #                 mode='fan_in',
# #                 nonlinearity='relu',
# #             )
#             torch.nn.init.xavier_normal_(module.weight, gain=self.gain)
#             #torch.nn.init.xavier_uniform_(module.weight, gain=self.gain)

#             if hasattr(module.weight, 'proj'):
#                 # initially call the projection
#                 module.weight.proj(True)

#             if module.bias is not None:
#                 module.bias.data.fill_(0)

    def calculate_downsampling_padding3d(self, tensor):
        # calculate pad size
        factor = 2 ** self.num_pool_layers
        imshape = np.array(tensor.shape[-4:])
        paddings = np.ceil(imshape / factor) * factor - imshape
        paddings = paddings.astype(np.int) // 2
        # reversed order of paddings
        p3d = (paddings[2], paddings[2],
               paddings[1], paddings[1],
               #paddings[0], paddings[0])
               0, 0) # no striding in time dimension!
        return p3d

    def pad3d(self, tensor, p3d):
        # print("padding", p3d)
        if np.any(p3d):
            tensor = optoth.pad.pad3d(tensor, [p3d[0], p3d[0], p3d[1], p3d[1], p3d[2], p3d[2]], mode='symmetric')
        return tensor

    def unpad3d(self, tensor, shape):
        if tensor.shape == shape:
            return tensor
        else:
            return self.complex_center_crop3d(tensor, shape)

    def complex_center_crop3d(self, data, shape):
        """
        Apply a center crop to the input real image or batch of real images.

        Args:
            data (torch.Tensor): The input tensor to be center cropped. It should have at
                least 2 dimensions and the cropping is applied along the last two dimensions.
            shape (int, int): The output shape. The shape should be smaller than the
                corresponding dimensions of data.

        Returns:
            torch.Tensor: The center cropped image
        """
        assert 0 < shape[0] <= data.shape[-3]
        assert 0 < shape[1] <= data.shape[-2]
        assert 0 < shape[2] <= data.shape[-1]
        d_from = (data.shape[-3] - shape[0]) // 2
        w_from = (data.shape[-2] - shape[1]) // 2
        h_from = (data.shape[-1] - shape[2]) // 2
        d_to = d_from + shape[0]
        w_to = w_from + shape[1]
        h_to = h_from + shape[2]
        return data[..., d_from:d_to, w_from:w_to, h_from:h_to]

    # def normalize(self, data, eps=0.):
    #     """
    #     Normalize the given tensor using:
    #         (data - mean) / (stddev + eps)
    #     Args:
    #         data (torch.Tensor): Input data to be normalized
    #         mean (float): Mean value
    #         stddev (float): Standard deviation
    #         eps (float): Added to stddev to prevent dividing by zero
    #     Returns:
    #         torch.Tensor: Normalized tensor
    #     """
    #     mean = data.mean()
    #     stddev = data.std()
    #     return (data - mean) / (stddev + eps), mean, stddev

    # def unnormalize(self, data, mean, stddev, eps=0.):
    #     return data * (stddev + eps) + mean

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_channels, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_channels, height, width]
        """
        stack = []
        
        #output, mean, stddev = self.normalize(input, self.eps)
        output = input
        #print(output.max(), output.min())

        # if self.pad_data:
        orig_shape3d = output.shape[-4:]

        # calculate padding and pad data
        p3d = self.calculate_downsampling_padding3d(output)
        #print(p3d)
        output = self.pad3d(output, p3d)
        output = self.block_in(output)

        #print('block in', output.shape)

        # Apply down-sampling layers
        for i in range(self.num_pool_layers):
            output = self.block_enc[i](output)
            #print('Enc', output.shape)
            stack.append(output)
            output = self.ds_enc[i](output)
            #print('Ds', output.shape)

        output = self.block_enc[self.num_pool_layers](output)
        #print('Enc', output.shape)
        #print('stack', stack[-1].shape)

        # Apply up-sampling layers
        for i in range(self.num_pool_layers):
            output = self.us_dec[i](output, stack[-1].shape)
            #print('Us', output.shape)
            output = torch.cat([output, stack.pop()], dim=1)
            #print('Cat', output.shape)
            output = self.cat_dec[i](output)
            #print('Cat block', output.shape)
            output = self.block_dec[i](output)
            #print('Block dec', output.shape)

        output = self.conv_out(output)
        #output = self.norm(output)
        #print('Conv out', output.shape)

        output = self.unpad3d(output, orig_shape3d)

        #output = self.unnormalize(output, mean, stddev, 0)
        #print(output.max(), output.min())

        return output

class ResidualUnetModel2dt(ResidualUnetModel):
    """
    Residual Unet
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            channels,
            num_pool_layers,
            local_residual=False,
            bias=True,
            multiplier=1,
            gain=1,
            activation = 'cPReLU',
            activation_xy = False,
            normalization = 'no',
            kernel_size_sp = 3,
            kernel_size_t = 3,
            **kwargs,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input to the U-Net model.
            out_channels (int): Number of channels in the output to the U-Net model.
            channels (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            local_residual (bool): whether to add local residual or not
            use_instance_norm (bool): whether to use instance norm or not
            use_dropout (bool): use drop out
            drop_prob (float): Dropout probability.
        """
        super(ResidualUnetModel2dt, self).__init__(
            in_channels,
            out_channels,
            channels,
            num_pool_layers,
            local_residual,
            bias,
            multiplier,
            activation='cPReLU',
            activation_xy=False,
            normalization = normalization,
            gain=gain,
            **kwargs,)

        # first conv bloc
        self.block_in = ComplexConvBlock2dt(in_channels=in_channels,
                                     inter_channels=channels,
                                     out_channels=channels,
                                     bias=bias,
                                     activation=activation,
                                     activation_xy=activation_xy,
                                     normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)

        # contraction path
        self.block_enc = nn.ModuleList()
        self.ds_enc = nn.ModuleList()
        layer_ch = channels


        EncDecBlock  = ComplexResidualBlock2dt if self.local_residual else ComplexConvBlock2dt
        for i in range(num_pool_layers):
            self.block_enc += [EncDecBlock(in_channels=layer_ch,
                                           inter_channels=layer_ch,
                                           out_channels=layer_ch,
                                           bias=bias,
                                           activation=activation,
                                           activation_xy=activation_xy,
                                           normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            self.ds_enc += [torch.nn.Sequential(MagnitudeMaxPool3D(),
                            ComplexConvBlock2dt(in_channels=layer_ch,
                                             inter_channels=layer_ch*multiplier,
                                             out_channels=layer_ch*multiplier,
                                             bias=bias,
                                             stride=1,
                                             activation=activation,
                                             activation_xy=activation_xy,
                                             normalization=normalization,
                                            kernel_size_sp=kernel_size_sp,
                                            kernel_size_t=kernel_size_t))]
            layer_ch *= multiplier

        self.block_enc += [EncDecBlock(in_channels=layer_ch,
                                        inter_channels=layer_ch,
                                        out_channels=layer_ch,
                                        bias=bias,
                                        activation=activation,
                                        activation_xy=activation_xy,
                                        normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]

        self.block_dec = nn.ModuleList()
        self.cat_dec = nn.ModuleList()
        self.us_dec = nn.ModuleList()
        
        for i in range(num_pool_layers):
            self.us_dec +=  [ComplexConvBlock2dtUpsampling(in_channels=layer_ch,
                                                 inter_channels=layer_ch,
                                                 out_channels=layer_ch//multiplier,
                                                 bias=bias,
                                                 stride=2,
                                                 activation=activation,
                                                 activation_xy=activation_xy,
                                                 normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            self.cat_dec += [ComplexConvBlock2dt(in_channels=layer_ch*2//multiplier,
                                        inter_channels=layer_ch//multiplier,
                                        out_channels=layer_ch//multiplier,
                                        bias=bias,
                                        activation=activation,
                                        activation_xy=activation_xy,
                                        normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            self.block_dec += [EncDecBlock(in_channels=layer_ch//multiplier,
                                        inter_channels=layer_ch//multiplier,
                                        out_channels=layer_ch//multiplier,
                                        bias=bias,
                                        activation=activation,
                                        activation_xy=activation_xy,
                                        normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            layer_ch //= multiplier
        
        # self.conv_out = ComplexPadConvRealWeight3d(layer_ch,
        #                                         out_channels,
        #                                         kernel_size_sp_x=1,
        #                                         kernel_size_sp_y=1,
        #                                         kernel_size_t=1,
        #                                         bias=False,
        #                                         zero_mean=True,
        #                                         bound_norm=True)
        

        out_bias = kwargs.pop('out_bias', False)
        out_zero_mean = kwargs.pop('out_zero_mean', False)
        #print('bias=', out_bias, 'zero_mean=', out_zero_mean)

        self.conv_out = ComplexPadConv3D(layer_ch,
                                                out_channels,
                                                kernel_size = (1, 1, 1),
                                                bias=out_bias,
                                                zero_mean=out_zero_mean,
                                                # bound_norm=False,
                                                )
        
        self.apply(self.weight_init)

    def num_inter_channels(self, ksp, kt, nfin, nfout):
        nom = ksp ** 2 * kt * nfin * nfout
        denom = ksp ** 2 * nfin + kt * nfout
        return int(np.floor(nom/denom))

class ResidualUnetModelFast(ResidualUnetModel):
    """
    Residual Unet
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            channels,
            num_pool_layers,
            local_residual=False,
            bias=True,
            multiplier=1,
            gain=1,
            activation = 'cPReLU',
            normalization = 'no',
            kernel_size_sp = 3,
            kernel_size_t = 3,
            **kwargs,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input to the U-Net model.
            out_channels (int): Number of channels in the output to the U-Net model.
            channels (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            local_residual (bool): whether to add local residual or not
            use_instance_norm (bool): whether to use instance norm or not
            use_dropout (bool): use drop out
            drop_prob (float): Dropout probability.
        """
        super(ResidualUnetModelFast, self).__init__(
            in_channels,
            out_channels,
            channels,
            num_pool_layers,
            local_residual,
            bias,
            multiplier,
            gain=gain,
            normalization=normalization,
            **kwargs,)

        # first conv block
        self.block_in = ComplexSplitFast( in_channels=in_channels,
                                     inter_channels=channels,
                                     out_channels=channels,
                                     bias=bias,
                                     activation=activation,
                                     normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)

        # contraction path
        self.block_enc = nn.ModuleList()
        self.ds_enc = nn.ModuleList()
        layer_ch = channels
    
        EncDecBlock  = ComplexResidualBlockSplitFast if self.local_residual else ComplexSplitFast

        for i in range(num_pool_layers):
            self.block_enc += [EncDecBlock(in_channels=layer_ch,
                                           inter_channels=layer_ch,
                                           out_channels=layer_ch,
                                           bias=bias,
                                           activation=activation,
                                           normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            self.ds_enc += [torch.nn.Sequential(MagnitudeMaxPool3D(),
                            ComplexSplitFast(in_channels=layer_ch,
                                             inter_channels=layer_ch*multiplier,
                                             out_channels=layer_ch*multiplier,
                                             bias=bias,
                                             stride=1,
                                             activation=activation,
                                             normalization=normalization,
                                            kernel_size_sp=kernel_size_sp,
                                            kernel_size_t=kernel_size_t))]
            layer_ch *= multiplier

        self.block_enc += [EncDecBlock(in_channels=layer_ch,
                                        inter_channels=layer_ch,
                                        out_channels=layer_ch,
                                        bias=bias,
                                        activation=activation,
                                        normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]

        self.block_dec = nn.ModuleList()
        self.cat_dec = nn.ModuleList()
        self.us_dec = nn.ModuleList()
        
        for i in range(num_pool_layers):
            self.us_dec +=  [ComplexSplitFastUpsampling(in_channels=layer_ch,
                                                 inter_channels=layer_ch,
                                                 out_channels=layer_ch//multiplier,
                                                 bias=bias,
                                                 stride=(1, 2, 2),
                                                 activation=activation,
                                                 normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            self.cat_dec += [ComplexSplitFast(in_channels=layer_ch*2//multiplier,
                                        inter_channels=layer_ch//multiplier,
                                        out_channels=layer_ch//multiplier,
                                        bias=bias,
                                        activation=activation,
                                        normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            self.block_dec += [EncDecBlock(in_channels=layer_ch//multiplier,
                                        inter_channels=layer_ch//multiplier,
                                        out_channels=layer_ch//multiplier,
                                        bias=bias,
                                        activation=activation,
                                        normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            layer_ch //= multiplier
        
        # self.conv_out = ComplexPadConvRealWeight3d(layer_ch,
        #                                         out_channels,
        #                                         kernel_size_sp_x=1,
        #                                         kernel_size_sp_y=1,
        #                                         kernel_size_t=1,
        #                                         bias=False,
        #                                         zero_mean=True,
        #                                         bound_norm=True)
        

        out_bias = kwargs.pop('out_bias', False)
        out_zero_mean = kwargs.pop('out_zero_mean', False)
        #print('bias=', out_bias, 'zero_mean=', out_zero_mean)

        self.conv_out = ComplexPadConv3D(layer_ch,
                                                out_channels,
                                                kernel_size = (1, 1, 1),
                                                bias=out_bias,
                                                zero_mean=out_zero_mean,
                                                # bound_norm=False
                                                )
        
        self.apply(self.weight_init)

    
class ResidualUnetModel3d(ResidualUnetModel):
    """
    Residual Unet
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            channels,
            num_pool_layers,
            local_residual=False,
            bias=True,
            multiplier=1,
            activation = 'cPReLU',
            normalization = 'no',
            kernel_size_sp = 3,
            kernel_size_t = 3,
            **kwargs,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input to the U-Net model.
            out_channels (int): Number of channels in the output to the U-Net model.
            channels (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            local_residual (bool): whether to add global residual or not
            use_instance_norm (bool): whether to use instance norm or not
            use_dropout (bool): use drop out
            drop_prob (float): Dropout probability.
        """
        super(ResidualUnetModel3d, self).__init__(
            in_channels,
            out_channels,
            channels,
            num_pool_layers,
            local_residual,
            bias,
            multiplier,
            normalization=normalization,
            **kwargs,)

        # first conv block
        self.block_in = ComplexConvBlock3d( in_channels=in_channels,
                                     out_channels=channels,
                                     bias=bias,
                                     activation=activation,
                                     normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)

        # contraction path
        self.block_enc = nn.ModuleList()
        self.ds_enc = nn.ModuleList()
        layer_ch = channels
        EncDecBlock  = ComplexResidualBlock3d if self.local_residual else ComplexConvBlock3d

        for i in range(num_pool_layers):
            self.block_enc += [EncDecBlock(in_channels=layer_ch,
                                           out_channels=layer_ch,
                                           bias=bias,
                                           activation=activation,
                                           normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,)]
            self.ds_enc += [torch.nn.Sequential(MagnitudeMaxPool3D(),
                            ComplexConvBlock3d(in_channels=layer_ch,
                                             out_channels=layer_ch*multiplier,
                                             bias=bias,
                                             stride=1,
                                             activation=activation,
                                             normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t))]
            layer_ch *= multiplier

        self.block_enc += [EncDecBlock(in_channels=layer_ch,
                                        out_channels=layer_ch,
                                        bias=bias,
                                        activation=activation,
                                        normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]

        self.block_dec = nn.ModuleList()
        self.cat_dec = nn.ModuleList()
        self.us_dec = nn.ModuleList()
        
        for i in range(num_pool_layers):
            self.us_dec +=  [ComplexConvBlock3dUpsampling(in_channels=layer_ch,
                                                 out_channels=layer_ch//multiplier,
                                                 bias=bias,
                                                 stride=(1, 2, 2),
                                                 activation=activation,
                                                 normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            self.cat_dec += [ComplexConvBlock3d(in_channels=layer_ch*2//multiplier,
                                        out_channels=layer_ch//multiplier,
                                        bias=bias,
                                        activation=activation,
                                        normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            self.block_dec += [EncDecBlock(in_channels=layer_ch//multiplier,
                                        out_channels=layer_ch//multiplier,
                                        bias=bias,
                                        activation=activation,
                                        normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t)]
            layer_ch //= multiplier
        
        # self.conv_out = ComplexPadConvRealWeight3d(layer_ch,
        #                                         out_channels,
        #                                         kernel_size_sp_x=1,
        #                                         kernel_size_sp_y=1,
        #                                         kernel_size_t=1,
        #                                         bias=False,
        #                                         zero_mean=True,
        #                                         bound_norm=True)

        out_bias = kwargs.pop('out_bias', False)
        out_zero_mean = kwargs.pop('out_zero_mean', False)
        #print('bias=', out_bias, 'zero_mean=', out_zero_mean)

        self.conv_out = ComplexPadConv3D(layer_ch,
                                                out_channels,
                                                kernel_size = (1, 1, 1),
                                                bias=out_bias,
                                                zero_mean=out_zero_mean,
                                                # bound_norm=False
                                                )
        
        self.apply(self.weight_init)

