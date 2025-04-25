import torch
import torch.nn as nn
from merlinth.models.unet_complex_reg_2dt import (
    ComplexConvBlock2dt, 
    ComplexConvBlock3d,
    ComplexSplitFast)
from merlinth.layers.complex_init import *
from merlinth.layers.complex_norm import get_normalization
from merlinth.layers.complex_maxpool import MagnitudeMaxPool3D

__all__ = ['ComplexResidualBlock3d',
           'ComplexResidualBlock2dt',
           'ComplexResidualBlockSplitFast']

class ComplexResidualBlock(nn.Module):
    """
    Residual Unet
    """

    def __init__(
            self,
            channels,
            num_layers=1,
            bias=True,
            **kwargs,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input to the U-Net model.
            out_channels (int): Number of channels in the output to the U-Net model.
            channels (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            global_residual (bool): whether to add global residual or not
            use_instance_norm (bool): whether to use instance norm or not
            use_dropout (bool): use drop out
            drop_prob (float): Dropout probability.
        """
        super(ComplexResidualBlock, self).__init__()

        self.num_layers = num_layers
               
        self.gain = kwargs.pop('gain', 1.0)
        self.mode = kwargs.pop('mode', 'glorot')
        self.ortho = kwargs.pop('ortho', False)
        #print('mode=', self.mode, 'ortho=', self.ortho)


    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_channels, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_channels, height, width]
        """       
        return  input + self.residual(input)

class ComplexResidualBlock3d(ComplexResidualBlock):
    """
    Residual Unet
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            num_layers=1,
            bias=True,
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
            num_layers (int): Number of down-sampling and up-sampling layers.
        """
        super(ComplexResidualBlock3d, self).__init__(
            num_layers,
            bias,
            normalization=normalization,
            **kwargs,)

        # residual block
        self.residual = torch.nn.Sequential(*[
                        ComplexConvBlock3d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            bias=bias,
                                            stride=1,
                                            activation=activation,
                                            normalization=normalization,
                                    kernel_size_sp=kernel_size_sp,
                                    kernel_size_t=kernel_size_t) for l in range(num_layers)])

class ComplexResidualBlock2dt(ComplexResidualBlock):
    """
    Residual Unet
    """

    def __init__(
            self,
            in_channels,
            inter_channels,
            out_channels,
            num_layers=1,
            bias=True,
            activation = 'cPReLU',
            normalization = 'no',
            kernel_size_sp = 3,
            kernel_size_t = 3,
            activation_xy=True,
            **kwargs,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input to the U-Net model.
            out_channels (int): Number of channels in the output to the U-Net model.
            channels (int): Number of output channels of the first convolution layer.
            num_layers (int): Number of down-sampling and up-sampling layers.
        """
        super(ComplexResidualBlock2dt, self).__init__(
            num_layers,
            bias,
            normalization=normalization,
            **kwargs,)

        # residual block
        self.residual = torch.nn.Sequential(*[
                        ComplexConvBlock2dt(in_channels=in_channels,
                                        inter_channels=inter_channels,
                                        out_channels=out_channels,
                                        bias=bias,
                                        activation=activation,
                                        activation_xy=activation_xy,
                                        normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t) for l in range(num_layers)])

class ComplexResidualBlockSplitFast(ComplexResidualBlock):
    """
    Residual Unet
    """

    def __init__(
            self,
            in_channels,
            inter_channels,
            out_channels,
            num_layers=1,
            bias=True,
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
            num_layers (int): Number of down-sampling and up-sampling layers.
        """
        super(ComplexResidualBlockSplitFast, self).__init__(
            num_layers,
            bias,
            normalization=normalization,
            **kwargs,)

        # residual block
        self.residual = torch.nn.Sequential(*[
                        ComplexSplitFast(in_channels=in_channels,
                                           inter_channels=inter_channels,
                                           out_channels=out_channels,
                                           bias=bias,
                                           activation=activation,
                                           normalization=normalization,
                                     kernel_size_sp=kernel_size_sp,
                                     kernel_size_t=kernel_size_t) for l in range(num_layers)])

