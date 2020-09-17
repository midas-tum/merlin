import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest
from .complex_reg_2dt import *
from .complex_conv3d import *
from .complex_init import *
from .complex_regularizer import *
from .complex_norm import get_normalization
from .complex_pool import *

__all__ = ['ComplexResidualBlock3d', 'ComplexResidualBlock2dt', 'ComplexResidualBlockSplitFast']

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

class TestComplexResidualBlock3d(unittest.TestCase):
    def _test_unet(self, depth, height, width, nf, nl,activation):
        x = torch.randn(1, nf, depth, height, width, 2).cuda()
        model =  ComplexResidualBlock3d(
            nf, nl,
            bias=True,
            activation=activation).cuda()
        count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        #print(model)
        print(f'Num parameters: {count}')
        y = model(x)
    
    def test1(self):
        self._test_unet(10, 180, 180, 16, 1, 'cPReLU')
    def test2(self):
        self._test_unet(10, 180, 180, 16, 2, 'ModReLU')

class TestComplexResidualBlock2dt(unittest.TestCase):
    def _test_unet(self, depth, height, width, nf, nl,activation):
        x = torch.randn(1, nf, depth, height, width, 2).cuda()
        model =  ComplexResidualBlock2dt(
            nf, nl,
            bias=True,
            activation=activation).cuda()
        count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        #print(model)
        print(f'Num parameters: {count}')
        y = model(x)
    
    def test1(self):
        self._test_unet(10, 180, 180, 16, 1, 'cPReLU')
    def test2(self):
        self._test_unet(10, 180, 180, 16, 2, 'ModReLU')

class TestComplexResidualBlockFast(unittest.TestCase):
    def _test_unet(self, depth, height, width, nf, nl,activation):
        x = torch.randn(1, nf, depth, height, width, 2).cuda()
        model =  ComplexResidualBlockSplitFast(
            nf, nl,
            bias=True,
            activation=activation).cuda()
        count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        print(f'Num parameters: {count}')
        y = model(x)
    
    def test1(self):
        self._test_unet(10, 180, 180, 16, 1, 'cPReLU')
    def test2(self):
        self._test_unet(10, 180, 180, 16, 2, 'ModReLU')

if __name__ == "__main__":
    unittest.test()
