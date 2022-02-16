import unittest
import torch
import numpy as np
from merlinth.models.unet_complex_residual_block import (
    ComplexResidualBlock3d,
    ComplexResidualBlock2dt,
    ComplexResidualBlockSplitFast
)

class TestComplexResidualBlock3d(unittest.TestCase):
    def _test_unet(self, depth, height, width, nl, activation):
        x = torch.randn(1, nl, depth, height, width, dtype=torch.complex64).cuda()
        model =  ComplexResidualBlock3d(
            nl, nl,
            bias=True,
            activation=activation).cuda()
        count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        #print(model)
        print(f'Num parameters: {count}')
        y = model(x)
    
    def test1(self):
        self._test_unet(10, 180, 180, 16, 'cPReLU')
    def test2(self):
        self._test_unet(10, 180, 180, 16, 'ModReLU')

class TestComplexResidualBlock2dt(unittest.TestCase):
    def _test_unet(self, depth, height, width, nl, activation):
        x = torch.randn(1, nl, depth, height, width, dtype=torch.complex64).cuda()
        model =  ComplexResidualBlock2dt(
            nl, nl//2, nl,
            bias=True,
            activation=activation).cuda()
        count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        #print(model)
        print(f'Num parameters: {count}')
        y = model(x)
    
    def test1(self):
        self._test_unet(10, 180, 180, 16, 'cPReLU')
    def test2(self):
        self._test_unet(10, 180, 180, 16, 'ModReLU')

class TestComplexResidualBlockFast(unittest.TestCase):
    def _test_unet(self, depth, height, width, nl, activation):
        x = torch.randn(1, nl, depth, height, width, dtype=torch.complex64).cuda()
        model =  ComplexResidualBlockSplitFast(
            nl, nl//2, nl,
            bias=True,
            activation=activation).cuda()
        count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        print(f'Num parameters: {count}')
        y = model(x)
    
    def test1(self):
        self._test_unet(10, 180, 180, 16, 'cPReLU')
    def test2(self):
        self._test_unet(10, 180, 180, 16, 'ModReLU')

if __name__ == "__main__":
    unittest.main()
