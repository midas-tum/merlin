import unittest
import torch
import numpy as np
from merlinth.models.unet_residual import (
    ResidualUnetModelFast,
    ResidualUnetModel3d,
    ResidualUnetModel2dt
)

class TestUnetFast(unittest.TestCase):
    def _test_unet(self, depth, height, width, nf, nl, multiplier, activation, local_residual):
        x = torch.randn(5, 1, depth, height, width, dtype=torch.complex64).cuda()
        model =  ResidualUnetModelFast(
            1, 1, nf, nl,
            local_residual=local_residual,
            bias=True,
            multiplier=multiplier,
            activation=activation).cuda()
        print(model)
        count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        #print(model)
        print(f'Num parameters: {count}')
        y = model(x)

    @unittest.expectedFailure
    def test1(self):
        self._test_unet(16, 96, 96, 8, 2, 1, 'cPReLU', False)

    @unittest.expectedFailure
    def test2(self):
        self._test_unet(16, 96, 96, 8, 2, 1, 'ModReLU', True)

class TestUnet3d(unittest.TestCase):
    def _test_unet(self, depth, height, width, nf, nl, multiplier, activation, local_residual):
        x = torch.randn(5, 1, depth, height, width, dtype=torch.complex64).cuda()
        model =  ResidualUnetModel3d(
            1, 1, nf, nl,
            local_residual=local_residual,
            bias=True,
            multiplier=multiplier,
            activation=activation).cuda()
        print(model)
        count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        #print(model)
        print(f'Num parameters: {count}')
        y = model(x)

    @unittest.expectedFailure
    def test1(self):
        self._test_unet(16, 96, 96, 16, 2, 1, 'cPReLU', False)

    @unittest.expectedFailure
    def test2(self):
        self._test_unet(16, 96, 96, 16, 2, 1, 'ModReLU', True)

class TestUnet2dt(unittest.TestCase):
    def _test_unet(self, depth, height, width, nf, nl, multiplier, activation, local_residual):
        x = torch.randn(5, 1, depth, height, width, dtype=torch.complex64).cuda()
        model =  ResidualUnetModel2dt(
            1, 1, nf, nl,
            local_residual=local_residual,
            bias=True,
            multiplier=multiplier,
            activation=activation).cuda()
        print(model)
        count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        #print(model)
        print(f'Num parameters: {count}')
        y = model(x)
    
    @unittest.expectedFailure
    def test1(self):
        self._test_unet(16, 96, 96, 16, 2, 1, 'cPReLU', False)

    @unittest.expectedFailure
    def test2(self):
        self._test_unet(16, 96, 96, 16, 2, 1, 'ModReLU', True)

if __name__ == "__main__":
    unittest.main()
