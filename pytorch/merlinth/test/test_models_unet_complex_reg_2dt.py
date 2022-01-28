import unittest
import torch
import numpy as np

from merlinth.layers.convolutional.complex_padconv import (
    ComplexPadConv3D,
    ComplexPadConvScale3D,
    ComplexPadConvScaleTranspose3D
)
from merlinth.layers.complex_init import *
from merlinth.models.unet_complex_reg_2dt import (
    ComplexSplitFast,
    ComplexConvBlock3d,
)

class SplitFastTest(unittest.TestCase):
    def _test(self, ksp, kst, stride, activation):
        nBatch = 3
        M = 12
        N = 12
        D = 6
        nf_in = 5
        nf_out = 10
        
        model = ComplexSplitFast(nf_in, nf_out // 2, nf_out, kernel_size_sp=ksp, kernel_size_t=kst, stride=stride, activation=activation).cuda()
        
        x = torch.randn(nBatch, nf_in, D, M, N, dtype=torch.complex64).cuda()
        Kx = model(x)

        count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        print(model)
        print(f'Num parameters: {count}')

    def test1(self):
        self._test(3,3,2,'cPReLU')

    def test2(self):
        self._test(5,3,1,'ModReLU')

class InitTest(unittest.TestCase):
    def _test_complex_init(self, ksp, kst, nf_in, nf_out, mode, activation, ortho):
        nBatch = 1
        M = 12
        N = 12
        D = 6

        #model = ComplexConvBlock2dt(nf_in, nf_out // 2, nf_out, kernel_size_sp=ksp, kernel_size_t=kst, stride=1, activation=activation).cuda()
        model = ComplexConvBlock3d(nf_in, nf_out, kernel_size_sp=ksp, kernel_size_t=kst, stride=1, activation=activation).cuda()

        def weight_init(module):
            if isinstance(module, ComplexPadConv3D) \
            or isinstance(module, ComplexPadConvScale3D) \
            or isinstance(module, ComplexPadConvScaleTranspose3D):
                #print('weight init ortho', ortho, module)
                #print('before: ', module.weight.min(), module.weight.max())
                if ortho:
                    complex_independent_filters_init(module.weight, mode=mode)
                else:
                    complex_init(module.weight, mode=mode)
                weight = torch.view_as_real(module.weight.data)
                print('weight', mode, ortho, weight.min(), weight.max())
        model.apply(weight_init)

        x = torch.randn(nBatch, nf_in, D, M, N, dtype=torch.complex64).cuda()
        model.cuda()
        Kx = model(x)
        print('Kx: ', Kx.abs().min(), Kx.abs().max())

    def test_complex_init_1(self):
        self._test_complex_init(3, 3, 16, 32, 'he', 'ModReLU', False)
    def test_complex_init_2(self):
        self._test_complex_init(3, 3, 16, 32, 'he', 'cPReLU', False)
    def test_complex_init_3(self):
        self._test_complex_init(3, 3, 16, 32, 'glorot', 'ModReLU', False)
    def test_complex_init_4(self):
        self._test_complex_init(3, 3, 16, 32, 'glorot', 'cPReLU', False)
    def test_complex_init_1_ortho(self):
        self._test_complex_init(3, 3, 16, 32, 'he', 'ModReLU', True)
    def test_complex_init_2_ortho(self):
        self._test_complex_init(3, 3, 16, 32, 'he', 'cPReLU', True)
    def test_complex_init_3_ortho(self):
        self._test_complex_init(3, 3, 16, 32, 'glorot', 'ModReLU', True)
    def test_complex_init_4_ortho(self):
        self._test_complex_init(3, 3, 16, 32, 'glorot', 'cPReLU', True)
    def test_complex_init_5_ortho(self):
        self._test_complex_init(3, 3, 16, 32, 'glorot', 'cReLU', True)

if __name__ == "__main__":
    unittest.test()

