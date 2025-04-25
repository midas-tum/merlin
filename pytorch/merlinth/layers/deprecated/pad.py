import torch
import optoth.pad

def real_pad2d(x, pad, mode='symmetric'):
    return optoth.pad.pad2d(x, pad, mode=mode)

def real_pad2d_transpose(x, pad, mode='symmetric'):
    return optoth.pad.pad2d_transpose(x, pad, mode=mode)

def real_pad3d(x, pad, mode='symmetric'):
    return optoth.pad.pad3d(x, pad, mode=mode)

def real_pad3d_transpose(x, pad, mode='symmetric'):
    return optoth.pad.pad3d_transpose(x, pad, mode=mode)

def complex_pad2d(x, pad, mode='symmetric'):
    xp_re = optoth.pad.pad2d(x[...,0].contiguous(), pad, mode=mode)
    xp_im = optoth.pad.pad2d(x[...,1].contiguous(), pad, mode=mode)

    new_shape = list(xp_re.shape)
    new_shape.append(2)
    xp = torch.zeros(*new_shape, device=x.device, dtype=x.dtype)
    xp[...,0] = xp_re
    xp[...,1] = xp_im

    return xp

def complex_pad2d_transpose(x, pad, mode='symmetric'):
    xp_re = optoth.pad.pad2d_transpose(x[...,0].contiguous(), pad, mode=mode)
    xp_im = optoth.pad.pad2d_transpose(x[...,1].contiguous(), pad, mode=mode)

    new_shape = list(xp_re.shape)
    new_shape.append(2)
    xp = torch.zeros(*new_shape, device=x.device, dtype=x.dtype)
    xp[...,0] = xp_re
    xp[...,1] = xp_im

    return xp

def complex_pad3d(x, pad, mode='symmetric'):
    xp_re = optoth.pad.pad3d(x[...,0].contiguous(), pad, mode=mode)
    xp_im = optoth.pad.pad3d(x[...,1].contiguous(), pad, mode=mode)

    new_shape = list(xp_re.shape)
    new_shape.append(2)
    xp = torch.zeros(*new_shape, device=x.device, dtype=x.dtype)
    xp[...,0] = xp_re
    xp[...,1] = xp_im

    return xp

def complex_pad3d_transpose(x, pad, mode='symmetric'):
    xp_re = optoth.pad.pad3d_transpose(x[...,0].contiguous(), pad, mode=mode)
    xp_im = optoth.pad.pad3d_transpose(x[...,1].contiguous(), pad, mode=mode)

    new_shape = list(xp_re.shape)
    new_shape.append(2)
    xp = torch.zeros(*new_shape, device=x.device, dtype=x.dtype)
    xp[...,0] = xp_re
    xp[...,1] = xp_im

    return xp


import unittest
import numpy as np

class Testpad3dFunction(unittest.TestCase):
    
    def _test_adjointness(dtype, mode):                   
        # setup the hyper parameters for each test
        S, C, D, M, N =4, 3, 16, 32, 32

        pad = [3,3,2,2,1,1]

        # transfer to torch
        cuda = torch.device('cuda')
        x = torch.randn(S, C, D, M, N, dtype=dtype, device=cuda).requires_grad_(True)
        p = torch.randn(S, C, D+pad[4]+pad[5], M+pad[2]+pad[3], N+pad[0]+pad[1], dtype=dtype, device=cuda).requires_grad_(True)

        Ax = optoth.pad.pad3d(x, pad, mode)
        ATp = torch.autograd.grad(Ax, x, p)[0]

        lhs = (Ax * p).sum().item()
        rhs = (x * ATp).sum().item()

        print(Ax.shape, x.shape)

        print('forward: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

        Ap = optoth.pad.pad3d_transpose(p, pad, mode)
        ATx = torch.autograd.grad(Ap, p, x)[0]

        lhs = (Ap * x).sum().item()
        rhs = (p * ATx).sum().item()

        print('adjoint: dtype={} diff: {}'.format(dtype, np.abs(lhs - rhs)))
        self.assertTrue(np.abs(lhs - rhs) < 1e-3)

    def test_float_symmetric(self):
        self._test_adjointness(torch.float32, 'symmetric')

    def test_float_replicate(self):
        self._test_adjointness(torch.float32, 'replicate')
    
    def test_float_reflect(self):
        self._test_adjointness(torch.float32, 'reflect')
    
    def test_double_symmetric(self):
        self._test_adjointness(torch.float64, 'symmetric')

    def test_double_replicate(self):
        self._test_adjointness(torch.float64, 'replicate')
    
    def test_double_reflect(self):
        self._test_adjointness(torch.float64, 'reflect')

if __name__ == "__main__":
    unittest.main()
