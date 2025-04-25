import unittest
import torch
import numpy as np
from merlinth.layers.complex_norm import (
    ComplexInstanceNormalization2d,
    ComplexInstanceNormalization3d,
    ComplexInstanceNorm_fun)

class ComplexConv2dTest(unittest.TestCase):
    def test_conv2d_complex(self):
        nBatch = 5
        M = 320
        N = 320
        nf_in = 2
        
        model = ComplexInstanceNormalization2d().cuda()
        
        x = torch.randn(nBatch, nf_in, M, N, 2).cuda()
        xn = torch.view_as_real(model(torch.view_as_complex(x)))
        axes=tuple(range(2, x.dim()-1))
        print(xn.mean(axes).min(), xn.mean(axes).max())
        var = x.var(unbiased=False, dim=axes )
        uu = var[...,0]
        vv = var[...,1]
        uv = (x[...,0] * x[...,1]).mean(dim=axes)
        print(uu.min(), uu.max(), vv.min(), vv.max(), uv.min(), uv.max())
        
class ComplexConv3dTest(unittest.TestCase):
    def test_conv2d_complex(self):
        nBatch = 5
        M = 320
        N = 320
        D = 10
        nf_in = 2
        
        model = ComplexInstanceNormalization3d().cuda()
        
        x = torch.randn(nBatch, nf_in, D, M, N, 2).cuda()
        xn = model(torch.view_as_complex(x))

class TestComplexInstanceNorm(unittest.TestCase):   
    def _test_gradient(self, shape):
        # setup the hyper parameters for each test
        #dtype = torch.float64
        dtype = torch.complex128    # TODO: unknown issues: change to complex64 the test will fail.

        # perform a gradient check:
        epsilon = 1e-6

        # prefactors
        a = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x =  torch.randn(*shape).to(dtype=dtype, device=cuda)
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        op = ComplexInstanceNorm_fun()

        # setup the model
        compute_loss = lambda a: 0.5 * torch.sum(op.apply(th_x*a, 1e-5)**2)
        th_loss = compute_loss(th_a)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon).cpu().numpy()
            l_an = compute_loss(th_a-epsilon).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

    def test_gradient1(self):
        self._test_gradient((2,5,5,5,5))
    def test_gradient2(self):
        self._test_gradient((2,5,10,10))

if __name__ == "__main__":
    unittest.main()
