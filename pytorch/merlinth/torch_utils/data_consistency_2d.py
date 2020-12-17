import torch
from merlinth.mytorch.complex import complex_mult, complex_mult_conj
from merlinth.mytorch.fft import fft2, ifft2, fft2c, ifft2c
import unittest
from .complex_cg import CGClass
import unittest
import numpy as np

class MulticoilForwardOp(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = fft2c
        else:
            self.fft2 = fft2

    def forward(self, image, mask, smaps):
        # img [N]
        img_pad = torch.unsqueeze(torch.squeeze(image, -2), -4)
        kspace = self.fft2(complex_mult(img_pad.expand_as(smaps), smaps)) * torch.unsqueeze(mask, -1)
        return kspace

class MulticoilAdjointOp(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.ifft2 = ifft2c
        else:
            self.ifft2 = ifft2

    def forward(self, kspace, mask, smaps):
        th_img = torch.sum(complex_mult_conj(self.ifft2(kspace * torch.unsqueeze(mask, -1)), smaps), dim=(-4))
        return th_img.unsqueeze_(-2)

class ForwardOp(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = fft2c
        else:
            self.fft2 = fft2

    def forward(self, image, mask):
        return self.fft2(torch.squeeze(image, -2)) * torch.unsqueeze(mask, -1)

class AdjointOp(torch.nn.Module):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.ifft2 = ifft2c
        else:
            self.ifft2 = ifft2

    def forward(self, kspace, mask):
        return torch.unsqueeze(self.ifft2(kspace * torch.unsqueeze(mask, -1)), -2)

class DCGD2D(torch.nn.Module):
    def __init__(self, config, center=False, multicoil=True, name='dc-gd'):
        super().__init__()
        if multicoil:
            self.A = MulticoilForwardOp(center)
            self.AH = MulticoilAdjointOp(center)
        else:
            self.A = ForwardOp(center)
            self.AH = AdjointOp(center)

        self.train_scale = config['lambda']['train_scale'] if 'train_scale' in config['lambda'] else 1
        self._weight = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.reset_scalar(self._weight, **config['lambda'])
    
    def reset_scalar(self, scalar, init=1., min=0, max=1000, requires_grad=True, **kwargs):
        scalar.data = torch.tensor(init, dtype=scalar.dtype)
        # add a positivity constraint
        scalar.proj = lambda: scalar.data.clamp_(min, max)
        scalar.requires_grad = requires_grad

    @property
    def weight(self):
        return self._weight * self.train_scale

    def forward(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        stepsize = self.weight * scale
        return x - stepsize * self.AH(self.A(x, *constants) - y, *constants)

    def __repr__(self):
        return f'DCGD2D(lambda_init={self._weight.item():.4g}, train_scale={self.train_scale})'

class DCPM2D(torch.nn.Module):
    def __init__(self, config, center=False, multicoil=True, name='dc-pm', **kwargs):
        super().__init__()
        if multicoil:
            A = MulticoilForwardOp(center)
            AH = MulticoilAdjointOp(center)
            max_iter = kwargs.get('max_iter', 10)
            tol = kwargs.get('tol', 1e-10)
            self.prox = CGClass(A, AH, max_iter=max_iter, tol=tol)
        else:
            raise ValueError

        self.train_scale = config['lambda']['train_scale'] if 'train_scale' in config['lambda'] else 1
        self._weight = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.reset_scalar(self._weight, **config['lambda'])
    
    def reset_scalar(self, scalar, init=1., min=0, max=1000, requires_grad=True, **kwargs):
        scalar.data = torch.tensor(init, dtype=scalar.dtype)
        # add a positivity constraint
        scalar.proj = lambda: scalar.data.clamp_(min, max)
        scalar.requires_grad = requires_grad

    @property
    def weight(self):
        return self._weight * self.train_scale

    def forward(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        lambdaa = 1.0 / torch.maximum(self.weight * scale, torch.tensor(1e-9))
        return self.prox(lambdaa, x, y, *constants)

    def __repr__(self):
        return f'DCGD2D(lambda_init={self._weight.item():.4g}, train_scale={self.train_scale})'

class TestMulticoil(unittest.TestCase):
    def test(self):
        N = 4
        nFE = 128
        nPE = 128
        nCh = 5

        x = torch.randn(N, nFE, nPE, 1, 2)
        smaps = torch.randn(N, nCh, nFE, nPE, 2)
        mask = torch.randn(N, 1, 1, nPE)
        
        A = MulticoilForwardOp(center=True)
        AH = MulticoilAdjointOp(center=True)

        Ax = A(x, mask, smaps)

        AHAx = AH(Ax, mask, smaps)
        self.assertTrue(AHAx.shape == x.shape)

class TestSinglecoil(unittest.TestCase):
    def test(self):
        N = 4
        nFE = 128
        nPE = 128

        x = torch.randn(N, nFE, nPE, 1, 2)
        mask = torch.randn(N, 1, nPE)
        
        A = ForwardOp(center=True)
        AH = AdjointOp(center=True)

        Ax = A(x, mask)

        AHAx = AH(Ax, mask)
        self.assertTrue(AHAx.shape == x.shape)


class CgTest(unittest.TestCase):
    def testcg(self):
        config = {'lambda' : {'init' : 1.0}}
        dc = DCPM2D(config, center=True, multicoil=True, max_iter=100, tol=1e-18)

        shape=(5,10,10,1)
        kshape=(5,3,10,10)
        x = torch.randn(*shape, 2, dtype=torch.float64)
        y = torch.randn(*kshape, 2, dtype=torch.float64)
        mask = torch.ones(*kshape, dtype=torch.float64)
        smaps = torch.randn(*kshape, 2, dtype=torch.float64)

        th_a = torch.tensor(1.1, requires_grad=True, dtype=torch.float64)
        th_b = torch.tensor(1.1, requires_grad=True, dtype=torch.float64)

        # perform a gradient check:
        epsilon = 1e-5

        def compute_loss(a, b):
            arg = dc([x*a, y, mask, smaps], scale=1/b) # take 1/b
            return 0.5 * torch.sum(arg * arg)

        th_loss = compute_loss(th_a, th_b)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()
        grad_b = th_b.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon, th_b).cpu().numpy()
            l_an = compute_loss(th_a-epsilon, th_b).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

        # numerical gradient w.r.t. the weights
        with torch.no_grad():
            l_bp = compute_loss(th_a, th_b+epsilon).cpu().numpy()
            l_bn = compute_loss(th_a, th_b-epsilon).cpu().numpy()
            grad_b_num = (l_bp - l_bn) / (2 * epsilon)

        print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
            grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
        self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)
        
if __name__ == "__main__":
    TestMulticoil().test()
