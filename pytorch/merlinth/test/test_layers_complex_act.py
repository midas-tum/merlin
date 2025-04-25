import unittest
import torch
import numpy as np
import merlinth
from merlinth.layers.complex_act import *

# class TestComplexStudentT(unittest.TestCase):  
#     def _test_gradient(self, shape):
#         # setup the hyper parameters for each test
#         dtype = torch.float64
#         nf=5

#         # perform a gradient check:
#         epsilon = 1e-6

#         # prefactors
#         a = 1.1
#         b = 1.1
#         c = 1.1

#         # transfer to torch
#         cuda = torch.device('cuda')
#         th_x = torch.randn(*shape).to(dtype=dtype, device=cuda)
#         th_v = torch.ones(nf).to(dtype=dtype, device=cuda)*0.1
#         th_v = th_v.view(1, -1, *[1 for d in range(th_x.ndim-2)])
#         th_w = torch.ones(nf).to(dtype=dtype, device=cuda)*0.1
#         th_w = th_w.view(1, -1, *[1 for d in range(th_x.ndim-2)])
#         th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
#         th_b = torch.tensor(b, requires_grad=True, dtype=th_v.dtype, device=cuda)
#         th_c = torch.tensor(c, requires_grad=True, dtype=th_w.dtype, device=cuda)
#         op = ComplexStudentT_fun()

#         # setup the model
#         loss = ComplexL2Loss()
#         compute_loss = lambda a, b, c: loss(op.apply(th_x*a, th_v*b, th_w*c))
#         th_loss = compute_loss(th_a, th_b, th_c)

#         # backpropagate the gradient
#         th_loss.backward()
#         grad_a = th_a.grad.cpu().numpy()
#         grad_b = th_b.grad.cpu().numpy()
#         grad_c = th_c.grad.cpu().numpy()

#         # numerical gradient w.r.t. the input
#         with torch.no_grad():
#             l_ap = compute_loss(th_a+epsilon, th_b, th_c).cpu().numpy()
#             l_an = compute_loss(th_a-epsilon, th_b, th_c).cpu().numpy()
#             grad_a_num = (l_ap - l_an) / (2 * epsilon)

#         print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
#             grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
#         #self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

#         # numerical gradient w.r.t. the weights
#         with torch.no_grad():
#             l_bp = compute_loss(th_a, th_b+epsilon, th_c).cpu().numpy()
#             l_bn = compute_loss(th_a, th_b-epsilon, th_c).cpu().numpy()
#             grad_b_num = (l_bp - l_bn) / (2 * epsilon)

#         print("grad_v: {:.7f} num_grad_v {:.7f} success: {}".format(
#             grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
#         self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)

#         # numerical gradient w.r.t. the weights
#         with torch.no_grad():
#             l_cp = compute_loss(th_a, th_b, th_c+epsilon).cpu().numpy()
#             l_cn = compute_loss(th_a, th_b, th_c-epsilon).cpu().numpy()
#             grad_c_num = (l_cp - l_cn) / (2 * epsilon)

#         print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
#             grad_c, grad_c_num, np.abs(grad_c - grad_c_num) < 1e-4))
#         self.assertTrue(np.abs(grad_c - grad_c_num) < 1e-4)

#     def test_gradient1(self):
#         self._test_gradient((2,5,10,2))
#     def test_gradient2(self):
#         self._test_gradient((2,5,10,2))

                
# class TestModReLU(unittest.TestCase):   
#     def _test_gradient(self, shape):
#         # setup the hyper parameters for each test
#         dtype = torch.float64
#         nf=5

#         # perform a gradient check:
#         epsilon = 1e-6

#         # prefactors
#         a = 1.1
#         b = 1.1

#         # transfer to torch
#         cuda = torch.device('cuda')
#         th_x = torch.randn(*shape).to(dtype=dtype, device=cuda)
#         th_w = torch.randn(nf).to(dtype=dtype, device=cuda)
#         th_w = th_w.view(1, -1, *[1 for d in range(th_x.ndim-2)])

#         th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
#         th_b = torch.tensor(b, requires_grad=True, dtype=th_x.dtype, device=cuda)

#         op = ComplexModReLU_fun()
#         loss = ComplexL2Loss()
#         # setup the model
#         compute_loss = lambda a, b: loss(op.apply(th_x*a, th_w*b, 1e-12))
#         th_loss = compute_loss(th_a, th_b)

#         # backpropagate the gradient
#         th_loss.backward()
#         grad_a = th_a.grad.cpu().numpy()
#         grad_b = th_b.grad.cpu().numpy()

#         # numerical gradient w.r.t. the input
#         with torch.no_grad():
#             l_ap = compute_loss(th_a+epsilon, th_b).cpu().numpy()
#             l_an = compute_loss(th_a-epsilon, th_b).cpu().numpy()
#             grad_a_num = (l_ap - l_an) / (2 * epsilon)

#         print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
#             grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
#         self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

#         with torch.no_grad():
#             l_bp = compute_loss(th_a, th_b+epsilon).cpu().numpy()
#             l_bn = compute_loss(th_a, th_b-epsilon).cpu().numpy()
#             grad_b_num = (l_bp - l_bn) / (2 * epsilon)

#         print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
#             grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
#         self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)

#     def test_gradient1(self):
#         self._test_gradient((2,5,2))
#     def test_gradient2(self):
#         self._test_gradient((2,5,10,2))

class TestActivation(unittest.TestCase):   
    def _test(self, act, args, shape):
        model = act(**args)
        x = merlinth.random_normal_complex(shape)
        Kx = model(x)
    
    def test_cReLU(self):
        self._test(cReLU, {}, [5, 32])

    def test_cPReLU(self):
        self._test(cPReLU, {'num_parameters': 32, 'alpha_init':0.1, }, [5, 32])

    def test_ModReLU(self):
        self._test(ModReLU, {'num_parameters': 32, 'bias_init':0.1, 'requires_grad':True}, [5, 32])

    def test_Cardioid(self):
        self._test(Cardioid, {'num_parameters': 32, 'bias_init':0.1, 'requires_grad':True}, [5, 32])

    def test_ModPReLU(self):
        self._test(ModPReLU, {'num_parameters': 32, 'alpha_init':0.1, 'bias_init':0.01, 'requires_grad':True}, [5, 32])

    def test_cStudentT(self):
        self._test(cStudentT, {'num_parameters': 32, 'alpha_init':0.1, 'requires_grad':True}, [5, 32])

    def test_ModStudentT(self):
        self._test(ModStudentT, {'num_parameters': 32, 'alpha_init':0.1, 'beta_init':0.01, 'requires_grad':True}, [5, 32])


class TestActivation2(unittest.TestCase):   
    def _test(self, act, args, shape):
        model = act(**args)
        x = merlinth.random_normal_complex(shape)
        x.requires_grad_(True)
        fx, dfx, dfxH = model(x)
        loss = 0.5 * torch.sum(torch.conj(fx) * fx)
        loss.backward()
        grad_x = x.grad
        x_autograd = grad_x.numpy()

        zH = fx
        z = torch.conj(zH)
        fprimex = z * dfxH + zH * torch.conj(dfx)
        x_bwd = fprimex.detach().numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test_ModStudentT2(self):
        self._test(ModStudentT2, {'num_parameters' : 32, 'alpha_init':0.1, 'beta_init':0.01, 'requires_grad':True}, [5, 32])

    def test_cStudentT2(self):
        self._test(cStudentT2, {'num_parameters' : 32, 'alpha_init' : 0.1, 'requires_grad':True}, [5, 32])

    def test_Cardioid2(self):
        self._test(Cardioid2, {'num_parameters' : 32, 'bias_init' : 0.1, 'requires_grad' : True}, [5, 32])


if __name__ == "__main__":
    unittest.main()