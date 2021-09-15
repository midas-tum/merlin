import torch
from merlinth.complex import complex_abs
import unittest
from optoth.activations import TrainableActivation
import numpy as np
from .complex_loss import *

__all__ = ['cPReLU',
            'cReLU',
           'ModReLU',
           'ModPReLU',
           'ComplexStudentT2',
           'ComplexStudentT',
           'ComplexTrainablePolarActivation',
           'ComplexTrainablePolarActivationBias',
           'ComplexTrainableMagnitudeActivation',
           'ComplexTrainableMagnitudeActivationBias']

class cPReLU(torch.nn.Module):
    def __init__(self, num_parameters):
        super(cPReLU, self).__init__()
        self.num_parameters = num_parameters
        self.prelu_re = torch.nn.PReLU(num_parameters=num_parameters, init=0.1)
        self.prelu_im = torch.nn.PReLU(num_parameters=num_parameters, init=0.1)

    def forward(self, z):
        assert z.shape[-1] == 2
        act_re = self.prelu_re(z[...,0]).unsqueeze_(-1)
        act_im = self.prelu_im(z[...,1]).unsqueeze_(-1)
        return torch.cat([act_re, act_im], -1)

class cReLU(torch.nn.Module):
    def __init__(self, num_parameters):
        super(cReLU, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, z):
        assert z.shape[-1] == 2
        return self.relu(z)

class Identity(torch.nn.Module):
    def forward(self, z):
        return z

# class ModReLU(torch.nn.Module):
#     """Arjovsky et al. Unitary evolution recurrent neural networks. In ICML, 2016"""
#     def __init__(self, num_parameters):
#         super(ModReLU, self).__init__()
#         self.relu = torch.nn.ReLU()
#         self.num_parameters = num_parameters
#         self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(num_parameters)))
    
#     def forward(self, z):
#         eps=1e-9
#         magn = complex_abs(z, eps=eps, keepdim=True)
#         bias = self.bias.view(1, -1, *[1 for _ in range(z.ndim - 2)])
#         return (self.relu(magn + bias) * z) / magn

#     def extra_repr(self):
#         s = "num_parameters={num_parameters}"
#         return s.format(**self.__dict__)

class ComplexModReLU_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, bias, eps):
        ctx.save_for_backward(z)
        ctx.bias = bias
        ctx.eps = eps
        magn = complex_abs(z, eps=eps, keepdim=True)
        return torch.clamp(magn + bias, min=0) * z / magn

    @staticmethod
    def backward(ctx, grad_in):
        z = ctx.saved_tensors[0]
        bias = ctx.bias
        eps = ctx.eps

        magn = complex_abs(z, eps=eps, keepdim=True)

        grad_inH = complex_conj(grad_in)
        dz = 1 + bias / (2 * magn)
        dz = torch.stack([dz[...,0], torch.zeros(*dz.shape[:-1], dtype=z.dtype, device=z.device)], dim=-1)

        dzH = - bias * complex_mult(z, z) / (2 * magn**3)
        grad_out = complex_mult(grad_in, dz) + complex_mult(grad_inH, dzH)
       
        re = torch.where((magn + bias)[...,0] > 0, grad_out[...,0], torch.zeros(*grad_out.shape[:-1], dtype=grad_out.dtype, device=grad_out.device) )
        im = torch.where((magn + bias)[...,0] > 0, grad_out[...,1], torch.zeros(*grad_out.shape[:-1], dtype=grad_out.dtype, device=grad_out.device) )
        grad_in = torch.cat([re.unsqueeze_(-1), im.unsqueeze_(-1)], -1)
        
        dbias = z/magn
        dbiasH = complex_conj(dbias)
        grad_bias = complex_mult(grad_inH, dbias) + complex_mult(grad_in, dbiasH)
        #grad_bias = grad_bias[...,0].unsqueeze_(-1)
        grad_bias = grad_bias.sum(-1, keepdim=True)
        grad_bias = torch.where(magn + bias > 0, grad_bias, torch.zeros(*magn.shape, dtype=magn.dtype, device=magn.device) )
        dim=(0,*[l for l in range(2,z.ndim)])
        grad_bias = grad_bias.sum(dim=dim, keepdim=True)
        return grad_in, grad_bias * 0.5, None

class ModReLU(torch.nn.Module):
    """Arjovsky et al. Unitary evolution recurrent neural networks. In ICML, 2016"""
    def __init__(self, num_parameters):
        super(ModReLU, self).__init__()
        self.num_parameters = num_parameters
        self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(num_parameters)))
    
    def forward(self, z, eps=1e-9):
        bias = self.bias.view(1, -1, *[1 for _ in range(z.ndim - 2)])
        op = ComplexModReLU_fun()
        return op.apply(z, bias, eps)

    def extra_repr(self):
        s = "num_parameters={num_parameters}"
        return s.format(**self.__dict__)
class ModPReLU(torch.nn.Module):
    def __init__(self, num_parameters):
        super(ModPReLU, self).__init__()
        self.num_parameters = num_parameters
        self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(num_parameters)))
        self.register_parameter(name='slope', param=torch.nn.Parameter(torch.ones(num_parameters))*0.1)

    def forward(self, z, eps):
        bias = self.bias.view(1, -1, *[1 for _ in range(z.ndim - 2)])
        slope = self.slope.view(1, -1, *[1 for _ in range(z.ndim - 2)])
        op = ComplexModReLU_fun()
        return op.apply(z, bias, slope, eps)

    def extra_repr(self):
        s = "num_parameters={num_parameters}"
        return s.format(**self.__dict__)

class StudentT_fun2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        d = 1+alpha*x**2
        return torch.log(d)/(2*alpha), x/d

    @staticmethod
    def backward(ctx, grad_in1, grad_in2):
        x = ctx.saved_tensors[0]
        d = 1+ctx.alpha*x**2
        return (x/d) * grad_in1 + (1-ctx.alpha*x**2)/d**2 * grad_in2, None


class ComplexStudentT2(torch.nn.Module):
    def __init__(self,alpha):
        super(ComplexStudentT2, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        act_re, act_re_prime = StudentT_fun2().apply(x[...,0], self.alpha)
        act_im, act_im_prime = StudentT_fun2().apply(x[...,1], self.alpha)
        return torch.cat([act_re.unsqueeze_(-1), act_im.unsqueeze_(-1)], -1), \
               torch.cat([act_re_prime.unsqueeze_(-1), act_im_prime.unsqueeze_(-1)], -1)

class StudentT_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        d = 1+alpha*x**2
        return torch.log(d)/(2*alpha)

    @staticmethod
    def backward(ctx, grad_in):
        x = ctx.saved_tensors[0]
        d = 1+ctx.alpha*x**2
        return (x/d) * grad_in

class ComplexStudentT_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, beta):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.beta = beta

        magn = complex_abs(x, keepdim=True, eps=1e-9)
        norm = x / magn

        d = 1 + alpha * magn**2 # + beta
        fmagn = torch.log(d) / (2 * alpha) * beta

        return fmagn * norm

    @staticmethod
    def backward(ctx, grad_in):
        x = ctx.saved_tensors[0]
        alpha = ctx.alpha
        beta = ctx.beta

        axes = (0, *[d for d in range(2,x.ndim)])
        magn = complex_abs(x, keepdim=True, eps=1e-9)

        d = 1 + alpha * magn ** 2 # + beta

        dx = beta * (torch.log(d) / 2 + alpha * magn ** 2 / d) / (2 * alpha * magn)
        dx = torch.stack([dx[...,0], torch.zeros(*dx.shape[:-1], dtype=x.dtype, device=x.device)], dim=-1)
        dxH = beta * (complex_mult(x, x) / (2 * alpha) * (alpha / (d * magn) - torch.log(d) / (2 * magn ** 3)))

        grad_inH = complex_conj(grad_in)
        grad_out = complex_mult(grad_in, dx) + complex_mult(grad_inH, dxH)

        dalpha = ( (alpha * magn / d - torch.log(d) / magn) / (2 * alpha ** 2)) * x * beta
        grad_alpha = complex_mult(grad_inH, dalpha)+\
                     complex_mult(grad_in, complex_conj(dalpha))
        grad_alpha = grad_alpha.sum(dim=axes, keepdim=True) / 2

        dbeta = x * torch.log(d) / (2 * alpha * magn)
        grad_beta = complex_mult(grad_inH, dbeta)+\
                    complex_mult(grad_in, complex_conj(dbeta))
        grad_beta = grad_beta.sum(dim=axes, keepdim=True)/ 2

        return grad_out, grad_alpha, grad_beta


class ComplexStudentT(torch.nn.Module):
    def __init__(self, num_parameters):
        super(ComplexStudentT, self).__init__()
        self.register_parameter(name='alpha', param=torch.nn.Parameter(torch.ones(num_parameters)*0.1))
        self.register_parameter(name='beta', param=torch.nn.Parameter(torch.ones(num_parameters)))

        self.alpha.proj = lambda: self.alpha.data.clamp_(1e-4, 1000)
        self.beta.proj = lambda: self.beta.data.clamp_(0, 10)

    def forward(self, x):
        alpha = self.alpha.view(1, -1, *[1 for i in range(x.ndim-2)])
        beta = self.beta.view(1, -1, *[1 for i in range(x.ndim-2)])
        return ComplexStudentT_fun().apply(x, alpha, beta)

class ComplexTrainablePolarActivation(torch.nn.Module):
    def __init__(self, num_parameters):
        super(ComplexTrainablePolarActivation, self).__init__()

        config_f_abs ={
            'num_channels': num_parameters,
            'vmin': 0.0,
            'vmax': 3.0,
            'num_weights': 5,
            'base_type': 'linear',
            'init': 'linear',
            'init_scale': 1.0,
        }
        config_f_phi = {
            'num_channels': num_parameters,
            'vmin':  -np.pi,
            'vmax':  np.pi,
            'num_weights': 5,
            'base_type': 'linear',
            'init': 'linear',
            'init_scale': 1,
        }

        self.f_abs = TrainableActivation(**config_f_abs)
        self.f_phi = TrainableActivation(**config_f_phi)
        self.f_abs.weight.lrscale = 10
        self.f_phi.weight.lrscale = 10

    def forward(self, x):
        magn = self.f_abs(complex_abs(x, eps=1e-9) )
        angle = self.f_phi(complex_angle(x, eps=1e-9) )
        re = magn * torch.cos(angle)
        im = magn * torch.sin(angle)

        fx = torch.cat([re.unsqueeze_(-1), im.unsqueeze_(-1)], -1)

        return fx

class ComplexTrainableMagnitudeActivation(torch.nn.Module):
    def __init__(self, num_parameters):
        super(ComplexTrainableMagnitudeActivation, self).__init__()

        config_f_abs ={
            'num_channels': num_parameters,
            'vmin': 0,
            'vmax': 3.0,
            'num_weights': 5,
            'base_type': 'linear',
            'init': 'linear',
            'init_scale': 1.0,
        }

        self.f_abs = TrainableActivation(**config_f_abs)
        self.f_abs.weight.lrscale = 10

    def forward(self, x):
        magn = self.f_abs(complex_abs(x, eps=1e-9) )

        angle = complex_angle(x, eps=1e-9)

        re = magn * torch.cos(angle)
        im = magn * torch.sin(angle)

        fx = torch.cat([re.unsqueeze_(-1), im.unsqueeze_(-1)], -1)

        return fx

class ComplexTrainablePolarActivationBias(torch.nn.Module):
    def __init__(self, num_parameters):
        super(ComplexTrainablePolarActivationBias, self).__init__()

        config_f_abs ={
            'num_channels': num_parameters,
            'vmin': 0.0,
            'vmax': 3.0,
            'num_weights': 5,
            'base_type': 'linear',
            'init': 'linear',
            'init_scale': 1.0,
        }
        config_f_phi = {
            'num_channels': num_parameters,
            'vmin':  -np.pi,
            'vmax':  np.pi,
            'num_weights': 5,
            'base_type': 'linear',
            'init': 'linear',
            'init_scale': 1,
        }

        self.f_abs = TrainableActivation(**config_f_abs)
        self.f_phi = TrainableActivation(**config_f_phi)
        self.f_abs.weight.lrscale = 10
        self.f_phi.weight.lrscale = 10
        self.register_parameter(name='bias_abs', param=torch.nn.Parameter(torch.zeros(num_parameters)))
        self.register_parameter(name='bias_phase', param=torch.nn.Parameter(torch.zeros(num_parameters)))

    def forward(self, x):
        bias_abs = self.bias_abs.view(1, -1, *[1 for i in range(x.ndim-3)])
        bias_phase = self.bias_phase.view(1, -1, *[1 for i in range(x.ndim-3)])

        magn = self.f_abs(complex_abs(x, eps=1e-9) + bias_abs)
        angle = self.f_phi(complex_angle(x, eps=1e-9) + bias_phase)
        re = magn * torch.cos(angle)
        im = magn * torch.sin(angle)

        fx = torch.cat([re.unsqueeze_(-1), im.unsqueeze_(-1)], -1)

        return fx

class ComplexTrainableMagnitudeActivationBias(torch.nn.Module):
    def __init__(self, num_parameters):
        super(ComplexTrainableMagnitudeActivationBias, self).__init__()

        config_f_abs ={
            'num_channels': num_parameters,
            'vmin': 0,
            'vmax': 3.0,
            'num_weights': 5,
            'base_type': 'linear',
            'init': 'linear',
            'init_scale': 1.0,
        }

        self.f_abs = TrainableActivation(**config_f_abs)
        self.f_abs.weight.lrscale = 10
        self.register_parameter(name='bias_abs', param=torch.nn.Parameter(torch.zeros(num_parameters)))

    def forward(self, x):
        bias_abs = self.bias_abs.view(1, -1, *[1 for i in range(x.ndim-3)])

        magn = self.f_abs(complex_abs(x, eps=1e-9) + bias_abs)
        #print(magn.max(), magn.min())
        angle = complex_angle(x, eps=1e-9)

        re = magn * torch.cos(angle)
        im = magn * torch.sin(angle)

        fx = torch.cat([re.unsqueeze_(-1), im.unsqueeze_(-1)], -1)

        return fx
# class ComplexStudentT(torch.nn.Module):
#     def __init__(self,alpha):
#         super(ComplexStudentT, self).__init__()
#         self.alpha = alpha
#     def forward(self, x):
#         act_re = StudentT_fun().apply(x[...,0], self.alpha)
#         act_im = StudentT_fun().apply(x[...,1], self.alpha)
#         return torch.cat([act_re.unsqueeze_(-1), act_im.unsqueeze_(-1)], -1)               


class TestComplexTrainablePolarActivation(unittest.TestCase):
    def test2d(self):
        nf = 5
        act = ComplexTrainablePolarActivation(nf).cuda()
        x = torch.randn(2, nf, 2).cuda()
        y = act(x)
        self.assertTrue(y.shape == x.shape)

class TestComplexStudentT(unittest.TestCase):  
    def _test_gradient(self, shape):
        # setup the hyper parameters for each test
        dtype = torch.float64
        nf=5

        # perform a gradient check:
        epsilon = 1e-6

        # prefactors
        a = 1.1
        b = 1.1
        c = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x = torch.randn(*shape).to(dtype=dtype, device=cuda)
        th_v = torch.ones(nf).to(dtype=dtype, device=cuda)*0.1
        th_v = th_v.view(1, -1, *[1 for d in range(th_x.ndim-2)])
        th_w = torch.ones(nf).to(dtype=dtype, device=cuda)*0.1
        th_w = th_w.view(1, -1, *[1 for d in range(th_x.ndim-2)])
        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        th_b = torch.tensor(b, requires_grad=True, dtype=th_v.dtype, device=cuda)
        th_c = torch.tensor(c, requires_grad=True, dtype=th_w.dtype, device=cuda)
        op = ComplexStudentT_fun()

        # setup the model
        loss = ComplexL2Loss()
        compute_loss = lambda a, b, c: loss(op.apply(th_x*a, th_v*b, th_w*c))
        th_loss = compute_loss(th_a, th_b, th_c)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()
        grad_b = th_b.grad.cpu().numpy()
        grad_c = th_c.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon, th_b, th_c).cpu().numpy()
            l_an = compute_loss(th_a-epsilon, th_b, th_c).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        #self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

        # numerical gradient w.r.t. the weights
        with torch.no_grad():
            l_bp = compute_loss(th_a, th_b+epsilon, th_c).cpu().numpy()
            l_bn = compute_loss(th_a, th_b-epsilon, th_c).cpu().numpy()
            grad_b_num = (l_bp - l_bn) / (2 * epsilon)

        print("grad_v: {:.7f} num_grad_v {:.7f} success: {}".format(
            grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
        self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)

        # numerical gradient w.r.t. the weights
        with torch.no_grad():
            l_cp = compute_loss(th_a, th_b, th_c+epsilon).cpu().numpy()
            l_cn = compute_loss(th_a, th_b, th_c-epsilon).cpu().numpy()
            grad_c_num = (l_cp - l_cn) / (2 * epsilon)

        print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
            grad_c, grad_c_num, np.abs(grad_c - grad_c_num) < 1e-4))
        self.assertTrue(np.abs(grad_c - grad_c_num) < 1e-4)

    def test_gradient1(self):
        self._test_gradient((2,5,10,2))
    def test_gradient2(self):
        self._test_gradient((2,5,10,2))

                
# class Cardioid(torch.nn.Module):
#     def __init__(self, num_parameters):
#         super(Cardioid, self).__init__()
#         self.num_parameters = num_parameters
#         self.prelu_re = torch.nn.PReLU(num_parameters=num_parameters, init=0.25)
#         self.prelu_im = torch.nn.PReLU(num_parameters=num_parameters, init=0.25)

#     def forward(self, z):
#         assert z.shape[-1] == 2
#         act_re = self.prelu_re(z[...,0]).unsqueeze_(-1)
#         act_im = self.prelu_im(z[...,1]).unsqueeze_(-1)
#         return torch.cat([act_re, act_im], -1)
class TestModReLU(unittest.TestCase):   
    def _test_gradient(self, shape):
        # setup the hyper parameters for each test
        dtype = torch.float64
        nf=5

        # perform a gradient check:
        epsilon = 1e-6

        # prefactors
        a = 1.1
        b = 1.1

        # transfer to torch
        cuda = torch.device('cuda')
        th_x = torch.randn(*shape).to(dtype=dtype, device=cuda)
        th_w = torch.randn(nf).to(dtype=dtype, device=cuda)
        th_w = th_w.view(1, -1, *[1 for d in range(th_x.ndim-2)])

        th_a = torch.tensor(a, requires_grad=True, dtype=th_x.dtype, device=cuda)
        th_b = torch.tensor(b, requires_grad=True, dtype=th_x.dtype, device=cuda)

        op = ComplexModReLU_fun()
        loss = ComplexL2Loss()
        # setup the model
        compute_loss = lambda a, b: loss(op.apply(th_x*a, th_w*b, 1e-12))
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

        with torch.no_grad():
            l_bp = compute_loss(th_a, th_b+epsilon).cpu().numpy()
            l_bn = compute_loss(th_a, th_b-epsilon).cpu().numpy()
            grad_b_num = (l_bp - l_bn) / (2 * epsilon)

        print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
            grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
        self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)

    def test_gradient1(self):
        self._test_gradient((2,5,2))
    def test_gradient2(self):
        self._test_gradient((2,5,10,2))

if __name__ == "__main__":
    unittest.test()