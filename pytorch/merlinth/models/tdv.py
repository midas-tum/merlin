import torch

from merlinth.models.foe import Regularizer
from merlinth.layers.convolutional.padconv import (
    PadConv2d,
    PadConvScale2d,
    PadConvScaleTranspose2d,
    PadConv3d,
    PadConvScale3d,
    PadConvScaleTranspose3d,   
)
from merlinth.layers.convolutional.complex_padconv import (
    ComplexPadConv2d,
    ComplexPadConvScale2d,
    ComplexPadConvScaleTranspose2d,
    ComplexPadConv3d,
    ComplexPadConvScale3d,
    ComplexPadConvScaleTranspose3d,   
)
from merlinth.layers.complex_act import ModStudentT2
from merlinth.layers.module import ComplexModule

from merlinth.complex import (
    complex_norm,
    complex_abs
)
__all__ = ['TDV']

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


class StudentT2(torch.nn.Module):
    def __init__(self,alpha):
        super(StudentT2, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        return StudentT_fun2().apply(x, self.alpha)

class MicroBlock(torch.nn.Module):
    def __init__(self, dim, num_features, alpha=1, bound_norm=False):
        super(MicroBlock, self).__init__()
        
        if dim == '2D':
            conv_module = PadConv2d
        elif dim == '3D':
            conv_module = PadConv3d
        else:
            raise RuntimeError(f"TDV regularizer not defined for {dim}!")

        self.conv1 = conv_module(num_features, num_features, kernel_size=3, bound_norm=bound_norm, bias=False)
        self.act = StudentT2(alpha=alpha)
        self.conv2 = conv_module(num_features, num_features, kernel_size=3, bound_norm=bound_norm, bias=False)

        # save the gradient of the the activation function for the backward path
        self.act_prime = None

    def forward(self, x):
        a, ap = self.act(self.conv1(x))
        self.act_prime = ap
        x = x + self.conv2(a)
        return x

    def backward(self, grad_out):
        assert not self.act_prime is None
        out = grad_out + self.conv1.backward(self.act_prime*self.conv2.backward(grad_out))
        if not self.act_prime.requires_grad:
            self.act_prime = None
        return out

class ComplexMicroBlock(ComplexModule):
    def __init__(self, dim, num_features, alpha=1, bound_norm=False):
        super(ComplexMicroBlock, self).__init__()

        if dim == '2D':
            conv_module = ComplexPadConv2d
        elif dim == '3D':
            conv_module = ComplexPadConv3d
        else:
            raise RuntimeError(f"TDV regularizer not defined for {dim}!")

        self.conv1 = conv_module(num_features, num_features, kernel_size=3, bound_norm=bound_norm, bias=False)
        self.act = ModStudentT2(num_features, alpha_init=alpha)
        self.conv2 = conv_module(num_features, num_features, kernel_size=3, bound_norm=bound_norm, bias=False)

        # save the gradient of the the activation function for the backward path
        self.act_prime = None

    def forward(self, x):
        a, ap, apH = self.act(self.conv1(x))
        self.act_prime = ap
        self.act_primeH = apH
        x = x + self.conv2(a)
        return x

    def backward(self, grad_out):
        assert not self.act_prime is None
        z = self.conv2.backward(grad_out)
        zH = torch.conj(z)
        out = grad_out + self.conv1.backward(zH * self.act_primeH + z * torch.conj(self.act_prime))
        if not self.act_prime.requires_grad:
            self.act_prime = None
            self.act_primeH = None
        return out

class MacroBlock(ComplexModule):
    def __init__(self, dim, num_features, num_scales=3, multiplier=1, bound_norm=False, alpha=1.0, is_complex=False):
        super(MacroBlock, self).__init__()

        self.num_scales = num_scales

        if is_complex:
            micro_block_module = ComplexMicroBlock
        else:
            micro_block_module = MicroBlock

        if dim == '2D' and is_complex:
            conv_module_scale = ComplexPadConvScale2d
            conv_module_scale_transpose = ComplexPadConvScaleTranspose2d
        elif dim == '2D':
            conv_module_scale = PadConvScale2d
            conv_module_scale_transpose = PadConvScaleTranspose2d
        elif dim == '3D' and is_complex:
            conv_module_scale = ComplexPadConvScale3d
            conv_module_scale_transpose = ComplexPadConvScaleTranspose3d
        elif dim == '3D':
            conv_module_scale = PadConvScale3d
            conv_module_scale_transpose = PadConvScaleTranspose3d
        else:
            raise RuntimeError(f"MacroBlock not defined for {dim}!")
        # micro blocks
        self.mb = []
        for i in range(num_scales-1):
            b = torch.nn.ModuleList([
                micro_block_module(dim, num_features * multiplier**i, bound_norm=bound_norm, alpha=alpha),
                micro_block_module(dim, num_features * multiplier**i, bound_norm=bound_norm, alpha=alpha)
            ])
            self.mb.append(b)
        # the coarsest scale has only one microblock
        self.mb.append(torch.nn.ModuleList([
                micro_block_module(dim, num_features * multiplier**(num_scales-1), bound_norm=bound_norm, alpha=alpha)
        ]))
        self.mb = torch.nn.ModuleList(self.mb)

        # get conv module
        if dim == '2D':
            strides = 2
        elif dim == '3D':
            strides = (1,2,2)
        else:
            raise RuntimeError(f"MacroBlock not defined for {dim}!")

        # down/up sample
        self.conv_down = []
        self.conv_up = []
        for i in range(1, num_scales):
            self.conv_down.append(
                conv_module_scale(num_features * multiplier**(i-1), num_features * multiplier**i, kernel_size=3, stride=strides, bias=False, bound_norm=bound_norm)
            )
            self.conv_up.append(
                conv_module_scale_transpose(num_features * multiplier**(i-1), num_features * multiplier**i, kernel_size=3, stride=strides, bias=False, bound_norm=bound_norm)
            )
        self.conv_down = torch.nn.ModuleList(self.conv_down)
        self.conv_up = torch.nn.ModuleList(self.conv_up)

    def forward(self, x):
        assert len(x) == self.num_scales

        # down scale and feature extraction
        for i in range(self.num_scales-1):
            # 1st micro block of scale
            x[i] = self.mb[i][0](x[i])
            # down sample for the next scale
            x_i_down = self.conv_down[i](x[i])
            if x[i+1] is None:
                x[i+1] = x_i_down
            else:
                x[i+1] = x[i+1] + x_i_down
        
        # on the coarsest scale we only have one micro block
        x[self.num_scales-1] = self.mb[self.num_scales-1][0](x[self.num_scales-1])

        # up scale the features
        for i in range(self.num_scales-1)[::-1]:
            # first upsample the next coarsest scale
            x_ip1_up = self.conv_up[i](x[i+1], x[i].shape)
            # skip connection
            x[i] = x[i] + x_ip1_up
            # 2nd micro block of scale
            x[i] = self.mb[i][1](x[i])

        return x

    def backward(self, grad_x):

        # backward of up scale the features
        for i in range(self.num_scales-1):
            # 2nd micro block of scale
            grad_x[i] = self.mb[i][1].backward(grad_x[i])
            # first upsample the next coarsest scale
            grad_x_ip1_up = self.conv_up[i].backward(grad_x[i])
            # skip connection
            if grad_x[i+1] is None:
                grad_x[i+1] = grad_x_ip1_up
            else:
                grad_x[i+1] = grad_x[i+1] + grad_x_ip1_up

        # on the coarsest scale we only have one micro block
        grad_x[self.num_scales-1] = self.mb[self.num_scales-1][0].backward(grad_x[self.num_scales-1])

        # down scale and feature extraction
        for i in range(self.num_scales-1)[::-1]:
            # down sample for the next scale
            grad_x_i_down = self.conv_down[i].backward(grad_x[i+1], grad_x[i].shape)
            grad_x[i] = grad_x[i] + grad_x_i_down
            # 1st micro block of scale
            grad_x[i] = self.mb[i][0].backward(grad_x[i])
        
        return grad_x


class TDV(Regularizer):
    """
    total deep variation (TDV) regularizer
    """
    def __init__(self, config=None, file=None):
        super(TDV, self).__init__()

        if (config is None and file is None) or \
            (not config is None and not file is None):
            raise RuntimeError('specify EITHER a config dictionary OR a `.pth`-file!')

        if not file is None:
            if not file.endswith('.pth'):
                raise ValueError('file needs to end with `.pth`!')
            checkpoint = torch.load(file)
            config = checkpoint['config']
            state_dict = checkpoint['model']
            self.tau = checkpoint['tau']
        else:
            state_dict = None
            self.tau = 1.0

        self.in_channels = config['in_channels']
        self.num_features = config['num_features']
        self.multiplier = config['multiplier']
        self.num_mb = config['num_mb']
        self.is_complex = config['is_complex']

        if 'zero_mean' in config.keys():
            self.zero_mean = config['zero_mean']
        else:
            self.zero_mean = True
        if 'num_scales' in config.keys():
            self.num_scales = config['num_scales']
        else:
            self.num_scales = 3
        if 'alpha' in config.keys():
            self.alpha = config['alpha']
        else:
            self.alpha = 1.0

        if config['dim'] == '2D' and self.is_complex:
            conv_module = ComplexPadConv2d
        elif config['dim'] == '2D':
            conv_module = PadConv2d
        elif config['dim'] == '3D' and self.is_complex:
            conv_module = ComplexPadConv3d
        elif config['dim'] == '3D':
            conv_module = PadConv3d
        else:
            raise RuntimeError(f"TDV regularizer not defined for {config['dim']}!")

        if self.is_complex:
            self._potential = self._potential_complex
            self._activation = self._activation_complex
        else:
            self._potential = self._potential_real
            self._activation = self._activation_real
        # construct the regularizer
        self.K1 = conv_module(self.in_channels, self.num_features, 3, zero_mean=self.zero_mean, bound_norm=True, bias=False)

        self.mb = torch.nn.ModuleList([MacroBlock(config['dim'], self.num_features, num_scales=self.num_scales, bound_norm=False,  multiplier=self.multiplier, alpha=self.alpha, is_complex=config['is_complex']) 
                                        for _ in range(self.num_mb)])

        self.KN = conv_module(self.num_features, 1, 1, bound_norm=False, bias=False)

        if not state_dict is None:
            self.load_state_dict(state_dict)

    def _transformation(self, x):
        # extract features
        x = self.K1(x)
        # apply mb
        x = [x,] + [None for i in range(self.num_scales-1)]
        for i in range(self.num_mb):
            x = self.mb[i](x)
        # compute the output
        out = self.KN(x[0])
        return out

    def _activation_real(self, x):
        # scale by the number of features
        return torch.ones_like(x) / self.num_features

    def _potential_real(self, x):
        return x / self.num_features

    def _activation_complex(self, x):
        nx = complex_norm(x)
        return  nx / (2 * self.num_features)

    def _potential_complex(self, x):
        return complex_abs(x) / self.num_features

    def _transformation_T(self, grad_out):
        # compute the output
        grad_x = self.KN.backward(grad_out)
        # apply mb
        grad_x = [grad_x,] + [None for i in range(self.num_scales-1)]
        for i in range(self.num_mb)[::-1]:
            grad_x = self.mb[i].backward(grad_x)
        # extract features
        grad_x = self.K1.backward(grad_x[0])
        return grad_x

    def energy(self, x):
        x = self._transformation(x)
        return self._potential(x)

    def grad(self, x, get_energy=False):
        # compute the energy
        x = self._transformation(x)
        if get_energy:
            energy = self._potential(x)
        # and its gradient
        x = self._activation(x)
        grad = self._transformation_T(x)
        if get_energy:
            return energy, grad
        else:
            return grad

