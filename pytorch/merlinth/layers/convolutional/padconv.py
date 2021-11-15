import torch
import numpy as np
import optoth.pad
import unittest
import merlinth.utils

__all__ = ['PadConv1d',
           'PadConv2d',
           'PadConv3d',
           'PadConvScale2d',
           'PadConvScale3d',
           'PadConvScale2dTranspose',
           'PadConvScale3dTranspose',
           ]

class PadConv(torch.nn.Module):
    def __init__(self, 
                 rank,
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = merlinth.utils.validate_input_dimension(rank, kernel_size)
        self.stride = merlinth.utils.validate_input_dimension(rank, stride)
        self.dilation = merlinth.utils.validate_input_dimension(rank, dilation)
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(filters)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.pad = pad
        self.padding_mode = padding
        self.rank = rank

        # add the parameter
        self.weight = torch.nn.Parameter(torch.empty(filters, in_channels, *self.kernel_size))
        # insert them using a normal distribution
        torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1/np.prod(in_channels*np.prod(kernel_size))))

        # specify reduction index
        self.weight.L_init = 1e+4
        if zero_mean or bound_norm:
            self.weight.reduction_dim = tuple([i+1 for i in range(rank+1)])
    
            # define a projection
            def l2_proj(surface=False):
                # reduce the mean
                if zero_mean:
                    mean = torch.sum(self.weight.data, self.weight.reduction_dim, True) / (self.in_channels*np.prod(self.kernel_size))
                    self.weight.data.sub_(mean)
                # normalize by the l2-norm
                if bound_norm:
                    norm = torch.sum(self.weight.data**2, self.weight.reduction_dim, True).sqrt_()
                    if surface:
                        self.weight.data.div_(norm)
                    else:
                        self.weight.data.div_(
                            torch.max(norm, torch.ones_like(norm)))
            self.weight.proj = l2_proj

            # initially call the projection
            self.weight.proj(True)

        if rank == 1:
            self._convolution_op = torch.nn.functional.conv1d
            self._padding_op = optoth.pad.pad1d
            self._convolutionT_op = torch.nn.functional.conv_transpose1d
            self._paddingT_op = optoth.pad.pad1d_transpose
        elif rank == 2:
            self._convolution_op = torch.nn.functional.conv2d
            self._padding_op = optoth.pad.pad2d
            self._convolutionT_op = torch.nn.functional.conv_transpose2d
            self._paddingT_op = optoth.pad.pad2d_transpose
        elif rank == 3:
            self._convolution_op = torch.nn.functional.conv3d
            self._padding_op = optoth.pad.pad3d
            self._convolutionT_op = torch.nn.functional.conv_transpose3d
            self._paddingT_op = optoth.pad.pad3d_transpose

    def get_weight(self):
        weight = self.weight
        return weight

    def _compute_optox_padding(self):
        pad = []
        for w in self.get_weight().shape[2:][::-1]:
            pad += [w//2, w//2]
        return pad

    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # then pad
        pad = self._compute_optox_padding()
        if self.pad and any(pad) > 0:
            x = self._padding_op(x, pad, mode=self.padding_mode)
        # compute the convolution
        x = self._convolution_op(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = [output_shape[-self.rank+i] - ((x.shape[-self.rank+i]-1)*self.stride[i]+1) for i in range(self.rank)]
        else:
            output_padding = 0

        # compute the convolution
        x = self._convolutionT_op(x, weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

        pad = self._compute_optox_padding()
        if self.pad and any(pad) > 0:
            x = self._paddingT_op(x, pad, mode=self.padding_mode)
        return x

    def extra_repr(self):
        s = "({filters}, {in_channels}, {kernel_size}), invariant={invariant}"
        if self.stride != 1:
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.bias is None:
            s += ", bias=True"
        if self.zero_mean:
            s += ", zero_mean={zero_mean}"
        if self.bound_norm:
            s += ", bound_norm={bound_norm}"
        return s.format(**self.__dict__)


class PadConv1d(PadConv):
    def __init__(self, 
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__(1,
                        in_channels=in_channels,
                        filters=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias, 
                        zero_mean=zero_mean,
                        bound_norm=bound_norm,
                        pad=pad)

class PadConv2d(PadConv):
    def __init__(self, 
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__(2,
                        in_channels=in_channels,
                        filters=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias, 
                        zero_mean=zero_mean,
                        bound_norm=bound_norm,
                        pad=pad)

class PadConvScale2d(PadConv2d):
    def __init__(self, in_channels, filters, kernel_size=3, groups=1, stride=2, bias=False, padding='symmetric', zero_mean=False, bound_norm=False):
        super(PadConvScale2d, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, padding=padding, 
            stride=stride, dilation=1, groups=groups, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)
        assert self.kernel_size[0] == self.kernel_size[1]
        assert self.stride[0] == self.stride[1]

        # create the convolution kernel
        if any(self.stride) > 1:
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            np_k = np.reshape(np_k, (1, 1, 5, 5))
            self.register_buffer('blur', torch.from_numpy(np_k))

    def get_weight(self):
        weight = super().get_weight()
        if any(self.stride) > 1:
            weight = weight.reshape(-1, 1, *self.kernel_size)
            for i in range(self.stride//2): 
                weight = torch.nn.functional.conv2d(weight, self.blur, padding=4)
            weight = weight.reshape(self.filters,
                                    self.in_channels,
                                    self.kernel_size[0]+2*self.stride[0],
                                    self.kernel_size[1]+2*self.stride[1])
        return weight

class PadConvScaleTranspose2d(PadConvScale2d):
    def __init__(self, in_channels, filters, kernel_size=3, padding='symmetric',
                 groups=1, stride=2, bias=False, zero_mean=False, bound_norm=False):
        super(PadConvScaleTranspose2d, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, 
            padding=padding, groups=groups, stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)

class PadConv3d(PadConv):
    def __init__(self, 
                 in_channels,
                 filters,
                 kernel_size,
                 stride=1,
                 padding='symmetric',
                 dilation=1,
                 groups=1,
                 bias=False, 
                 zero_mean=False,
                 bound_norm=False,
                 pad=True):
        super().__init__(3,
                        in_channels=in_channels,
                        filters=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias, 
                        zero_mean=zero_mean,
                        bound_norm=bound_norm,
                        pad=pad)

class PadConvScale3d(PadConv3d):
    def __init__(self, in_channels, filters, kernel_size=3, padding='symmetric',
                 groups=1, stride=(1,2,2), bias=False, zero_mean=False, bound_norm=False):
        super(PadConvScale3d, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, padding=padding,
            stride=stride, dilation=1, groups=groups, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)
        assert self.kernel_size[1] == self.kernel_size[2]
        assert self.kernel_size[1] > 1
        assert self.stride[1] == self.stride[2]
        assert self.stride[1] > 1

        # create the convolution kernel
        if self.stride[1] > 1:
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            np_k = np.reshape(np_k, (1, 1, 5, 5))
            self.register_buffer('blur', torch.from_numpy(np_k))

    def get_weight(self):
        weight = super().get_weight()
        if self.stride[1] > 1:
            weight = weight.reshape(-1, 1, self.kernel_size[1], self.kernel_size[2])
            for i in range(self.stride[1]//2): 
                weight = torch.nn.functional.conv2d(weight, self.blur, padding=4)
            weight = weight.reshape(self.filters, self.in_channels, self.kernel_size[0], self.kernel_size[1]+2*self.stride[1], self.kernel_size[1]+2*self.stride[1])
        return weight

class PadConvScaleTranspose3d(PadConvScale3d):
    def __init__(self, in_channels, filters, kernel_size=3, groups=1, stride=(1,2,2), bias=False, zero_mean=False, bound_norm=False):
        super(PadConvScaleTranspose3d, self).__init__(
            in_channels=in_channels, filters=filters, kernel_size=kernel_size, 
            groups=groups, stride=stride, bias=bias, 
            zero_mean=zero_mean, bound_norm=bound_norm)

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)

class PadConv1dTest(unittest.TestCase):
    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = PadConv1d(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True).cuda()
        np_weight = model.weight.detach().cpu().numpy()
        reduction_dim = model.weight.reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))
        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test_grad(self, conv_fun, kernel_size, strides, dilation_rate, padding):
        nBatch = 5
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, nf_in, N]

        model = conv_fun(nf_in, nf_out, kernel_size=kernel_size, stride=strides, padding=padding, zero_mean=False, bound_norm=False)
        model.cuda()
        x = torch.randn(shape).cuda()
        x.requires_grad_(True)

        Kx = model(x)
        loss = 0.5 * torch.sum(Kx ** 2)
        loss.backward()
        grad_x = x.grad
        x_autograd = grad_x.cpu().numpy()

        KHKx = model.backward(Kx, x.shape)
        x_bwd = KHKx.detach().cpu().numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test1(self):
        self._test_grad(PadConv1d, 5, 1, 1, 'symmetric')

class PadConv2dTest(unittest.TestCase):
    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = PadConv2d(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True)
        np_weight = model.weight.detach().numpy()
        reduction_dim = model.weight.reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))
        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test_grad(self, conv_fun, kernel_size, strides, dilation_rate, padding):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, nf_in, M, N]

        model = conv_fun(nf_in, nf_out, kernel_size=kernel_size, stride=strides, padding=padding, zero_mean=False, bound_norm=False)
        model.cuda()
        x = torch.randn(shape).cuda()
        x.requires_grad_(True)
        Kx = model(x)
        loss = 0.5 * torch.sum(Kx ** 2)
        loss.backward()
        grad_x = x.grad
        x_autograd = grad_x.detach().cpu().numpy()

        KHKx = model.backward(Kx, x.shape)
        x_bwd = KHKx.detach().cpu().numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test1(self):
        self._test_grad(PadConv2d, 5, 1, 1, 'symmetric')

    def test2(self):
        self._test_grad(PadConv2d, [3, 5], 1, 1, 'symmetric')

    def test3(self):
        self._test_grad(PadConvScale2d, 3, 2, 1, 'symmetric')
        
class PadConvScaleTranspose2dTest(unittest.TestCase):
    def test_conv_transpose2d(self):
        nBatch = 5
        M = 320
        N = 320
        nf_in = 1
        nf_out = 32
        
        model = PadConvScaleTranspose2d(nf_in, nf_out, kernel_size=3).cuda()
        
        x = torch.randn(nBatch, nf_in, M, N).cuda()
        Kx = model.backward(x)
        
        y = torch.randn(*Kx.shape).cuda()
        KHy = model.forward(y, output_shape=x.shape)

        rhs = torch.sum(Kx * y).detach().cpu().numpy()
        lhs = torch.sum(x * KHy).detach().cpu().numpy()

        self.assertTrue(rhs, lhs)

class PadConv3dTest(unittest.TestCase):
    def test_constraints(self):
        nf_in = 1
        nf_out = 32
        
        model = PadConv3d(nf_in, nf_out, kernel_size=3, zero_mean=True, bound_norm=True).cuda()
        np_weight = model.weight.detach().cpu().numpy()
        reduction_dim = model.weight.reduction_dim

        weight_mean = np.mean(np_weight, axis=reduction_dim)
        self.assertTrue(np.max(np.abs(weight_mean)) < 1e-6)

        weight_norm = np.sqrt(np.sum(np.conj(np_weight) * np_weight, axis=reduction_dim))

        self.assertTrue(np.max(np.abs(weight_norm-1)) < 1e-6)

    def _test_grad(self, conv_fun, kernel_size, strides, dilation_rate, padding):
        nBatch = 5
        M = 256
        N = 256
        D = 10
        nf_in = 2
        nf_out = 16
        shape = [nBatch, nf_in, D, M, N]

        model = conv_fun(nf_in, nf_out, kernel_size=kernel_size, stride=strides, padding=padding, zero_mean=False, bound_norm=False)
        model.double().cuda()
        x = torch.randn(shape).double().cuda()
        x.requires_grad_(True)

        Kx = model(x)
        loss = 0.5 * torch.sum(Kx ** 2)
        loss.backward()
        grad_x = x.grad
        x_autograd = grad_x.detach().cpu().numpy()

        KHKx = model.backward(Kx, x.shape)
        x_bwd = KHKx.detach().cpu().numpy()
        self.assertTrue(np.sum(np.abs(x_autograd - x_bwd))/x_autograd.size < 1e-5)

    def test1(self):
        self._test_grad(PadConv3d, 5, 1, 1, 'symmetric')

    def test2(self):
        self._test_grad(PadConv3d, [3, 5, 5], 1, 1, 'symmetric')

    def test3(self):
        self._test_grad(PadConvScale3d, 3, (1, 2, 2), 1, 'symmetric')

class PadConvScaleTranspose3dTest(unittest.TestCase):
    def test_conv_transpose3d(self):
        nBatch = 5
        M = 320
        N = 320
        D = 10
        nf_in = 1
        nf_out = 32
        
        model = PadConvScaleTranspose3d(nf_in, nf_out, kernel_size=3).cuda()
        
        x = torch.randn(nBatch, nf_in, D, M, N).cuda()
        Kx = model.backward(x)
        
        y = torch.randn(*Kx.shape).cuda()
        KHy = model.forward(y, output_shape=x.shape)

        rhs = torch.sum(Kx * y).detach().cpu().numpy()
        lhs = torch.sum(x * KHy).detach().cpu().numpy()

        self.assertTrue(rhs, lhs)

if __name__ == "__main__":
    unittest.test()
