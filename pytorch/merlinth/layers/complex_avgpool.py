import sys
import torch
import merlinth
try:
    import optoth.averagepooling
except:
    print('optoth could not be imported')
import six


def get(identifier):
    return MagnitudeAveragePooling(identifier)


def MagnitudeAveragePooling(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'MagnitudeAveragePool' + str(identifier).upper().replace('T', 't')
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret max pooling function identifier: {}'.format(identifier))


def deserialize(op):
    if op == 'MagnitudeAveragePool1D' or op == 'MagnitudeAveragePooling1D':
        return MagnitudeAveragePool1D
    elif op == 'MagnitudeAveragePool2D' or op == 'MagnitudeAveragePooling2D':
        return MagnitudeAveragePool2D
    elif op == 'MagnitudeAveragePool2Dt' or op == 'MagnitudeAveragePooling2Dt':
        return MagnitudeAveragePool2Dt
    elif op == 'MagnitudeAveragePool3D' or op == 'MagnitudeAveragePooling3D':
        return MagnitudeAveragePool3D
    elif op == 'MagnitudeAveragePool3Dt' or op == 'MagnitudeAveragePooling3Dt':
        return MagnitudeAveragePool3Dt
    else:
        raise ValueError(f"Selected operation '{op}' not implemented in complex convolutional")


class MagnitudeAveragePool(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=True, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeAvgPool', alpha=1, beta=1, **kwargs):
        super(MagnitudeAveragePool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.alpha = alpha
        self.beta = beta
        self.ceil_mode = ceil_mode
        self.layer_name = layer_name
        self.padding_mode = padding_mode
        self.return_indices = return_indices
        self.channel_first = True  # default PyTorch order: [N, C, H, W, ....]
        self.optox = optox and (True if 'optoth.averagepooling' in sys.modules else False)

        if not self.optox:
            if self.layer_name == 'MagnitudeAvgPool3D':
                self.pool = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
            elif self.layer_name == 'MagnitudeAvgPool2D':
                self.pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
            elif self.layer_name == 'MagnitudeAvgPool1D':
                self.pool = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        if self.optox and merlinth.iscomplex(x):
            print('forward:', x.shape, self.kernel_size, self.padding, self.stride, self.dilation, self.alpha,
                  self.beta, self.padding_mode, x.real.dtype, self.channel_first, self.ceil_mode)
            return self.op.apply(x, self.kernel_size, self.padding, self.stride, self.dilation, self.alpha, self.beta,
                                 self.padding_mode, x.real.dtype, self.channel_first, self.ceil_mode)
        else:
            magn = merlinth.complex_abs(x, eps=1e-9)
            _, indices = self.pool(magn)
            pool_re = self.retrieve_elements_from_indices(torch.real(x), indices)
            pool_im = self.retrieve_elements_from_indices(torch.imag(x), indices)
            return torch.complex(pool_re, pool_im)

    def retrieve_elements_from_indices(self, tensor, indices):
        flattened_tensor = tensor.flatten(start_dim=2)
        output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        return output


class MagnitudeAveragePool1D(MagnitudeAveragePool):
    def __init__(self, kernel_size=(1, ), stride=(2, ), padding=(0, ), dilation=(1, ), return_indices=False, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeAvgPool1D', alpha=1, beta=1, **kwargs):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode, padding_mode, optox, layer_name, alpha, beta, **kwargs)
        self.op = optoth.averagepooling.Averagepooling1dFunction


class MagnitudeAveragePool2D(MagnitudeAveragePool):
    def __init__(self, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), return_indices=False, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeAvgPool2D', alpha=1, beta=1, **kwargs):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode, padding_mode, optox, layer_name, alpha, beta, **kwargs)
        self.op = optoth.averagepooling.Averagepooling2dFunction


class MagnitudeAveragePool3D(MagnitudeAveragePool):
    def __init__(self, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), return_indices=False, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeAvgPool3D', alpha=1, beta=1, **kwargs):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode, padding_mode, optox, layer_name, alpha, beta, **kwargs)
        self.op = optoth.averagepooling.Averagepooling3dFunction


class MagnitudeAveragePool2Dt(MagnitudeAveragePool):
    def __init__(self, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), return_indices=False, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeAvgPool2Dt', alpha=1, beta=1, **kwargs):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode, padding_mode, optox, layer_name, alpha, beta, **kwargs)
        self.op = optoth.averagepooling.Averagepooling3dFunction


class MagnitudeAveragePool3Dt(MagnitudeAveragePool):
    def __init__(self, kernel_size=(2, 2, 2, 2), stride=(2, 2, 2, 2), padding=(0, 0, 0, 0), dilation=(1, 1, 1, 1), return_indices=False, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeAvgPool3Dt', alpha=1, beta=1, **kwargs):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode, padding_mode, optox, layer_name, alpha, beta, **kwargs)
        self.op = optoth.averagepooling.Averagepooling4dFunction


# Aliases
MagnitudeAveragePool4D = MagnitudeAveragePool3Dt
