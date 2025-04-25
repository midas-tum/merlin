import sys
import torch
import merlinth
try:
    import optoth.maxpooling
except:
    print('optoth could not be imported')
import six


def get(identifier):
    return MagnitudeMaxPooling(identifier)


def MagnitudeMaxPooling(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'MagnitudeMaxPool' + str(identifier).upper().replace('T', 't')
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret max pooling function identifier: {}'.format(identifier))


def deserialize(op):
    if op == 'MagnitudeMaxPool1D' or op == 'MagnitudeMaxPooling1D':
        return MagnitudeMaxPool1D
    elif op == 'MagnitudeMaxPool2D' or op == 'MagnitudeMaxPooling2D':
        return MagnitudeMaxPool2D
    elif op == 'MagnitudeMaxPool2Dt' or op == 'MagnitudeMaxPooling2Dt':
        return MagnitudeMaxPool2Dt
    elif op == 'MagnitudeMaxPool3D' or op == 'MagnitudeMaxPooling3D':
        return MagnitudeMaxPool3D
    elif op == 'MagnitudeMaxPool3Dt' or op == 'MagnitudeMaxPooling3Dt':
        return MagnitudeMaxPool3Dt
    else:
        raise ValueError(f"Selected operation '{op}' not implemented in complex convolutional")



class MagnitudeMaxPool(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=True, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeMaxPool', alpha=1, beta=1, **kwargs):
        super(MagnitudeMaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.alpha = alpha
        self.beta = beta
        self.ceil_mode = ceil_mode
        self.padding_mode = padding_mode
        self.layer_name = layer_name
        self.channel_first = True  # default PyTorch order: [N, C, H, W, ....]
        self.optox = optox and (True if 'optoth.maxpooling' in sys.modules else False)

        if not self.optox:
            if self.layer_name == 'MagnitudeMaxPool3D':
                self.op = torch.nn.MaxPool3d(kernel_size, stride, padding, dilation, return_indices)
            elif self.layer_name == 'MagnitudeMaxPool2D':
                self.op = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices)
            elif self.layer_name == 'MagnitudeMaxPool1D':
                self.op = torch.nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices)
    
    def forward(self, x):
        if self.optox and merlinth.iscomplex(x):
            print('forward:', x.shape, self.kernel_size, self.padding, self.stride, self.dilation, self.alpha,
                  self.beta, self.padding_mode, x.real.dtype, self.channel_first, self.ceil_mode)
            return self.op.apply(x, self.kernel_size, self.padding, self.stride, self.dilation,  self.alpha, self.beta,
                                 self.padding_mode, x.real.dtype, self.channel_first)
        else:
            magn = merlinth.complex_abs(x, eps=1e-9)
            _, indices = self.op(magn)
            pool_re = self.retrieve_elements_from_indices(torch.real(x), indices)
            pool_im = self.retrieve_elements_from_indices(torch.imag(x), indices)
            return torch.complex(pool_re, pool_im)

    def retrieve_elements_from_indices(self, tensor, indices):
        flattened_tensor = tensor.flatten(start_dim=2)
        output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        return output


class MagnitudeMaxPool1D(MagnitudeMaxPool):
    def __init__(self, kernel_size=(2,), stride=(2,), padding=(0,), dilation=(1,), return_indices=True, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeMaxPool1D', alpha=1, beta=1, **kwargs):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode, padding_mode, optox, layer_name, alpha, beta, **kwargs)
        self.op = optoth.maxpooling.Maxpooling1dFunction


class MagnitudeMaxPool2D(MagnitudeMaxPool):
    def __init__(self, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), return_indices=True, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeMaxPool2D', alpha=1, beta=1, **kwargs):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode, padding_mode, optox, layer_name, alpha, beta, **kwargs)
        self.op = optoth.maxpooling.Maxpooling2dFunction


class MagnitudeMaxPool3D(MagnitudeMaxPool):
    def __init__(self, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), return_indices=True, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeMaxPool2D', alpha=1, beta=1, **kwargs):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode, padding_mode, optox, layer_name, alpha, beta, **kwargs)
        self.op = optoth.maxpooling.Maxpooling3dFunction


class MagnitudeMaxPool2Dt(MagnitudeMaxPool):
    def __init__(self, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), return_indices=True, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeMaxPool2Dt', alpha=1, beta=1, **kwargs):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode, padding_mode, optox, layer_name, alpha, beta, **kwargs)
        self.op = optoth.maxpooling.Maxpooling3dFunction


class MagnitudeMaxPool3Dt(MagnitudeMaxPool):
    def __init__(self, kernel_size=(2, 2, 2, 2), stride=(2, 2, 2, 2), padding=(0, 0, 0, 0), dilation=(1, 1, 1, 1), return_indices=True, ceil_mode=False, padding_mode='SAME', optox=True, layer_name='MagnitudeMaxPool2Dt', alpha=1, beta=1, **kwargs):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode, padding_mode, optox, layer_name, alpha, beta, **kwargs)
        self.op = optoth.maxpooling.Maxpooling4dFunction


# Aliases
MagnitudeMaxPool4D = MagnitudeMaxPool3Dt
