import torch
import merlinth
try:
    import optoth.averagepooling
except:
    print('optoth could not be imported')

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
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, padding_mode='SAME', return_indices=False, rank=2):
        super(MagnitudeAveragePool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.alpha = 1
        self.beta = 1
        self.padding_mode = padding_mode
        self.return_indices = return_indices
        self.channel_first = True  # default PyTorch order: [N, C, H, W, ....]
        self.optox = optox and (True if 'optoth.averagepooling' in sys.modules else False)

        if not self.optox:
            if rank == 3:
                self.pool = torch.nn.AvgPool3d(kernel_size, stride, return_indices=True)
            elif rank == 2:
                self.pool = torch.nn.AvgPool2d(kernel_size, stride, return_indices=True)
            else:
                raise ValueError(f"pooling for dim={rank} not defined")
    
    def forward(self, x):
        magn = merlinth.complex_abs(x, eps=1e-9)
        _, indices = self.pool(magn)
        pool_re = self.retrieve_elements_from_indices(torch.real(x), indices)
        pool_im = self.retrieve_elements_from_indices(torch.imag(x), indices)
        return torch.complex(pool_re, pool_im)

    def retrieve_elements_from_indices(self, tensor, indices):
        flattened_tensor = tensor.flatten(start_dim=2)
        output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        return output


class MagnitudeAveragePool2D(MagnitudeAveragePool):
    def __init__(self, kernel_size, stride=(2, 2), padding=(0, 0), dilation=(1, 1), padding_mode='SAME', return_indices=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, rank=2)

    def forward(self, x):
        return optoth.averagepooling.averagepooling2d(x, self.kernel_size, self.padding, self.stride, self.dilation,
                                              self.alpha, self.beta, self.padding_mode, x.dtype, self.channel_first)

    def backward(self, grad_output):
        return optoth.averagepooling.averagepooling2d_backward(TODO, TODO, TODO, self.kernel_size, self.padding, self.stride, self.dilation,
                                                       self.alpha, self.beta, self.padding_mode, x.dtype, self.channel_first)

class MagnitudeAveragePool3D(MagnitudeAveragePool):
    def __init__(self, kernel_size, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), padding_mode='SAME', return_indices=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, rank=3)

    def forward(self, x):
        return optoth.averagepooling.averagepooling3d(x, self.kernel_size, self.padding, self.stride, self.dilation,
                                              self.alpha, self.beta, self.padding_mode, x.dtype, self.channel_first)

    def backward(self, grad_output):
        return optoth.averagepooling.averagepooling3d_backward(TODO, TODO, TODO, self.kernel_size, self.padding, self.stride,
                                                       self.dilation,
                                                       self.alpha, self.beta, self.padding_mode, x.dtype,
                                                       self.channel_first)

class MagnitudeAveragePool2Dt(MagnitudeAveragePool):
    def __init__(self, kernel_size, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), padding_mode='SAME', return_indices=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, rank=3)

    def forward(self, x):
        return optoth.averagepooling.averagepooling3d(x, self.kernel_size, self.padding, self.stride, self.dilation,
                                              self.alpha, self.beta, self.padding_mode, x.dtype, self.channel_first)

    def backward(self, grad_output):
        return optoth.averagepooling.averagepooling3d_backward(TODO, TODO, TODO, self.kernel_size, self.padding, self.stride,
                                                       self.dilation,
                                                       self.alpha, self.beta, self.padding_mode, x.dtype,
                                                       self.channel_first)

class MagnitudeAveragePool3Dt(MagnitudeAveragePool):
    def __init__(self, kernel_size, stride=(2, 2, 2, 2), padding=(0, 0, 0, 0), dilation=(1, 1, 1, 1), padding_mode='SAME', return_indices=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, rank=4)

    def forward(self, x):
        return optoth.averagepooling.averagepooling4d(x, self.kernel_size, self.padding, self.stride, self.dilation,
                                              self.alpha, self.beta, self.padding_mode, x.dtype, self.channel_first)

    def backward(self, grad_output):
        return optoth.averagepooling.averagepooling4d_backward(TODO, TODO, TODO, self.kernel_size, self.padding, self.stride,
                                                       self.dilation,
                                                       self.alpha, self.beta, self.padding_mode, x.dtype,
                                                       self.channel_first)

# Aliases
MagnitudeAveragePool4D = MagnitudeAveragePool3Dt