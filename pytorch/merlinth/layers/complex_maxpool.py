import torch
import merlinth
try:
    import optoth.maxpooling
except:
    print('optoth could not be imported')

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
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, padding_mode='SAME', return_indices=False, rank=2):
        super(MagnitudeMaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.alpha = 1
        self.beta = 1
        self.padding_mode = padding_mode
        self.return_indices = return_indices
        self.channel_first = True  # default PyTorch order: [N, C, H, W, ....]
        self.optox = optox and (True if 'optoth.maxpooling' in sys.modules else False)

        if not self.optox:
            if rank == 3:
                self.pool = torch.nn.MaxPool3d(kernel_size, stride, return_indices=True)
            elif rank == 2:
                self.pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True)
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


class MagnitudeMaxPool2D(MagnitudeMaxPool):
    def __init__(self, kernel_size, stride=(2, 2), padding=(0, 0), dilation=(1, 1), padding_mode='SAME', return_indices=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, rank=2)

    def forward(self, x):
        return optoth.maxpooling.maxpooling2d(x, self.kernel_size, self.padding, self.stride, self.dilation,
                                              self.alpha, self.beta, self.padding_mode, x.dtype, self.channel_first)

    def backward(self, grad_output):
        return optoth.maxpooling.maxpooling2d_backward(TODO, TODO, TODO, self.kernel_size, self.padding, self.stride, self.dilation,
                                                       self.alpha, self.beta, self.padding_mode, x.dtype, self.channel_first)

class MagnitudeMaxPool3D(MagnitudeMaxPool):
    def __init__(self, kernel_size, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), padding_mode='SAME', return_indices=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, rank=3)

    def forward(self, x):
        return optoth.maxpooling.maxpooling3d(x, self.kernel_size, self.padding, self.stride, self.dilation,
                                              self.alpha, self.beta, self.padding_mode, x.dtype, self.channel_first)

    def backward(self, grad_output):
        return optoth.maxpooling.maxpooling3d_backward(TODO, TODO, TODO, self.kernel_size, self.padding, self.stride,
                                                       self.dilation,
                                                       self.alpha, self.beta, self.padding_mode, x.dtype,
                                                       self.channel_first)

class MagnitudeMaxPool2Dt(MagnitudeMaxPool):
    def __init__(self, kernel_size, stride=(2, 2, 2), padding=(0, 0, 0), dilation=(1, 1, 1), padding_mode='SAME', return_indices=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, rank=3)

    def forward(self, x):
        return optoth.maxpooling.maxpooling3d(x, self.kernel_size, self.padding, self.stride, self.dilation,
                                              self.alpha, self.beta, self.padding_mode, x.dtype, self.channel_first)

    def backward(self, grad_output):
        return optoth.maxpooling.maxpooling3d_backward(TODO, TODO, TODO, self.kernel_size, self.padding, self.stride,
                                                       self.dilation,
                                                       self.alpha, self.beta, self.padding_mode, x.dtype,
                                                       self.channel_first)

class MagnitudeMaxPool3Dt(MagnitudeMaxPool):
    def __init__(self, kernel_size, stride=(2, 2, 2, 2), padding=(0, 0, 0, 0), dilation=(1, 1, 1, 1), padding_mode='SAME', return_indices=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, rank=4)

    def forward(self, x):
        return optoth.maxpooling.maxpooling4d(x, self.kernel_size, self.padding, self.stride, self.dilation,
                                              self.alpha, self.beta, self.padding_mode, x.dtype, self.channel_first)

    def backward(self, grad_output):
        return optoth.maxpooling.maxpooling4d_backward(TODO, TODO, TODO, self.kernel_size, self.padding, self.stride,
                                                       self.dilation,
                                                       self.alpha, self.beta, self.padding_mode, x.dtype,
                                                       self.channel_first)

# Aliases
MagnitudeMaxPool4D = MagnitudeMaxPool3Dt