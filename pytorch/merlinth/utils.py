import torch
import numpy as np

def get_ndim(dim):
    if dim == '1D' or dim == 1:
        n_dim = 1
    elif dim == '2D' or dim == 2:
        n_dim = 2
    elif dim == '3D' or dim == 3:
        n_dim = 3
    elif dim == '2Dt':
        n_dim = 3
    elif dim == '3Dt':
        n_dim = 4
    return n_dim

def validate_input_dimension(dim, param):
    n_dim = get_ndim(dim)
    if isinstance(param, tuple) or isinstance(param, list):
        if not len(param) == n_dim:
            raise RuntimeError("Parameter dimensions {} do not match requested dimensions {}!".format(len(param), n_dim))
        else:
            return param
    else:
        return tuple([param for _ in range(n_dim)])

def get_default_cdtype():
    if torch.get_default_dtype() == torch.float32:
        return torch.complex64
    elif torch.get_default_dtype() == torch.float64:
        return torch.complex128
    elif torch.get_default_dtype() == torch.float16:
        return torch.complex32
    else:
        raise ValueError(f"No equivalent for dtype='{torch.get_default_dtype()}' ")

class ToTorchIO():
    def __init__(self, input_keys, output_keys):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, sample):
        inputs = []
        outputs = []
        for key in self.input_keys:
            inputs.append(torch.from_numpy(sample[key]))
        for key in self.output_keys:
            outputs.append(torch.from_numpy(sample[key]))
        return inputs, outputs

class AddTorchChannelDim(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = sample[key][:, None]
        return sample

class Transpose():
    def __init__(self, transpose_list):
        self.transpose_list = transpose_list

    def __call__(self, sample):
        for key, axes in self.transpose_list:
            sample[key] = np.ascontiguousarray(np.transpose(sample[key], axes))
        return sample

class ToTorchCuda():
    def __call__(self, inputs):
        if isinstance(inputs, list):
            return [inp.cuda() for inp in inputs]
        else:
            return inputs.cuda()