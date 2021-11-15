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

def _get_np_float_dtype(complex_dtype):
    """ Get equivalent float type given current complex dtype """
    if complex_dtype == np.complex64:
        return np.float32
    elif complex_dtype == np.complex128:
        return np.float64
    else:
        return np.float128

def _get_np_complex_dtype(float_dtype):
    """ Get equivalent complex type given current float dtype """
    if float_dtype == np.float32:
        return np.complex64
    elif float_dtype == np.float64:
        return np.complex128
    else:
        return np.complex256

def torch_to_complex_numpy(data):
    data = data.numpy()
    complex_dtype = _get_np_complex_dtype(data.dtype)
    return np.ascontiguousarray(data).view(complex_dtype).squeeze(-1)

def torch_to_complex_abs_numpy(data):
    data = data.numpy()
    return np.abs(data[..., 0] + 1j * data[..., 1])

def torch_to_numpy(data):
    return data.numpy()

def numpy_to_torch(data):
    if np.iscomplexobj(data):
        float_dtype = _get_np_float_dtype(data.dtype)
        data = np.ascontiguousarray(data[..., np.newaxis]).view(float_dtype)
    return torch.from_numpy(data)

def numpy_to_torch_float(arr):
    if np.iscomplexobj(arr):
        arr = arr.astype(np.complex64)
    else:
        arr = arr.astype(np.float32)
    return numpy_to_torch(arr)

class ToTorchIO():
    def __init__(self, input_keys, output_keys):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, sample):
        inputs = []
        outputs = []
        for key in self.input_keys:
            inputs.append(numpy_to_torch_float(sample[key]))
        for key in self.output_keys:
            outputs.append(numpy_to_torch_float(sample[key]))
        return inputs, outputs

class ToTorchCuda():
    def __call__(self, inputs):
        if isinstance(inputs, list):
            return [inp.cuda() for inp in inputs]
        else:
            return inputs.cuda()