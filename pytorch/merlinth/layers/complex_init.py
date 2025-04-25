import numpy as np
import torch

def complex_independent_filters_init(weight, mode='glorot'):
    #print(f'Complex Independent Init {mode}')
    '''https://github.com/ChihebTrabelsi/deep_complex_networks'''
    assert mode in {'glorot', 'he'}

    num_fout = weight.shape[0]
    num_fin = weight.shape[1]
    kernel_size = weight.shape[2:-1]

    num_rows = num_fin * num_fout
    num_cols = np.prod(kernel_size)

    flat_shape = (int(num_rows), int(num_cols))

    r = np.random.uniform(size=flat_shape)
    i = np.random.uniform(size=flat_shape)
    z = r + 1j * i
    u, _, v = np.linalg.svd(z)
    unitary_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), np.conjugate(v).T))
    real_unitary = unitary_z.real
    imag_unitary = unitary_z.imag
            
    indep_real = np.reshape(real_unitary, (num_rows,) + tuple(kernel_size))
    indep_imag = np.reshape(imag_unitary, (num_rows,) + tuple(kernel_size))
    
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight[...,0])
    if mode == 'glorot':
        desired_var = 1. / (fan_in + fan_out)
    elif mode == 'he':
        desired_var = 1. / (fan_in)
    else:
        raise ValueError('Invalid criterion: ' + mode)

    multip_real = np.sqrt(desired_var / np.var(indep_real))
    multip_imag = np.sqrt(desired_var / np.var(indep_imag))
    scaled_real = multip_real * indep_real
    scaled_imag = multip_imag * indep_imag

    kernel_shape = (num_fout, num_fin, *kernel_size, 1)
    weight_np = np.concatenate([np.reshape(scaled_real, kernel_shape), np.reshape(scaled_imag, kernel_shape)], axis=-1)

    weight.data = torch.tensor(weight_np, dtype=weight.dtype)

def complex_init(weight, mode='glorot'):
    #print(f'Complex Init {mode}')
    '''https://github.com/ChihebTrabelsi/deep_complex_networks'''
    assert mode in {'glorot', 'he'}

    kernel_shape =  weight.shape[:-1]

    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight[...,0])

    if mode == 'glorot':
        s = 1. / (fan_in + fan_out)
    elif mode == 'he':
        s = 1. / fan_in
    else:
        raise ValueError('Invalid criterion: ' + mode)

    modulus = np.random.rayleigh(scale=np.sqrt(s), size=kernel_shape)
    phase = np.random.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)
    weight_np = np.concatenate([weight_real[...,None], weight_imag[...,None]], axis=-1)
    weight.data = torch.tensor(weight_np, dtype=weight.dtype)