import torch

def complex_abs(z, eps=1e-9):
    return torch.sqrt(torch.real(torch.conj(z) * z) + eps)

def complex_norm(z, eps=1e-9):
    return z / complex_abs(z, eps)

def complex_angle(z, eps=1e-9):
    return torch.atan2(torch.imag(z), torch.real(z) + eps)

def complex_scale(x, scale):
    return torch.complex(torch.real(x) * scale, torch.imag(x) * scale)
    
def complex_dot(x, y, **kwargs):
    return torch.sum(torch.conj(x) * y, **kwargs)

def complex2real(z, channel_last=False):
    stack_dim = -1 if channel_last else 1
    return torch.cat([torch.real(z), torch.imag(z)], stack_dim)

def real2complex(z, channel_last=False):
    stack_dim = -1 if channel_last else 1
    (real, imag) = torch.chunk(z, 2, axis=stack_dim)
    return torch.complex(real, imag)

def complex2magpha(z, channel_last=False):
    stack_dim = -1 if channel_last else 1
    return torch.cat([complex_abs(z), complex_angle(z)], stack_dim)

def magpha2complex(z, channel_last=False):
    stack_dim = -1 if channel_last else 1
    (mag, pha) = torch.chunk(z, 2, axis=stack_dim)
    return torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))

def random_normal_complex(shape, dtype=torch.float32):
    return torch.complex(torch.randn(shape, dtype=dtype), 
                         torch.randn(shape, dtype=dtype))

def iscomplex(x):
    if x.dtype == torch.complex32 or x.dtype == torch.complex64 or x.dtype == torch.complex128:
          return True
    else:
          return False

# def complex_div(data1, data2, dim=-1):
#     assert data1.size(dim) == 2
#     assert data2.size(dim) == 2
#     re1, im1 = torch.unbind(data1, dim=dim)
#     re2, im2 = torch.unbind(data2, dim=dim)
#     return torch.stack([re1 * re2 + im1 * im2, im1 * re2 - re1 * im2], dim = dim)/complex_abs(data2, keepdim=True)**2

# def complex_inv(data, dim=-1):
#     assert data.size(dim) == 2
#     re, im = torch.unbind(data, dim=dim)
#     return torch.stack([re, -im], dim = dim)/complex_abs(data, keepdim=True)**2