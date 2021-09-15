import torch

def complex_abs(data, dim=-1, keepdim=False, eps=0):
    assert data.size(dim) == 2
    return (data ** 2 + eps).sum(dim=dim, keepdim=keepdim).sqrt()

def complex_normalization(data, dim=-1, eps=1e-12):
    assert data.size(dim) == 2
    magn = complex_abs(data, eps=eps, keepdim=True)
    return data / magn

def complex_angle(data, dim=-1, keepdim=False, eps=1e-12):
    assert data.size(dim) == 2
    re, im = torch.unbind(data, dim=dim)
    angle = torch.atan2(im, re + eps)
    if keepdim:
        return angle.unsqueeze(dim)
    else:
        return angle
        
def complex_div(data1, data2, dim=-1):
    assert data1.size(dim) == 2
    assert data2.size(dim) == 2
    re1, im1 = torch.unbind(data1, dim=dim)
    re2, im2 = torch.unbind(data2, dim=dim)
    return torch.stack([re1 * re2 + im1 * im2, im1 * re2 - re1 * im2], dim = dim)/complex_abs(data2, keepdim=True)**2

def complex_inv(data, dim=-1):
    assert data.size(dim) == 2
    re, im = torch.unbind(data, dim=dim)
    return torch.stack([re, -im], dim = dim)/complex_abs(data, keepdim=True)**2

def complex_mult(data1, data2, dim=-1):
    """
    Element-wise complex matrix multiplication X^T Y
    Params:
      data1: tensor
      data2: tensor
      dim: dimension that represents the complex values
    """
    assert data1.size(dim) == 2
    assert data2.size(dim) == 2
    re1, im1 = torch.unbind(data1, dim=dim)
    re2, im2 = torch.unbind(data2, dim=dim)
    return torch.stack([re1 * re2 - im1 * im2, im1 * re2 + re1 * im2], dim = dim)

def complex_mult_conj(data1, data2, dim=-1):
    """
    Element-wise complex matrix multiplication with conjugation X^H Y
    Params:
      data1: tensor
      data2: tensor
      dim: dimension that represents the complex values
    """
    assert data1.size(dim) == 2
    assert data2.size(dim) == 2
    re1, im1 = torch.unbind(data1, dim=dim)
    re2, im2 = torch.unbind(data2, dim=dim)
    return torch.stack([re1 * re2 + im1 * im2, im1 * re2 - re1 * im2], dim = -1)

def complex_conj(data):
    assert data.size(-1) == 2
    data[...,1] *= -1
    return data

def complex_dotp(data1, data2):
    assert data1.size(-1) == 2
    assert data2.size(-1) == 2
    mult = complex_mult_conj(data1, data2)
    re, im = torch.unbind(mult, dim=-1)
    return torch.stack([torch.sum(re), torch.sum(im)])

def complex2real(z, channel_last=False):
    stack_dim = -1 if channel_last else 1
    re, im = torch.unbind(z, dim=-1)
    return torch.cat([re, im], stack_dim)

def real2complex(z, channel_last=False):
    stack_dim = -1 if channel_last else 1
    (re, im) = torch.chunk(z, 2, dim=stack_dim)
    return torch.stack([re, im], -1)

if __name__ == "__main__":
    x = torch.randn(10, 1, 5, 5, 2)
    y = complex2real(x)
    print(y.shape)
    y = real2complex(y)
    print(y.shape)
    assert y.shape == x.shape
