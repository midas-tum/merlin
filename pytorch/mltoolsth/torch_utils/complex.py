import torch

def complex_abs(data, dim=-1, keepdim=False, eps=0):
    assert data.size(dim) == 2
    return (data ** 2 + eps).sum(dim=dim, keepdim=keepdim).sqrt()

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
