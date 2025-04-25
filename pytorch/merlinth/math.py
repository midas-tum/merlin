
import torch

def dot(x, y, **kwargs):
    if torch.is_complex(x) and torch.is_complex(y):
        return torch.sum(torch.conj(x) * y, **kwargs)
    else:
        return torch.sum(x * y, **kwargs)
