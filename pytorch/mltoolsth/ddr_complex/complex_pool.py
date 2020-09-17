
import torch
from . import mytorch
import unittest

class MagnitudeMaxPool3d(torch.nn.Module):
    def __init__(self):
        super(MagnitudeMaxPool3d, self).__init__()
    
        self.pool = torch.nn.MaxPool3d((1,2,2), (1,2,2), return_indices=True)
    
    def forward(self, x):
        eps=1e-9
        magn = mytorch.complex.complex_abs(x, eps=eps)
        _, indices = self.pool(magn)
        pool_re = self.retrieve_elements_from_indices(x[...,0], indices).unsqueeze_(-1)
        pool_im = self.retrieve_elements_from_indices(x[...,1], indices).unsqueeze_(-1)
        return torch.cat([pool_re, pool_im], -1)

    def retrieve_elements_from_indices(self, tensor, indices):
        flattened_tensor = tensor.flatten(start_dim=2)
        output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        return output


class TestMagnitudePool3d(unittest.TestCase):
    def test3d(self):
        x = torch.randn(1,1,4,4,4,2)
        pool = MagnitudeMaxPool3d()

        y = pool(x)
        magn = mytorch.complex.complex_abs(y, eps=1e-9)


if __name__ == "__main__":
    unittest.test()
