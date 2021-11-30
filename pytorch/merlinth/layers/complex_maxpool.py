
import torch
import merlinth
import unittest

class MagnitudeMaxPool(torch.nn.Module):
    def __init__(self, rank):
        super(MagnitudeMaxPool, self).__init__()

        if rank == 3:
            self.pool = torch.nn.MaxPool3d((1,2,2), (1,2,2), return_indices=True)
        elif rank == 2:
            self.pool = torch.nn.MaxPool2d((2,2), (2,2), return_indices=True)
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

class MagnitudeMaxPool3D(MagnitudeMaxPool):
    def __init__(self):
        super().__init__(3)

class MagnitudeMaxPool2D(MagnitudeMaxPool):
    def __init__(self):
        super().__init__(2)

class TestMagnitudePool(unittest.TestCase):
    def test3d(self):
        shape = (1,1,4,4,4,)
        x = merlinth.random_normal_complex(shape, dtype=torch.get_default_dtype())
        pool = MagnitudeMaxPool3D()

        y = pool(x)
        self.assertTrue(y.is_complex())

    def test2d(self):
        shape = (1,1,4,4,)
        x = merlinth.random_normal_complex(shape, dtype=torch.get_default_dtype())
        pool = MagnitudeMaxPool2D()

        y = pool(x)
        self.assertTrue(y.is_complex())

if __name__ == "__main__":
    unittest.test()
