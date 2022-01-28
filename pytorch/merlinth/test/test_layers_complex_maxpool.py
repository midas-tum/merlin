import unittest
import torch
import merlinth

from merlinth.layers.complex_maxpool import (
    MagnitudeMaxPool2D,
    MagnitudeMaxPool3D
)

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
    unittest.main()
