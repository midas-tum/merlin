import unittest
import torch
import numpy as np
from merlinth.models.tdv import TDV

class GradientTest(unittest.TestCase):
    
    def test_tdv_gradient(self):
        # setup the data
        x = np.random.rand(2,1,64,64)
        x = torch.from_numpy(x).cuda()

        # define the TDV regularizer
        config ={
            'in_channels': 1,
            'out_channels': 1,
            'num_features': 4,
            'num_scales': 3,
            'num_mb': 2,
            'multiplier': 2,
        }
        R = TDV(config).double().cuda()

        def compute_loss(scale):
            return torch.sum(R.energy(scale*x))
        
        scale = 1.
        
        # compute the gradient using the implementation
        grad_scale = torch.sum(x*R.grad(scale*x)).item()

        # check it numerically
        epsilon = 1e-4
        with torch.no_grad():
            l_p = compute_loss(scale+epsilon).item()
            l_n = compute_loss(scale-epsilon).item()
            grad_scale_num = (l_p - l_n) / (2 * epsilon)

        condition = np.abs(grad_scale - grad_scale_num) < 1e-3
        print(f'grad_scale: {grad_scale:.7f} num_grad_scale {grad_scale_num:.7f} success: {condition}')
        self.assertTrue(condition)


if __name__ == "__main__":
    unittest.main()