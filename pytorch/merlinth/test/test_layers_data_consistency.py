import unittest
import torch
import numpy as np
from merlinth.layers.mri import MulticoilForwardOp, MulticoilAdjointOp
from merlinth.layers.data_consistency import DCPM
import merlinth

class CgTest(unittest.TestCase):
    def testcg(self):
        A = MulticoilForwardOp().double()
        AH = MulticoilAdjointOp().double()

        dc = DCPM(A, AH, weight_init=1.0, max_iter=100, tol=1e-18)

        shape=(5,1,10,10)
        kshape=(5,3,10,10)
        x = merlinth.random_normal_complex(shape, torch.float64)
        y = merlinth.random_normal_complex(kshape, torch.float64)
        mask = torch.ones(*kshape, dtype=torch.float64)
        smaps = merlinth.random_normal_complex(kshape, torch.float64)

        th_a = torch.tensor(1.1, requires_grad=True, dtype=torch.float64)
        th_b = torch.tensor(1.1, requires_grad=True, dtype=torch.float64)

        # perform a gradient check:
        epsilon = 1e-5

        def compute_loss(a, b):
            arg = dc([x*a, y, mask, smaps], scale=1/b) # take 1/b
            return 0.5 * torch.sum(torch.real(torch.conj(arg) * arg))

        th_loss = compute_loss(th_a, th_b)

        # backpropagate the gradient
        th_loss.backward()
        grad_a = th_a.grad.cpu().numpy()
        grad_b = th_b.grad.cpu().numpy()

        # numerical gradient w.r.t. the input
        with torch.no_grad():
            l_ap = compute_loss(th_a+epsilon, th_b).cpu().numpy()
            l_an = compute_loss(th_a-epsilon, th_b).cpu().numpy()
            grad_a_num = (l_ap - l_an) / (2 * epsilon)

        print("grad_x: {:.7f} num_grad_x {:.7f} success: {}".format(
            grad_a, grad_a_num, np.abs(grad_a - grad_a_num) < 1e-4))
        self.assertTrue(np.abs(grad_a - grad_a_num) < 1e-4)

        # numerical gradient w.r.t. the weights
        with torch.no_grad():
            l_bp = compute_loss(th_a, th_b+epsilon).cpu().numpy()
            l_bn = compute_loss(th_a, th_b-epsilon).cpu().numpy()
            grad_b_num = (l_bp - l_bn) / (2 * epsilon)

        print("grad_w: {:.7f} num_grad_w {:.7f} success: {}".format(
            grad_b, grad_b_num, np.abs(grad_b - grad_b_num) < 1e-4))
        self.assertTrue(np.abs(grad_b - grad_b_num) < 1e-4)
        
if __name__ == "__main__":
    unittest.main()
