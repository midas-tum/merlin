"""
Copyright (c) 2019 Imperial College London.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import unittest
import torch

from merlinth.losses.pairwise_loss import mse, nmse, psnr
from merlinpy.losses.pairwise_loss import mse as mse_py, nmse as nmse_py, psnr as psnr_py, ssim as ssim_py
from merlinth.losses.ssim import SSIM

def torch_to_numpy(data):
    return data.numpy()

def create_input_pair(shape, sigma=0.1):
    input = np.arange(np.product(shape)).reshape(shape).astype(float)
    input /= np.max(input)
    input2 = input + np.random.normal(0, sigma, input.shape)
    input = torch.from_numpy(input).float()
    input2 = torch.from_numpy(input2).float()
    return input, input2

class TestLosses(unittest.TestCase):
    def test_psnr(self):
        self._test_psnr([10, 32, 32])
        self._test_psnr([10, 64, 64])

    def _test_psnr(self, shape):
        input, input2 = create_input_pair(shape, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        err = psnr(input, input2, batch=False).item()
        err_numpy = psnr_py(input_numpy, input2_numpy)
        self.assertTrue(np.allclose(err, err_numpy))

    def test_psnr_batch(self):
        shape4d = [4, 6, 32, 32]
        input, input2 = create_input_pair(shape4d, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        err = psnr(input, input2, batch=True).item()
        err_numpy = 0
        for i in range(shape4d[0]):
            err_curr = psnr_py(input_numpy[i], input2_numpy[i])
            err_numpy += err_curr
        err_numpy /= shape4d[0]
        self.assertTrue(np.allclose(err, err_numpy))

        shape5d = [4, 6, 1, 32, 32]
        input, input2 = create_input_pair(shape5d, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        err = psnr(input, input2)
        err_numpy = 0
        for i in range(shape5d[0]):
            err_numpy += psnr_py(input_numpy[i][:,0], input2_numpy[i][:,0])
        err_numpy /= shape5d[0]

        self.assertTrue(np.allclose(err, err_numpy))

    def test_ssim(self):
        self._test_ssim([5, 320, 320])
        self._test_ssim([10, 64, 64])

    def _test_ssim(self, shape):
        input, input2 = create_input_pair(shape, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        torch_ssim = SSIM(win_size=7, device='cpu')

        err = torch_ssim(input.unsqueeze(1), input2.unsqueeze(1)).item()
        err_numpy = ssim_py(input_numpy, input2_numpy)
        self.assertTrue(abs(err - err_numpy) < 1e-4)

    def test_ssim_batch(self):
        shape4d = [4, 6, 96, 96]
        input, input2 = create_input_pair(shape4d, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        torch_ssim = SSIM(win_size=7, device='cpu')

        data_range = input.view(4, -1).max(1)[0].repeat(6)
        err = torch_ssim(
            input.reshape(24, 1, 96, 96),
            input2.reshape(24, 1, 96, 96),
            data_range=data_range,
        ).item()
        err_numpy = 0
        for i in range(shape4d[0]):
            err_curr = ssim_py(input_numpy[i], input2_numpy[i], win_size=7)
            err_numpy += err_curr
        err_numpy /= shape4d[0]
        self.assertTrue(abs(err - err_numpy) < 1e-3)

    def test_mse(self):
        self._test_mse([10, 32, 32])
        self._test_mse([10, 64, 64])
        self._test_mse([4, 6, 32, 32])
        self._test_mse([4, 6, 1, 32, 32])

    def _test_mse(self,shape):
        input, input2 = create_input_pair(shape, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        err = torch.nn.functional.mse_loss(input, input2).item()
        err_numpy = mse_py(input_numpy, input2_numpy)
        self.assertTrue(np.allclose(err, err_numpy))

    def test_mse_batch(self):
        shape4d = [4, 6, 32, 32]
        input, input2 = create_input_pair(shape4d, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        err = torch.nn.functional.mse_loss(input, input2).item()
        err_numpy = 0
        for i in range(shape4d[0]):
            err_curr = mse_py(input_numpy[i], input2_numpy[i])
            err_numpy += err_curr
        err_numpy /= shape4d[0]
        self.assertTrue(np.allclose(err, err_numpy))

        shape5d = [4, 6, 1, 32, 32]
        input, input2 = create_input_pair(shape5d, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        err = torch.nn.functional.mse_loss(input, input2).item()
        err_numpy = 0
        for i in range(shape5d[0]):
            err_numpy += mse_py(input_numpy[i][:,0], input2_numpy[i][:,0])
        err_numpy /= shape5d[0]

        self.assertTrue(np.allclose(err, err_numpy))

    def test_nmse(self):
        self._test_nmse([10, 32, 32])
        self._test_nmse([10, 64, 64])
        self._test_nmse([4, 6, 32, 32])
        self._test_nmse([4, 6, 1, 32, 32])

    def _test_nmse(self, shape):
        input, input2 = create_input_pair(shape, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        err = nmse(input, input2, batch=False).item()
        err_numpy = nmse_py(input_numpy, input2_numpy)
        self.assertTrue(np.allclose(err, err_numpy))

    def test_nmse_batch(self):
        shape4d = [4, 6, 32, 32]
        input, input2 = create_input_pair(shape4d, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        err = nmse(input, input2).item()
        err_numpy = 0
        for i in range(shape4d[0]):
            err_curr = nmse_py(input_numpy[i], input2_numpy[i])
            err_numpy += err_curr
        err_numpy /= shape4d[0]
        self.assertTrue(np.allclose(err, err_numpy))

        shape5d = [4, 6, 1, 32, 32]
        input, input2 = create_input_pair(shape5d, sigma=0.1)
        input_numpy = torch_to_numpy(input)
        input2_numpy = torch_to_numpy(input2)

        err = nmse(input, input2).item()
        err_numpy = 0
        for i in range(shape5d[0]):
            err_numpy += nmse_py(input_numpy[i][:,0], input2_numpy[i][:,0])
        err_numpy /= shape5d[0]

        self.assertTrue(np.allclose(err, err_numpy))

if __name__ == "__main__":
    unittest.main()