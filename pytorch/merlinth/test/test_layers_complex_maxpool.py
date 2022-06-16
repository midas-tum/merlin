import unittest
import torch
import merlinth
import numpy as np

from merlinth.layers.complex_maxpool import (
    MagnitudeMaxPool1D,
    MagnitudeMaxPool2D,
    MagnitudeMaxPool3D,
    MagnitudeMaxPool4D
)

class TestMagnitudePool(unittest.TestCase):
    def test4d(self):
        self._test((1, 4, 6, 6, 6, 2), (2, 2, 2, 2), (2, 2, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 'valid')
        self._test((1, 4, 6, 6, 6, 2), (2, 2, 2, 2), (2, 2, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 'same')
        self._test((1, 5, 7, 7, 7, 2), (2, 2, 2, 2), (2, 2, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 'valid')
        self._test((1, 5, 7, 7, 7, 2), (2, 2, 2, 2), (2, 2, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 'same')

    def test3d(self):
        self._test((1, 4, 6, 6, 2), (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), 'valid')
        self._test((1, 4, 6, 6, 2), (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), 'same')
        self._test((1, 5, 7, 7, 2), (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), 'valid')
        self._test((1, 5, 7, 7, 2), (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), 'same')

        self._verify_shape((1, 4, 6, 6, 2), (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), 'valid')
        self._verify_shape((1, 4, 6, 6, 2), (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), 'same')
        self._verify_shape((1, 5, 7, 7, 2), (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), 'valid')
        self._verify_shape((1, 5, 7, 7, 2), (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1), 'same')

    def test2d(self):
        self._test((1, 4, 6, 2), (2, 2), (2, 2), (0, 0), (1, 1), 'valid')
        self._test((1, 4, 6, 2), (2, 2), (2, 2), (0, 0), (1, 1), 'same')
        self._test((1, 5, 7, 2), (2, 2), (2, 2), (0, 0), (1, 1), 'valid')
        self._test((1, 5, 7, 2), (2, 2), (2, 2), (0, 0), (1, 1), 'same')

        self._verify_shape((1, 4, 6, 2), (2, 2), (2, 2), (0, 0), (1, 1), 'valid')
        self._verify_shape((1, 4, 6, 2), (2, 2), (2, 2), (0, 0), (1, 1), 'same')
        self._verify_shape((1, 5, 7, 2), (2, 2), (2, 2), (0, 0), (1, 1), 'valid')
        self._verify_shape((1, 5, 7, 2), (2, 2), (2, 2), (0, 0), (1, 1), 'same')

    def test1d(self):
        self._test((1, 4, 2), (2,), (2,), (0,), (1,), 'valid')
        self._test((1, 4, 2), (2,), (2,), (0,), (1,), 'same')
        self._test((1, 5, 2), (2,), (2,), (0,), (1,), 'valid')
        self._test((1, 5, 2), (2,), (2,), (0,), (1,), 'same')

        self._verify_shape((1, 4, 2), (2,), (2,), (0,), (1,), 'valid')
        self._verify_shape((1, 4, 2), (2,), (2,), (0,), (1,), 'same')
        self._verify_shape((1, 5, 2), (2,), (2,), (0,), (1,), 'valid')
        self._verify_shape((1, 5, 2), (2,), (2,), (0,), (1,), 'same')

    def _padding_shape(self, input_spatial_shape, spatial_filter_shape, strides, dilations_rate, padding_mode):
        if padding_mode.lower() == 'valid':
            return np.ceil((input_spatial_shape - (spatial_filter_shape - 1) * dilations_rate) / strides)
        elif padding_mode.lower() == 'same':
            return np.ceil(input_spatial_shape / strides)
        else:
            raise Exception('padding_mode can be only valid or same!')

    def _verify_shape(self, shape, pool_size, strides, padding, dilations_rate, padding_mode='same'):
        x = merlinth.random_normal_complex(shape, dtype=torch.get_default_dtype())
        cuda1 = torch.device('cuda:0')
        x = x.to(device=cuda1)

        if len(shape) == 3:  # 1d
            op = MagnitudeMaxPool1D(pool_size, strides, padding, padding_mode=padding_mode)
            if padding_mode.lower() == 'valid':
                op_backend = torch.nn.MaxPool1d(pool_size, strides, padding, ceil_mode=False)
            else:
                op_backend = torch.nn.MaxPool1d(pool_size, strides, padding, ceil_mode=True)
        elif len(shape) == 4:  # 2d
            op = MagnitudeMaxPool2D(pool_size, strides, padding, padding_mode=padding_mode)
            if padding_mode.lower() == 'valid':
                op_backend = torch.nn.MaxPool2d(pool_size, strides, padding, ceil_mode=False)
            if padding_mode.lower() == 'same':
                op_backend = torch.nn.MaxPool2d(pool_size, strides, padding, ceil_mode=True)

        elif len(shape) == 5:  # 3d
            op = MagnitudeMaxPool3D(pool_size, strides, padding, padding_mode=padding_mode)
            if padding_mode.lower() == 'valid':
                op_backend = torch.nn.MaxPool3d(pool_size, strides, padding, ceil_mode=False)
            if padding_mode.lower() == 'same':
                op_backend = torch.nn.MaxPool3d(pool_size, strides, padding, ceil_mode=True)
        elif len(shape) == 6:  # 4d
            op = MagnitudeMaxPool4D(pool_size, strides, padding, padding_mode=padding_mode)

        out = op(x)
        out_backend = op_backend(merlinth.complex_abs(x))

        self.assertTrue(np.sum(np.abs(np.array(out.shape) - np.array(out_backend.shape))) == 0)

    def _test(self, shape, pool_size, strides, padding, dilations_rate, padding_mode='same'):
        x = merlinth.random_normal_complex(shape, dtype=torch.get_default_dtype())
        cuda1 = torch.device('cuda:0')
        x = x.to(device=cuda1)
        x.requires_grad_(True)

        if len(shape) == 3:  # 1d
            op = MagnitudeMaxPool1D(pool_size, strides, padding, dilations_rate, padding_mode=padding_mode).cuda()
        elif len(shape) == 4:  # 2d
            op = MagnitudeMaxPool2D(pool_size, strides, padding, dilations_rate, padding_mode=padding_mode).cuda()
        elif len(shape) == 5:  # 3d
            op = MagnitudeMaxPool3D(pool_size, strides, padding, dilations_rate, padding_mode=padding_mode).cuda()
        elif len(shape) == 6:  # 4d
            op = MagnitudeMaxPool4D(pool_size, strides, padding, dilations_rate, padding_mode=padding_mode).cuda()

        out_complex = op(x)
        out_complex.sum().backward()

        # (N, T, H, W, D, C)
        expected_shape = [shape[0]]
        for i in range(len(shape) - 2):
            expected_shape.append(
                self._padding_shape(shape[i + 1], pool_size[i], strides[i], dilations_rate[i], padding_mode))
        expected_shape.append(shape[-1])

        self.assertTrue(np.abs(np.array(expected_shape) - np.array(out_complex.shape)).all() < 1e-8)
        self.assertTrue(np.abs(np.array(expected_shape) - np.array(x.grad.shape)).all() < 1e-8)

if __name__ == "__main__":
    unittest.main()
