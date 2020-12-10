
import torch

from .regularizer import *

from .conv import *
from optoth.activations import TrainableActivation


import numpy as np
import unittest

__all__ = ['FoE2D']

class FoE2D(Regularizer):
    def __init__(self, config=None, file=None):
        super(FoE2D, self).__init__()

        if (config is None and file is None) or \
            (not config is None and not file is None):
            raise RuntimeError('specify EITHER a config dictionary OR a `.pth`-file!')

        if not file is None:
            if not file.endswith('.pth'):
                raise ValueError('file needs to end with `.pth`!')
            checkpoint = torch.load(file)
            self.config = checkpoint['config']
            self.ckpt_state_dict = checkpoint['model']
            self.tau = checkpoint['tau']
        else:
            self.ckpt_state_dict = None
            self.tau = 1.0
            self.config = config

        self.K1 = Conv2d(**config['K1'])
        self.f1 = TrainableActivation(**self.config["f1"])

    def _transformation(self, x):
        return self.K1(x)

    def _activation(self, x):
        return self.f1(x) / x.shape[1]

    def _transformation_T(self, grad_out):
        return self.K1.backward(grad_out)

    def grad(self, x):
        x = self._transformation(x)
        x = self._activation(x)
        return self._transformation_T(x)

    def get_vis(self):
        raise NotImplementedError
