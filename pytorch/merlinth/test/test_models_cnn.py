import unittest
import torch
import numpy as np
from merlinth.models.cnn import Real2chCNN, ComplexCNN

class TestCNN(unittest.TestCase):
    def testCnn(self):
        input_dim = 1
        x = np.random.randn(5, input_dim, 11, 11) + 1j * np.random.randn(
            5, input_dim, 11, 11
        )
        op = Real2chCNN(input_dim=input_dim).double()
        y = op(torch.from_numpy(x))
        print(op)

    def testCnnComplex(self):
        input_dim = 1
        x = np.random.randn(5, input_dim, 11, 11) + 1j * np.random.randn(
            5, input_dim, 11, 11
        )
        op = ComplexCNN(input_dim=input_dim).double()
        y = op(torch.from_numpy(x))
        print(op)