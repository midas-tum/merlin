"""
Copyright (c) 2019 Imperial College London.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from torch import nn

class ComplexInstanceNorm(nn.Module):
    """ Motivated by 'Deep Complex Networks' (https://arxiv.org/pdf/1705.09792.pdf) """
    def __init__(self):
        super(ComplexInstanceNorm, self).__init__()
        self.mean = 0
        self.cov_xx_half = 1/np.sqrt(2)
        self.cov_xy_half = 0
        self.cov_yx_half = 0
        self.cov_yy_half = 1/np.sqrt(2)

    def matrix_invert(self, xx, xy, yx, yy):
        det = xx * yy - xy * yx
        return yy.div(det), -xy.div(det), -yx.div(det), xx.div(det)

    def complex_instance_norm(self, x, eps=1e-5):
        """ Operates on images x of size [nBatch, nSmaps, nFE, nPE, 2] """
        #shape = list(x.shape)
        x_combined = torch.sum(x, dim=1, keepdim=True)
        mean = x_combined.mean(dim=(1, 2, 3), keepdim=True)
        x_m = x - mean
        # var = complex_mult_conj(x_m, x_m).sum(dim=list(range(1, len(shape))), keepdim=True) / (shape[2]*shape[3] - 1)
        # self.std = torch.sqrt(var + eps)
        self.mean = mean
        self.complex_pseudocovariance(x_m)

    def complex_pseudocovariance(self, data):
        """ Data variable hast to be already mean-free!
            Operates on images x of size [nBatch, nSmaps, nFE, nPE, 2] """
        assert data.size(-1) == 2
        shape = data.shape

        # compute number of elements
        N = shape[2]*shape[3]

        # seperate real/imaginary channel
        re, im = torch.unbind(data, dim=-1)

        # dimensions is now length of original shape - 1 (because channels are seperated)
        dim = list(range(1, len(shape)-1))

        # compute covariance entries. cxy = cyx
        cxx = (re * re).sum(dim=dim, keepdim=True) / (N - 1)
        cyy = (im * im).sum(dim=dim, keepdim=True) / (N - 1)
        cxy = (re * im).sum(dim=dim, keepdim=True) / (N - 1)

        # Eigenvalue decomposition C = V*S*inv(V)
        # compute eigenvalues
        s1 = (cxx + cyy) / 2 - torch.sqrt((cxx + cyy)**2 / 4 - cxx * cyy + cxy**2)
        s2 = (cxx + cyy) / 2 + torch.sqrt((cxx + cyy)**2 / 4 - cxx * cyy + cxy**2)

        # compute eigenvectors
        v1x = s1 - cyy
        v1y = cxy
        v2x = s2 - cyy
        v2y = cxy

        # normalize eigenvectors
        norm1 = torch.sqrt(torch.sum(v1x * v1x + v1y * v1y, dim=dim, keepdim=True))
        norm2 = torch.sqrt(torch.sum(v2x * v2x + v2y * v2y, dim=dim, keepdim=True))

        v1x = v1x.div(norm1)
        v1y = v1y.div(norm1)

        v2x = v2x.div(norm2)
        v2y = v2y.div(norm2)

        # now we need the sqrt of the covariance matrix.
        # C^{-0.5} = V * sqrt(S) * inv(V)
        det = v1x * v2y - v2x * v1y
        s1 = torch.sqrt(s1).div(det)
        s2 = torch.sqrt(s2).div(det)

        self.cov_xx_half = v1x * v2y * s1 - v1y * v2x * s2
        self.cov_yy_half = v1x * v2y * s2 - v1y * v2x * s1
        self.cov_xy_half = v1x * v2x * (s2 - s1)
        self.cov_yx_half = v1y * v2y * (s1 - s2)

    def forward(self, input):
        #self.complex_instance_norm(input)
        return self.normalize(input)

    def set_normalization(self, mean, cov):
        self.mean = mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.cov_xx_half = cov[:,0].view(-1, 1, 1, 1)
        self.cov_xy_half = cov[:,1].view(-1, 1, 1, 1)
        self.cov_yx_half = cov[:,2].view(-1, 1, 1, 1)
        self.cov_yy_half = cov[:,3].view(-1, 1, 1, 1)

    def normalize(self, x):
        x_m = x - self.mean
        re, im = torch.unbind(x_m, dim=-1)

        cov_xx_half_inv, cov_xy_half_inv, cov_yx_half_inv, cov_yy_half_inv = self.matrix_invert(self.cov_xx_half, self.cov_xy_half, self.cov_yx_half, self.cov_yy_half)
        x_norm_re = cov_xx_half_inv * re + cov_xy_half_inv * im
        x_norm_im = cov_yx_half_inv * re + cov_yy_half_inv * im
        img = torch.stack([x_norm_re, x_norm_im], dim=-1)
        img = img.clamp(-6, 6)
        return img

    def unnormalize(self, x):
        re, im = torch.unbind(x, dim=-1)
        x_unnorm_re = self.cov_xx_half * re + self.cov_xy_half * im
        x_unnorm_im = self.cov_yx_half * re + self.cov_yy_half * im
        return torch.stack([x_unnorm_re, x_unnorm_im], dim=-1) + self.mean
