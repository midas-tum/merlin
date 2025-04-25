"""
Copyright (c) 2019 Imperial College London.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from merlinth.losses.ssim import SSIM

def define_losses():
    _ssim = SSIM(device='cuda')

    def mse(x, xgt, sample):
        tmp = (x - xgt) * sample['fg_mask_normedsqrt'].cuda()
        loss = (tmp ** 2).sum()
        return loss / xgt.size()[0]

    def l1(x, xgt, sample):
        tmp = abs(x - xgt) * sample['fg_mask'].cuda()
        loss = tmp.sum()
        return loss / xgt.size()[0]

    def ssim(x, xgt, sample):
        SSIM_SCALE = 100
        batchsize, nFE, nPE = xgt.size()
        mask = sample['fg_mask'].cuda()
        dynamic_range = sample['attrs']['ref_max'].cuda()
        _, ssimmap = _ssim(
            xgt.view(batchsize, 1, nFE, nPE),
            x.view(batchsize, 1, nFE, nPE),
            data_range=dynamic_range, full=True,
        )

        # only take the mean over the foreground
        ssimmap = ssimmap.view(batchsize, -1)
        mask = mask.contiguous().view(batchsize, -1)
        mask_norm = mask.sum(-1, keepdim=True)
        mask_norm = torch.max(mask_norm, torch.ones_like(mask_norm))
        ssim_val = (ssimmap * mask).sum(-1, keepdim=True) / mask_norm
        return (1 - ssim_val.mean()) * SSIM_SCALE

    def gradient_loss(x, xgt, sample):
        tmp = (x - xgt) * sample['fg_mask_normedsqrt'].cuda()
        grad_x = tmp[..., 1:] - tmp[..., :-1]
        grad_y = tmp[..., 1:, :] - tmp[..., :-1, :]
        loss = ((grad_x ** 2).sum() + (grad_y ** 2).sum())
        return loss / xgt.size()[0]

    return {
        'mse': mse,
        'l1': l1,
        'ssim': ssim,
        'gradient': gradient_loss,
    }
