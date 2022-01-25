import os
import numpy as np
import tensorflow as tf
import unittest
from merlintf.keras.layers.data_consistency import DCPM
from merlintf.keras.layers.mri import MulticoilForwardOp, MulticoilAdjointOp


def recon(kspace, smap, mask=None, noisy=None, max_iter=10, tol=1e-12, weight_init=1.0, weight_scale=1.0):
    # kspace        raw k-space data as [batch, coils, X, Y] or [batch, coils, X, Y, Z] or [batch, coils, time, X, Y] or [batch, coils, time, X, Y, Z]
    # smap          coil sensitivity maps with same shape as kspace (or singleton dimension for time)
    # mask          subsampling including/excluding soft-weights with same shape as kspace
    # noisy         initialiaztion for reconstructed image, if None it is created from A^H(kspace)
    # max_iter      maximum number of iterations for CG/iterative SENSE
    # tol           tolerance for stopping condition for CG/iterative SENSE
    # weight_init   initial weighting for lambda regularization parameter
    # weight_scale  scaling factor for lambda regularization parameter

    # Forward and Adjoint operators
    A = MulticoilForwardOp(center=True)
    AH = MulticoilAdjointOp(center=True)

    if mask is None:
        mask = tf.ones(kshape, dtype=tf.float64)

    if noisy is None:
        noisy = AH(kspace, mask, smaps)

    model = DCPM(A, AH, weight_init=weight_init, weight_scale=weight_scale, max_iter=max_iter, tol=tol)

    return model([noisy, kspace, mask, smap])

class ItSenseTest(unittest.TestCase):
    def test_iterativeSENSE(self, acc=4):
        kspace = bart(1, 'phantom -3 -x 64 -k')
        smap = bart(1, 'phantom -3 -x 64 -S 8')

        reconimg = recon(kspace, smap)
        self.assertTrue(np.shape(reconimg) == (64, 64, 64))


if __name__ == "__main__":
    importsuccess = merlinpy.recon.BART.setup_bart('/home/gitaction/bart')
    if importsuccess:
        unittest.main()