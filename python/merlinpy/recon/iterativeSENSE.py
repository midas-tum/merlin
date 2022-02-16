import numpy as np
import tensorflow as tf
from merlintf.keras.layers.data_consistency import DCPM
from merlintf.keras.layers.mri import MulticoilForwardOp, MulticoilAdjointOp

def recon(kspace, smap, mask=None, noisy=None, channel_dim_defined=True, max_iter=10, tol=1e-12, weight_init=1.0, weight_scale=1.0):
    # kspace        raw k-space data as [batch, coils, X, Y] or [batch, coils, X, Y, Z] or [batch, coils, time, X, Y] or [batch, coils, time, X, Y, Z]
    # smap          coil sensitivity maps with same shape as kspace (or singleton dimension for time)
    # mask          subsampling including/excluding soft-weights with same shape as kspace
    # noisy         initialiaztion for reconstructed image, if None it is created from A^H(kspace)
    # max_iter      maximum number of iterations for CG/iterative SENSE
    # tol           tolerance for stopping condition for CG/iterative SENSE
    # weight_init   initial weighting for lambda regularization parameter
    # weight_scale  scaling factor for lambda regularization parameter

    # Forward and Adjoint operators
    A = MulticoilForwardOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined)
    AH = MulticoilAdjointOp(center=True, coil_axis=1, channel_dim_defined=channel_dim_defined)

    if mask is None:
        mask = tf.ones(np.shape(kspace), dtype=tf.float32)

    if noisy is None:
        noisy = AH(kspace, mask, smap)[..., 0]

    model = DCPM(A, AH, weight_init=weight_init, weight_scale=weight_scale, max_iter=max_iter, tol=tol)

    return model([noisy, kspace, mask, smap])
