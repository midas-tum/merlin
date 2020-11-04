import tensorflow as tf
import tensorflow.keras.backend as K
import unittest
import numpy as np

class ComplexNormalizationBase(tf.keras.layers.Layer):
    def __init__(self,
                channel_last=True,
                epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.channel_last = channel_last
        self.reduction_axes = []

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.channel_last:
            reduction_axes = list(range(1, ndim-1))
        else:
            reduction_axes = list(range(2, ndim))
        self.reduction_axes += reduction_axes

    def call(self, x):
        return self.whiten2x2(x, self.reduction_axes)
    
    def whiten2x2(self, x, reduction_axes):
        # reduction_axes consider for N-d
        #   - layer norm
        #   - batch norm
        #   - instance norm
        #   - channel_first / channel_last
        # print(reduction_axes)

        # 1. compute mean regarding the specified axes
        mu = K.mean(x, axis=reduction_axes, keepdims=True)
        x = x - mu

        xre = tf.math.real(x)
        xim = tf.math.imag(x)

        # 2. construct 2x2 covariance matrix
        # Stabilize by a small epsilon.
        cov_uu = K.var(xre, axis=reduction_axes, keepdims=True) +  self.epsilon
        cov_vv = K.var(xim, axis=reduction_axes, keepdims=True) +  self.epsilon
        cov_vu = cov_uv = K.mean(xre * xim, axis=reduction_axes, keepdims=True)

        # 3. get R = [[p, q], [r, s]], with E R c c^T R^T = R M R = I
        sqrdet = K.sqrt(cov_uu * cov_vv - cov_uv * cov_vu )
        denom = sqrdet * K.sqrt(cov_uu + 2 * sqrdet + cov_vv)

        p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
        r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom

        # 4. apply R to x (manually)
        out_re = xre * p + xim * r
        out_im = xre * q + xim * s

        return tf.complex(out_re, out_im)

class ComplexInstanceNormalization(ComplexNormalizationBase):
    def __init__(self,
                channel_last=True,
                epsilon=1e-4):
        super().__init__(channel_last=channel_last, epsilon=epsilon)
        # normalization along spatial dimension
        # [B, ..., F] or [B, F, ...]
        self.reduction_axes = []

class ComplexLayerNormalization(ComplexNormalizationBase):
    def __init__(self,
                channel_last=True,
                epsilon=1e-4):
        super().__init__(channel_last=channel_last, epsilon=epsilon)
        # normalization along spatial & dimension
        # [B, ...]
        if channel_last:
            self.reduction_axes = [-1,]
        else:
            self.reduction_axes = [1,]

# class ComplexBatchNormalization(ComplexNormalizationBase):
#     def __init__(self,
#                 channel_last=True,
#                 epsilon=1e-4):
#         super().__init__(channel_last=channel_last, epsilon=epsilon)
#         # normalization along spatial & dimension
#         # [..., F] or [:, F, ...]
#         self.reduction_axes = [0]

class ComplexNormTest(unittest.TestCase):
    def _test_norm(self, shape, channel_last=True, layer_norm=False):

        if layer_norm:
            model = ComplexLayerNormalization(channel_last=channel_last)
        else:
            model = ComplexInstanceNormalization(channel_last=channel_last)
        
        x = tf.complex(tf.random.normal(shape), tf.random.normal(shape)*2)
        xn = model(x)

        if channel_last:
            axes=tuple(range(1, tf.rank(x)-1))
            if layer_norm:
                axes += (-1,)
        else:
            axes=tuple(range(2, tf.rank(x)))
            if layer_norm:
                axes += (1,)

        #print('test axes', axes)

        xnre = tf.math.real(xn)
        xnim = tf.math.imag(xn)

        np_mu = K.mean(xn, axes).numpy()
        #print('mean', np_mu)
        self.assertTrue(np.linalg.norm(np_mu) < 1e-6)
        #print(xn.shape)
        uu = K.var(xnre, axes).numpy()
        vv = K.var(xnim, axes).numpy()
        uv = K.mean(xnre * xnim, axes).numpy()
        # print('vv', f'{vv}')
        # print('vv', f'{vv}')
        # print('uv', f'{uv}')
        self.assertTrue(np.linalg.norm(uu - 1) < 1e-3)
        self.assertTrue(np.linalg.norm(vv - 1) < 1e-3)
        self.assertTrue(np.linalg.norm(uv) < 1e-6)
        # uu = K.var(tf.math.real(x), axes).numpy()
        # vv = K.var(tf.math.imag(x), axes).numpy()
        # uv = K.mean(tf.math.real(x) * tf.math.imag(x), axes).numpy()

    def test1_instance(self):
        self._test_norm([1, 200, 200, 1], channel_last=True)

    def test2_instance(self):
        self._test_norm([3, 2, 320, 320], channel_last=False)

    def test3_instance(self):
        self._test_norm([3, 10, 320, 320, 2], channel_last=True)

    def test4_instance(self):
        self._test_norm([3, 2, 10, 320, 320], channel_last=False)

    def test1_layer(self):
        self._test_norm([3, 320, 320, 2], channel_last=True, layer_norm=True)

    def test2_layer(self):
        self._test_norm([3, 2, 320, 320], channel_last=False, layer_norm=True)

    def test3_layer(self):
        self._test_norm([3, 10, 320, 320, 2], channel_last=True, layer_norm=True)

    def test4_layer(self):
        self._test_norm([3, 2, 10, 320, 320], channel_last=False, layer_norm=True)


if __name__ == "__main__":
    unittest.test()
