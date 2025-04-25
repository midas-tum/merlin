import tensorflow as tf
import tensorflow.keras.backend as K

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
