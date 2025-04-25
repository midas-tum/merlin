import tensorflow as tf
from merlintf.keras.layers.fft import FFT2, FFT2c, IFFT2, IFFT2c
from merlintf.keras.layers.warp import WarpForward, WarpAdjoint
import merlintf

class Smaps(tf.keras.layers.Layer):
    def __init__(self, coil_axis=-3):
        super().__init__()
        self.coil_axis = coil_axis
        
    def call(self, img, smaps):
        return tf.expand_dims(img, self.coil_axis) * smaps

class SmapsAdj(tf.keras.layers.Layer):
    def __init__(self, coil_axis=-3):
        super().__init__()
        self.coil_axis = coil_axis

    def call(self, coilimg, smaps):
        return tf.reduce_sum(coilimg * tf.math.conj(smaps), self.coil_axis)

class MaskKspace(tf.keras.layers.Layer):
    def call(self, kspace, mask):
        return merlintf.complex_scale(kspace, mask)

class ForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False, channel_dim_defined=True):
        super().__init__()
        if center:
            self.fft2 = FFT2c()
        else:
            self.fft2 = FFT2()
        self.mask = MaskKspace()
        self.channel_dim_defined = channel_dim_defined

    def call(self, image, mask):
        if self.channel_dim_defined:
            kspace = self.fft2(image[...,0])
        else:
            kspace = self.fft2(image)
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace

class AdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False, channel_dim_defined=True):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = IFFT2c()
        else:
            self.ifft2 = IFFT2()
        self.channel_dim_defined = channel_dim_defined

    def call(self, kspace, mask):
        masked_kspace = self.mask(kspace, mask)
        img = self.ifft2(masked_kspace)
        if self.channel_dim_defined:
            return tf.expand_dims(img, -1)
        else:
            return img

class MulticoilForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        if center:
            self.fft2 = FFT2c()
        else:
            self.fft2 = FFT2()
        self.mask = MaskKspace()
        self.smaps = Smaps(coil_axis=coil_axis)
        self.channel_dim_defined = channel_dim_defined

    def call(self, image, mask, smaps):
        if self.channel_dim_defined:
            coilimg = self.smaps(image[...,0], smaps)
        else:
            coilimg = self.smaps(image, smaps)
        kspace = self.fft2(coilimg)
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace

class MulticoilAdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = IFFT2c()
        else:
            self.ifft2 = IFFT2()
        self.adj_smaps = SmapsAdj(coil_axis=coil_axis)
        self.channel_dim_defined = channel_dim_defined

    def call(self, kspace, mask, smaps):
        masked_kspace = self.mask(kspace, mask)
        coilimg = self.ifft2(masked_kspace)
        img = self.adj_smaps(coilimg, smaps)
        if self.channel_dim_defined:
            return tf.expand_dims(img, -1)
        else:
            return img

class MulticoilMotionForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        self.W = WarpForward()
        self.A = MulticoilForwardOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.channel_dim_defined = channel_dim_defined

    def call(self, x, mask, smaps, u):
        if self.channel_dim_defined:
            x = self.W(x[...,0], u)
        else:
            x = self.W(x, u)
        y = self.A(x, mask, smaps)
        return y

class MulticoilMotionAdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        self.AH = MulticoilAdjointOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.WH = WarpAdjoint()
        self.channel_dim_defined = channel_dim_defined

    def call(self, y, mask, smaps, u):
        x = self.AH(y, mask, smaps)
        x = self.WH(x, u)
        if self.channel_dim_defined:
            return tf.expand_dims(x, -1)
        else:
            return x
