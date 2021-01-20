import tensorflow as tf
from merlintf.keras.layers.fft import FFT2, FFT2c, IFFT2, IFFT2c
import merlintf

class Smaps(tf.keras.layers.Layer):
    def call(self, img, smaps):
        return tf.expand_dims(img, -3) * smaps

class SmapsAdj(tf.keras.layers.Layer):
    def call(self, coilimg, smaps):
        return tf.reduce_sum(coilimg * tf.math.conj(smaps), -3)

class MaskKspace(tf.keras.layers.Layer):
    def call(self, kspace, mask):
        return merlintf.complex_scale(kspace, mask)

class ForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = FFT2c()
        else:
            self.fft2 = FFT2()
        self.mask = MaskKspace()

    def call(self, image, mask):
        kspace = self.fft2(image[...,0])
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace

class AdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = IFFT2c()
        else:
            self.ifft2 = IFFT2()

    def call(self, kspace, mask):
        masked_kspace = self.mask(kspace, mask)
        img = self.ifft2(masked_kspace)
        return tf.expand_dims(img, -1)

class MulticoilForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = FFT2c()
        else:
            self.fft2 = FFT2()
        self.mask = MaskKspace()
        self.smaps = Smaps()

    def call(self, image, mask, smaps):
        coilimg = self.smaps(image[...,0], smaps)
        kspace = self.fft2(coilimg)
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace

class MulticoilAdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = IFFT2c()
        else:
            self.ifft2 = IFFT2()
        self.adj_smaps = SmapsAdj()

    def call(self, kspace, mask, smaps):
        masked_kspace = self.mask(kspace, mask)
        coilimg = self.ifft2(masked_kspace)
        img = self.adj_smaps(coilimg, smaps)
        return tf.expand_dims(img, -1)