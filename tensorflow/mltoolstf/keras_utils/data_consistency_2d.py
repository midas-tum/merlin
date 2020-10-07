import tensorflow as tf
import mltoolstf.ddr_complex as layers
from tensorflow.signal import fft2d, ifft2d, ifftshift, fftshift
from .complex import complex_scale
import tensorflow.keras.backend as K

class Smaps(tf.keras.layers.Layer):
    def call(self, img, smaps):
        return tf.expand_dims(img, 1) * smaps

class SmapsAdj(tf.keras.layers.Layer):
    def call(self, coilimg, smaps):
        return tf.reduce_sum(coilimg * tf.math.conj(smaps), 1)

class MaskKspace(tf.keras.layers.Layer):
    def call(self, kspace, mask):
        return complex_scale(kspace, mask)

class Ifft2c(tf.keras.layers.Layer):
    def call(self, kspace, *args):
        axes = [1,2] # TODO derive from ndims!
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), 'float32'))
        return complex_scale(fftshift(ifft2d(ifftshift(kspace, axes=axes)), axes=axes), scale)

class Fft2c(tf.keras.layers.Layer):
    def call(self, image, *args):
        axes = [1,2] # TODO derive from ndims!
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), 'float32'))
        return  complex_scale(fftshift(fft2d(ifftshift(image, axes=axes)), axes=axes), 1/scale)

class Ifft2(tf.keras.layers.Layer):
    def call(self, kspace, *args):
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), 'float32'))
        return complex_scale(ifft2d(kspace), scale)

class Fft2(tf.keras.layers.Layer):
    def call(self, image, *args):
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), 'float32'))
        return  complex_scale(fft2d(image), 1/scale)

class ForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = Fft2c()
        else:
            self.fft2 = Fft2()
        self.mask = MaskKspace()

    def call(self, image, mask):
        kspace = self.fft2(image[...,0])
        masked_kspace = self.mask(kspace, mask)
        return tf.expand_dims(masked_kspace, -1)

class AdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = Ifft2c()
        else:
            self.ifft2 = Ifft2()

    def call(self, kspace, mask):
        masked_kspace = self.mask(kspace[...,0], mask)
        img = self.ifft2(masked_kspace)
        return tf.expand_dims(img, -1)

class MulticoilForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = Fft2c()
        else:
            self.fft2 = Fft2()
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
            self.ifft2 = Ifft2c()
        else:
            self.ifft2 = Ifft2()
        self.adj_smaps = SmapsAdj()

    def call(self, kspace, mask, smaps):
        masked_kspace = self.mask(kspace, mask)
        coilimg = self.ifft2(masked_kspace)
        img = self.adj_smaps(coilimg, smaps)
        return tf.expand_dims(img, -1)


class DCGD2D(tf.keras.layers.Layer):
    def __init__(self, config, center=False, multicoil=True, name='dc-gd'):
        super().__init__()
        if multicoil:
            self.A = MulticoilForwardOp(center)
            self.AH = MulticoilAdjointOp(center)
        else:
            self.A = ForwardOp(center)
            self.AH = AdjointOp(center)

        self.train_scale = config['lambda']['train_scale'] if 'train_scale' in config['lambda'] else 1
        self._weight = self.add_weight(name='lambda',
                                     shape=(1,),
                                     constraint=tf.keras.constraints.NonNeg(),
                                     initializer=tf.keras.initializers.Constant(config['lambda']['init']))
    @property
    def weight(self):
        return self._weight * self.train_scale

    def call(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        return complex_scale(self.AH(self.A(x, *constants) - y, *constants), self.weight * scale)