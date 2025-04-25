import tensorflow as tf
from tensorflow.signal import fft2d, ifft2d, ifftshift, fftshift
import merlintf

class IFFT2c(tf.keras.layers.Layer):
    def call(self, kspace, *args):
        axes = [tf.rank(kspace)-2, tf.rank(kspace)-1] # axes have to be positive...
        dtype = tf.math.real(kspace).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), dtype))
        return merlintf.complex_scale(fftshift(ifft2d(ifftshift(kspace, axes=axes)), axes=axes), scale)

class FFT2c(tf.keras.layers.Layer):
    def call(self, image, *args):
        dtype = tf.math.real(image).dtype
        axes = [tf.rank(image)-2, tf.rank(image)-1] # axes have to be positive...
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), dtype))
        return merlintf.complex_scale(fftshift(fft2d(ifftshift(image, axes=axes)), axes=axes), 1/scale)

class IFFT2(tf.keras.layers.Layer):
    def call(self, kspace, *args):
        dtype = tf.math.real(kspace).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), dtype))
        return merlintf.complex_scale(ifft2d(kspace), scale)

class FFT2(tf.keras.layers.Layer):
    def call(self, image, *args):
        dtype = tf.math.real(image).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), dtype))
        return merlintf.complex_scale(fft2d(image), 1/scale)