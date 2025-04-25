import tensorflow.keras.backend as K
#K.set_floatx('float64')

import unittest
import numpy as np
import tensorflow as tf
import merlintf
from merlintf.keras.utils import validate_input_dimension
from merlintf.keras.layers.convolutional.complex_conv2dt import (
    calculate_intermediate_filters_2D, #TODO move this to testing?
    ComplexConv2Dt,
    ComplexConv2DtTranspose,
)
from merlintf.keras.layers.convolutional.complex_conv3dt import (
    calculate_intermediate_filters_3D, #TODO move this to testing?
    ComplexConv3Dt,
    ComplexConv3DtTranspose,
)
from merlintf.keras.layers.convolutional.complex_convolutional_realkernel import (
    ComplexConvRealWeight2D,
    ComplexConvRealWeight3D,
    ComplexDeconvRealWeight2D,
    ComplexDeconvRealWeight3D
)
from merlintf.keras.layers.convolutional.complex_convolutional import (
    ComplexConvolution1D,
    ComplexConvolution2D,
    ComplexConvolution3D,
    ComplexDeconvolution2D,
    ComplexDeconvolution3D,
    UpSampling1D,
    UpSampling2D,
    UpSampling3D,
    UpSampling4D,
    ZeroPadding1D,
    ZeroPadding2D,
    ZeroPadding3D,
    ZeroPadding4D,
    Cropping1D,
    Cropping2D,
    Cropping3D,
    Cropping4D
)
from merlintf.keras.layers.convolutional.conv2dt import (
    Conv2Dt,
    Conv2DtTranspose
)
from merlintf.keras.layers.convolutional.conv3dt import (
    Conv3Dt,
    Conv3DtTranspose
)

# complex_conv2dt.py
class ComplexConv2dtTest(unittest.TestCase):
    def test_ComplexConv2dt(self): #TODO split these tests!
        self._test_Conv2dt()
        self._test_Conv2dt(stride=(2, 2, 2))
        self._test_Conv2dt(channel_last=False)
        self._test_Conv2dt(use_3D_convs=False)
        self._test_Conv2dt(stride=(2, 2, 2), use_3D_convs=False)
        self._test_Conv2dt(channel_last=False, use_3D_convs=False)

    def test_ComplexConv2dtTranspose(self):
        self._test_Conv2dt(is_transpose=True)
        self._test_Conv2dt(is_transpose=True, stride=(2, 2, 2))
        self._test_Conv2dt(is_transpose=True, channel_last=False)
        self._test_Conv2dt(is_transpose=True, use_3D_convs=False)
        self._test_Conv2dt(is_transpose=True, stride=(2, 2, 2), use_3D_convs=False)
        self._test_Conv2dt(is_transpose=True, channel_last=False, use_3D_convs=False)

    def _test_Conv2dt(self, dim_in=[8, 32, 28], nBatch=2, nf_in=3, nf_out=18, ksz=(3, 5, 5), stride=(1, 1, 1),
                      channel_last=True, axis_conv_t=2, is_transpose=False, use_3D_convs=True):
        if is_transpose:
            dim_out = list((np.asarray(dim_in) * np.asarray(stride)).astype(int))
        else:
            dim_out = list((np.asarray(dim_in) / np.asarray(stride)).astype(int))

        if channel_last:
            shape = [nBatch] + dim_in + [nf_in]
            expected_shape = [nBatch] + dim_out + [nf_out]
            data_format = 'channels_last'
        else:
            shape = [nBatch] + [nf_in] + dim_in
            expected_shape = [nBatch] + [nf_out] + dim_out
            data_format = 'channels_first'

        ksz = validate_input_dimension('2Dt', ksz)
        nf_inter = calculate_intermediate_filters_2D(nf_out, ksz, nf_in)

        if is_transpose:
            model = ComplexConv2DtTranspose(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter,
                                     strides=stride, data_format=data_format, use_3D_convs=use_3D_convs)
        else:
            model = ComplexConv2Dt(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter,
                            strides=stride, data_format=data_format, use_3D_convs=use_3D_convs)

        x_real = tf.random.normal(shape, dtype=K.floatx())
        x_imag = tf.random.normal(shape, dtype=K.floatx())
        x = tf.complex(x_real, x_imag)
        Kx = model(x)

        self.assertTrue(Kx.shape == expected_shape)

# complex_conv3dt.py
class ComplexConv3dtTest(unittest.TestCase):
    def test_ComplexConv3dt(self):
        self._test_Conv3dt()
        self._test_Conv3dt(stride=(2, 2, 2, 2))
        self._test_Conv3dt(channel_last=False)

    def test_ComplexConv3dtTranspose(self):
        self._test_Conv3dt(is_transpose=True)
        self._test_Conv3dt(is_transpose=True, stride=(2, 2, 2, 2))
        self._test_Conv3dt(is_transpose=True, channel_last=False)

    def _test_Conv3dt(self, dim_in=[8, 32, 32, 12], nBatch=2, nf_in=3, nf_out=18, ksz=(3, 5, 5, 5), stride=(1, 1, 1, 1),
                      channel_last=True, axis_conv_t=2, is_transpose=False):
        if is_transpose:
            dim_out = list((np.asarray(dim_in) * np.asarray(stride)).astype(int))
        else:
            dim_out = list((np.asarray(dim_in) / np.asarray(stride)).astype(int))

        if channel_last:
            shape = [nBatch] + dim_in + [nf_in]
            expected_shape = [nBatch] + dim_out + [nf_out]
            data_format = 'channels_last'
        else:
            shape = [nBatch] + [nf_in] + dim_in
            expected_shape = [nBatch] + [nf_out] + dim_out
            data_format = 'channels_first'

        ksz = validate_input_dimension('3Dt', ksz)
        nf_inter = calculate_intermediate_filters_3D(nf_out, ksz, nf_in)

        if is_transpose:
            model = ComplexConv3DtTranspose(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter,
                                     strides=stride, data_format=data_format)
        else:
            model = ComplexConv3Dt(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter,
                            strides=stride, data_format=data_format)

        x_real = tf.random.normal(shape, dtype=K.floatx())
        x_imag = tf.random.normal(shape, dtype=K.floatx())
        x = tf.complex(x_real, x_imag)
        Kx = model(x)

        self.assertTrue(Kx.shape == expected_shape)

    def _testadjoint(self, dim_in=[8, 32, 32, 12], nBatch=2, nf_in=3, nf_out=18, ksz=(3, 5, 5, 5), stride=(1, 1, 1, 1),
                      channel_last=True, axis_conv_t=2, is_transpose=False):
        # gradient check not required only for matching FOVs
        if channel_last:
            shape = [nBatch] + dim_in + [nf_in]
            expected_shape = [nBatch] + list((np.asarray(dim_in) / np.asarray(stride)).astype(int)) + [nf_out]
            data_format = 'channels_last'
        else:
            shape = [nBatch] + [nf_in] + dim_in
            expected_shape = [nBatch] + [nf_out] + list((np.asarray(dim_in) / np.asarray(stride)).astype(int))
            data_format = 'channels_first'

        ksz = validate_input_dimension('3Dt', ksz)
        nf_inter = calculate_intermediate_filters_3D(nf_out, ksz, nf_in)

        if is_transpose:
            model = ComplexConv3DtTranspose(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter,
                                     strides=stride, data_format=data_format)
        else:
            model = ComplexConv3Dt(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter,
                            strides=stride, data_format=data_format)

        x = tf.complex(tf.random.normal(shape, dtype=K.floatx()), tf.random.normal(shape, dtype=K.floatx()))
        Kx = model(x)

        y = tf.complex(tf.random.normal(Kx.shape, dtype=K.floatx()), tf.random.normal(Kx.shape, dtype=K.floatx()))
        with tf.GradientTape() as g:
            g.watch(y)
            Ky = model(y)
            loss = 0.5 * tf.reduce_sum(tf.math.conj(Ky) * Ky)
        grad_y = g.gradient(loss, y)
        KHy = grad_y.numpy()
        #KHy = model.backward(y, x.shape)

        rhs = tf.reduce_sum(Kx * y).numpy()
        lhs = tf.reduce_sum(x * KHy).numpy()

        self.assertTrue(rhs, lhs)

# complex_convolutional_realkernel.py
class ComplexConv2DTest(unittest.TestCase):
    def _test_fwd(self, conv_fun, kernel_size, strides, dilations, activation, padding):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, M, N, nf_in]

        model = conv_fun(nf_out, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)
        x = merlintf.random_normal_complex(shape, dtype=K.floatx())
        Kx = model(x)

    def test1(self):
        self._test_fwd(ComplexConvRealWeight2D, 3, 2, 1, None, 'same')
    def test2(self):
        self._test_fwd(ComplexConvRealWeight2D, 3, 1, 2, None, 'same')
    def test1T(self):
        self._test_fwd(ComplexDeconvRealWeight2D, 3, 2, 1, None, 'same')
    def test2T(self):
        self._test_fwd(ComplexDeconvRealWeight2D, 3, 1, 2, None, 'same')

class ComplexConv3DTest(unittest.TestCase):
    def _test_fwd(self, conv_fun, kernel_size, strides, dilations, activation, padding):
        nBatch = 5
        M = 64
        N = 64
        D = 8
        nf_in = 10
        nf_out = 32
        shape = [nBatch, D, M, N, nf_in]

        model = conv_fun(nf_out, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)
        x = merlintf.random_normal_complex(shape, dtype=K.floatx())
        Kx = model(x)

    def test1(self):
        self._test_fwd(ComplexConvRealWeight3D, 3, 2, 1, None, 'same')
    def test2(self):
        self._test_fwd(ComplexConvRealWeight3D, 3, 1, 2, None, 'same')
    def test1T(self):
        self._test_fwd(ComplexDeconvRealWeight3D, 3, 2, 1, None, 'same')
    def test2T(self):
        self._test_fwd(ComplexDeconvRealWeight3D, 3, 1, 2, None, 'same')

# complex_convolutional.py
class ComplexConv2DTest(unittest.TestCase):
    def _test_fwd(self, conv_fun, kernel_size, strides, dilations, activation, padding):
        nBatch = 5
        M = 256
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, M, N, nf_in]

        model = conv_fun(nf_out, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)
        x = merlintf.random_normal_complex(shape, dtype=K.floatx())
        Kx = model(x)

    def test1(self):
        self._test_fwd(ComplexConvolution2D, 3, 2, 1, None, 'same')

    def test2(self):
        self._test_fwd(ComplexConvolution2D, 3, 1, 2, None, 'same')

    def test1T(self):
        self._test_fwd(ComplexDeconvolution2D, 3, 2, 1, None, 'same')

    def test2T(self):
        self._test_fwd(ComplexDeconvolution2D, 3, 1, 2, None, 'same')

    def _test_other(self, op, size=2):
        nBatch = 5
        M = 64
        N = 64
        nf_in = 10
        shape = [nBatch, M, N, nf_in]

        model = op(size)
        x = merlintf.random_normal_complex(shape, dtype=K.floatx())
        Kx = model(x)

    def test_pad(self):
        self._test_other(ZeroPadding2D, 2)

    def test_crop(self):
        self._test_other(Cropping2D, 2)

    def test_upsample(self):
        self._test_other(UpSampling2D, 2)

class ComplexConv3DTest(unittest.TestCase):
    def _test_fwd(self, conv_fun, kernel_size, strides, dilations, activation, padding):
        nBatch = 5
        M = 64
        N = 64
        D = 8
        nf_in = 10
        nf_out = 32
        shape = [nBatch, D, M, N, nf_in]

        model = conv_fun(nf_out, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)
        x = merlintf.random_normal_complex(shape, dtype=K.floatx())
        Kx = model(x)

    def test1(self):
        self._test_fwd(ComplexConvolution3D, 3, 2, 1, None, 'same')

    def test2(self):
        self._test_fwd(ComplexConvolution3D, 3, 1, 2, None, 'same')

    def test1T(self):
        self._test_fwd(ComplexDeconvolution3D, 3, 2, 1, None, 'same')

    def test2T(self):
        self._test_fwd(ComplexDeconvolution3D, 3, 1, 2, None, 'same')
      
    def _test_other(self, op, size=2):
        nBatch = 5
        M = 64
        N = 64
        D = 8
        nf_in = 10
        shape = [nBatch, D, M, N, nf_in]

        model = op(size)
        x = merlintf.random_normal_complex(shape, dtype=K.floatx())
        Kx = model(x)

    def test_pad(self):
        self._test_other(ZeroPadding3D, 2)

    def test_crop(self):
        self._test_other(Cropping3D, 2)

    def test_upsample(self):
        self._test_other(UpSampling3D, 2)

class ComplexConv1DTest(unittest.TestCase):
    def _test_fwd(self, conv_fun, kernel_size, strides, dilations, activation, padding):
        nBatch = 5
        N = 256
        nf_in = 10
        nf_out = 32
        shape = [nBatch, N, nf_in]

        model = conv_fun(nf_out, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding)
        x = merlintf.random_normal_complex(shape, dtype=K.floatx())
        Kx = model(x)

    def test1(self):
        self._test_fwd(ComplexConvolution1D, 3, 2, 1, None, 'same')

    def test2(self):
        self._test_fwd(ComplexConvolution1D, 3, 1, 2, None, 'same')

    def _test_other(self, op, size=2):
        nBatch = 5
        N = 256
        nf_in = 10
        shape = [nBatch, N, nf_in]

        model = op(size)
        x = merlintf.random_normal_complex(shape, dtype=K.floatx())
        Kx = model(x)

    def test_pad(self):
        self._test_other(ZeroPadding1D, 2)

    def test_crop(self):
        self._test_other(Cropping1D, 2)

    def test_upsample(self):
        self._test_other(UpSampling1D, 2)

# conv2dt.py
class Conv2dtTest(unittest.TestCase):
    def test_Conv2dt(self):
        self._test_Conv2dt()
        self._test_Conv2dt(stride=(2, 2, 2))
        self._test_Conv2dt(channel_last=False)
        self._test_Conv2dt(use_3D_convs=False)
        self._test_Conv2dt(stride=(2, 2, 2), use_3D_convs=False)
        self._test_Conv2dt(channel_last=False, use_3D_convs=False)

    def test_Conv2dtTranspose(self):
        self._test_Conv2dt(is_transpose=True)
        self._test_Conv2dt(is_transpose=True, stride=(2, 2, 2))
        self._test_Conv2dt(is_transpose=True, channel_last=False)
        self._test_Conv2dt(is_transpose=True, use_3D_convs=False)
        self._test_Conv2dt(is_transpose=True, stride=(2, 2, 2), use_3D_convs=False)
        self._test_Conv2dt(is_transpose=True, channel_last=False, use_3D_convs=False)

    def _test_Conv2dt(self, dim_in=[8, 32, 28], nBatch=2, nf_in=3, nf_out=18, ksz=(3, 5, 5), stride=(1, 1, 1),
                      channel_last=True, axis_conv_t=2, is_transpose=False, use_3D_convs=True):
        if is_transpose:
            dim_out = list((np.asarray(dim_in) * np.asarray(stride)).astype(int))
        else:
            dim_out = list((np.asarray(dim_in) / np.asarray(stride)).astype(int))

        if channel_last:
            shape = [nBatch] + dim_in + [nf_in]
            expected_shape = [nBatch] + dim_out + [nf_out]
            data_format = 'channels_last'
        else:
            shape = [nBatch] + [nf_in] + dim_in
            expected_shape = [nBatch] + [nf_out] + dim_out
            data_format = 'channels_first'

        ksz = validate_input_dimension('2Dt', ksz)
        nf_inter = calculate_intermediate_filters_2D(nf_out, ksz, nf_in)

        if is_transpose:
            model = Conv2DtTranspose(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter,
                                     strides=stride, data_format=data_format, use_3D_convs=use_3D_convs)
        else:
            model = Conv2Dt(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter,
                            strides=stride, data_format=data_format, use_3D_convs=use_3D_convs)

        x = tf.random.normal(shape, dtype=K.floatx())
        Kx = model(x)

        self.assertTrue(Kx.shape == expected_shape)

# conv3dt.py
class Conv3dtTest(unittest.TestCase):
    def test_Conv3dt(self):
        self._test_Conv3dt()
        self._test_Conv3dt(stride=(2, 2, 2, 2))
        self._test_Conv3dt(channel_last=False)

    def test_Conv3dtTranspose(self):
        self._test_Conv3dt(is_transpose=True)
        self._test_Conv3dt(is_transpose=True, stride=(2, 2, 2, 2))
        self._test_Conv3dt(is_transpose=True, channel_last=False)

    def _test_Conv3dt(self, dim_in=[8, 32, 32, 12], nBatch=2, nf_in=3, nf_out=18, ksz=(3, 5, 5, 5), stride=(1, 1, 1, 1),
                      channel_last=True, axis_conv_t=2, is_transpose=False):
        if is_transpose:
            dim_out = list((np.asarray(dim_in) * np.asarray(stride)).astype(int))
        else:
            dim_out = list((np.asarray(dim_in) / np.asarray(stride)).astype(int))

        if channel_last:
            shape = [nBatch] + dim_in + [nf_in]
            expected_shape = [nBatch] + dim_out + [nf_out]
            data_format = 'channels_last'
        else:
            shape = [nBatch] + [nf_in] + dim_in
            expected_shape = [nBatch] + [nf_out] + dim_out
            data_format = 'channels_first'

        ksz = validate_input_dimension('3Dt', ksz)
        nf_inter = calculate_intermediate_filters_3D(nf_out, ksz, nf_in)

        if is_transpose:
            model = Conv3DtTranspose(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter,
                                     strides=stride, data_format=data_format)
        else:
            model = Conv3Dt(nf_out, kernel_size=ksz, shapes=shape, axis_conv_t=2, intermediate_filters=nf_inter,
                            strides=stride, data_format=data_format)

        x = tf.random.normal(shape, dtype=K.floatx())
        Kx = model(x)

        self.assertTrue(Kx.shape == expected_shape)

class UpSampling4dTest(unittest.TestCase):
    def test_UpSampling4d(self):
        self._test_UpSampling4d()
        self._test_UpSampling4d(channel_last=False)

    def _test_UpSampling4d(self, dim_in=[8, 32, 32, 12], nBatch=2, nf_in=3, size=(2, 2, 2, 2), channel_last=True):
        if channel_last:
            shape = [nBatch] + dim_in + [nf_in]
            expected_shape = [nBatch] + [d * size[i] for i, d in enumerate(dim_in)] + [nf_in]
            data_format = 'channels_last'
        else:
            shape = [nBatch] + [nf_in] + dim_in
            expected_shape = [nBatch] + [nf_in] + [d * size[i] for i, d in enumerate(dim_in)]
            data_format = 'channels_first'

        model = UpSampling4D(size=size, data_format=data_format)

        x = tf.random.normal(shape, dtype=K.floatx())
        Kx = model(x)

        self.assertTrue(Kx.shape == expected_shape)

class ZeroPadding4dTest(unittest.TestCase):
    def test_ZeroPadding4d(self):
        self._test_ZeroPadding4d()
        self._test_ZeroPadding4d(padding=(3, 3, 3, 3))
        self._test_ZeroPadding4d(padding=((1, 3), (1, 3), (1, 3), (1, 3)))
        self._test_ZeroPadding4d(channel_last=False)

    def _test_ZeroPadding4d(self, dim_in=[8, 32, 32, 12], nBatch=2, nf_in=3, padding=(2, 2, 2, 2), channel_last=True):
        if isinstance(padding[0], int):
            dim_out = [d + 2*padding[i] for i, d in enumerate(dim_in)]
        else:
            dim_out = [d + padding[i][0] + padding[i][1] for i, d in enumerate(dim_in)]

        if channel_last:
            shape = [nBatch] + dim_in + [nf_in]
            expected_shape = [nBatch] + dim_out + [nf_in]
            data_format = 'channels_last'
        else:
            shape = [nBatch] + [nf_in] + dim_in
            expected_shape = [nBatch] + [nf_in] + dim_out
            data_format = 'channels_first'

        model = ZeroPadding4D(padding=padding, data_format=data_format)

        x = tf.random.normal(shape, dtype=K.floatx())
        Kx = model(x)

        self.assertTrue(Kx.shape == expected_shape)

class Cropping4dTest(unittest.TestCase):
    def test_Cropping4d(self):
        self._test_Cropping4d()
        self._test_Cropping4d(cropping=(3, 3, 3, 3))
        self._test_Cropping4d(cropping=((1, 3), (1, 3), (1, 3), (1, 3)))
        self._test_Cropping4d(channel_last=False)

    def _test_Cropping4d(self, dim_in=[8, 32, 32, 12], nBatch=2, nf_in=3, cropping=(2, 2, 2, 2), channel_last=True):
        if isinstance(cropping[0], int):
            dim_out = [d - 2*cropping[i] for i, d in enumerate(dim_in)]
        else:
            dim_out = [d - cropping[i][0] - cropping[i][1] for i, d in enumerate(dim_in)]

        if channel_last:
            shape = [nBatch] + dim_in + [nf_in]
            expected_shape = [nBatch] + dim_out + [nf_in]
            data_format = 'channels_last'
        else:
            shape = [nBatch] + [nf_in] + dim_in
            expected_shape = [nBatch] + [nf_in] + dim_out
            data_format = 'channels_first'

        model = Cropping4D(cropping=cropping, data_format=data_format)

        x = tf.random.normal(shape, dtype=K.floatx())
        Kx = model(x)

        self.assertTrue(Kx.shape == expected_shape)

if __name__ == "__main__":
    unittest.main()
