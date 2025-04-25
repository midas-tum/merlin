from merlintf.keras.layers.convolutional.complex_convolutional import ComplexConv1D
from merlintf.keras.layers.convolutional.complex_convolutional import ComplexConv1DTranspose
from merlintf.keras.layers.convolutional.complex_convolutional import ComplexConv2D
from merlintf.keras.layers.convolutional.complex_convolutional import ComplexConv2DTranspose
from merlintf.keras.layers.convolutional.complex_convolutional import ComplexConv3D
from merlintf.keras.layers.convolutional.complex_convolutional import ComplexConv3DTranspose
from merlintf.keras.layers.convolutional.complex_convolutional import ComplexConvolution
from merlintf.keras.layers.convolutional.complex_convolutional import ComplexConvolutionTranspose
from merlintf.keras.layers.convolutional.complex_convolutional import UpSampling, UpSampling1D, UpSampling2D, UpSampling3D, UpSampling4D
from merlintf.keras.layers.convolutional.complex_convolutional import ZeroPadding, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D, ZeroPadding4D
from merlintf.keras.layers.convolutional.complex_convolutional import Cropping, Cropping1D, Cropping2D, Cropping3D, Cropping4D
try:
    from merlintf.keras.layers.convolutional.complex_padconv import ComplexPadConv2D
    from merlintf.keras.layers.convolutional.complex_padconv import ComplexPadConv3D
    from merlintf.keras.layers.convolutional.complex_padconv import ComplexPadConvScale2D
    from merlintf.keras.layers.convolutional.complex_padconv import ComplexPadConvScale3D
    from merlintf.keras.layers.convolutional.complex_padconv import ComplexPadConvScale2DTranspose
    from merlintf.keras.layers.convolutional.complex_padconv import ComplexPadConvScale3DTranspose
    from merlintf.keras.layers.convolutional.complex_padconv_2dt import ComplexPadConv2Dt
    from merlintf.keras.layers.convolutional.padconv import PadConv1D
    from merlintf.keras.layers.convolutional.padconv import PadConv2D
    from merlintf.keras.layers.convolutional.padconv import PadConv3D
    from merlintf.keras.layers.convolutional.padconv import PadConvScale2D
    from merlintf.keras.layers.convolutional.padconv import PadConvScale3D
    from merlintf.keras.layers.convolutional.padconv import PadConvScale2DTranspose
    from merlintf.keras.layers.convolutional.padconv import PadConvScale3DTranspose
except:
    print('padconv layers in keras.layers.convolutional could not be loaded. Optox might not be installed.')

try:
    from merlintf.keras.layers.complex_maxpool import MagnitudeMaxPooling, MagnitudeMaxPool1D, \
        MagnitudeMaxPool2D, MagnitudeMaxPool2Dt, MagnitudeMaxPool3D, MagnitudeMaxPool3Dt
    from merlintf.keras.layers.complex_avgpool import MagnitudeAveragePooling, MagnitudeAveragePool1D, \
        MagnitudeAveragePool2D, MagnitudeAveragePool2Dt, MagnitudeAveragePool3D, MagnitudeAveragePool3Dt
except:
    print('pooling layers in keras.layers could not be loaded. Optox might not be installed.')

try:
    from merlintf.keras.layers.complex_pad import Padding, Padding1D, Padding1DTranspose, Padding2D, Padding2DTranspose, \
        Padding3D, Padding3DTranspose
except:
    print('padding layers in keras.layers could not be loaded. Optox might not be installed.')

from merlintf.keras.layers.convolutional.complex_conv2dt import ComplexConv2Dt, ComplexConv2DtTranspose
from merlintf.keras.layers.convolutional.complex_conv3dt import ComplexConv3Dt, ComplexConv3DtTranspose
from merlintf.keras.layers.convolutional.conv2dt import Conv2Dt, Conv2DtTranspose
from merlintf.keras.layers.convolutional.conv3dt import Conv3Dt, Conv3DtTranspose
from merlintf.keras.layers.complex_act import Activation
from merlintf.keras.layers.complex_bn import ComplexBatchNormalization
from merlintf.keras.layers.complex_norm import ComplexInstanceNormalization
try:
    from merlintf.keras.layers.complex_pad import Padding1D, Padding2D, Padding3D, Padding4D, Padding1DTranspose, Padding2DTranspose, Padding3DTranspose, Padding4DTranspose
except:
    print('keras.layers.complex_pad could not be loaded. Optox might not be installed.')
from merlintf.keras.layers.data_consistency import DCGD
from merlintf.keras.layers.data_consistency import DCPM
from merlintf.keras.layers.data_consistency import itSENSE
from merlintf.keras.layers.common import Scalar
from merlintf.keras.layers.mri import MulticoilForwardOp
from merlintf.keras.layers.mri import MulticoilAdjointOp
from merlintf.keras.layers.mri import ForwardOp
from merlintf.keras.layers.mri import AdjointOp
from merlintf.keras.layers.fft import FFT2
from merlintf.keras.layers.fft import FFT2c
from merlintf.keras.layers.fft import IFFT2
from merlintf.keras.layers.fft import IFFT2c