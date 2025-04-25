from merlinth.layers.convolutional.complex_conv import ComplexConv1d
from merlinth.layers.convolutional.complex_conv import ComplexConv2d
from merlinth.layers.convolutional.complex_conv import ComplexConv3d

try:
    from merlinth.layers.convolutional.padconv import PadConv1d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConv1d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConvRealWeight1d
    from merlinth.layers.convolutional.padconv import PadConv2d
    from merlinth.layers.convolutional.padconv import PadConvScale2d
    from merlinth.layers.convolutional.padconv import PadConvScaleTranspose2d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConv2d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConvScale2d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConvScaleTranspose2d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConvRealWeight2d
    from merlinth.layers.convolutional.padconv import PadConv3d
    from merlinth.layers.convolutional.padconv import PadConvScale3d
    from merlinth.layers.convolutional.padconv import PadConvScaleTranspose3d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConv3d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConvScale3d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConvScaleTranspose3d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConvRealWeight3d
    from merlinth.layers.convolutional.complex_padconv import ComplexPadConv2Dt
except:
    print('padconv layers could not be loaded. Optox might not be installed.')

try:
    from merlinth.layers.complex_maxpool import MagnitudeMaxPooling, MagnitudeMaxPool1D, \
        MagnitudeMaxPool2D, MagnitudeMaxPool2Dt, MagnitudeMaxPool3D, MagnitudeMaxPool3Dt
    from merlinth.layers.complex_avgpool import MagnitudeAveragePooling, MagnitudeAveragePool1D, \
        MagnitudeAveragePool2D, MagnitudeAveragePool2Dt, MagnitudeAveragePool3D, MagnitudeAveragePool3Dt
except:
    print('pooling layers in merlinth.layers could not be loaded. Optox might not be installed.')

from merlinth.layers.data_consistency import DCGD
from merlinth.layers.data_consistency import DCPM
from merlinth.layers.data_consistency import itSENSE
from merlinth.layers.mri import MulticoilForwardOp
from merlinth.layers.mri import MulticoilAdjointOp

try:
    from merlinth.layers.mri import MulticoilMotionForwardOp
    from merlinth.layers.mri import MulticoilMotionAdjointOp
    from merlinth.layers.warp import WarpForward
    from merlinth.layers.warp import WarpAdjoint
except:
    print('warp and MulticoilMotion operator layers could not be loaded. Optox might not be installed.')
from merlinth.layers.fft import fft2
from merlinth.layers.fft import fft2c
from merlinth.layers.fft import ifft2
from merlinth.layers.fft import ifft2c

# from .complex_regularizer import *
# from .complex_conv2d import *
# from .complex_foe2d import *
# from .complex_foe3d import *
# from .complex_layer import *
# from .unet import *
# from .complex_act import *
# from .foe2d import *
