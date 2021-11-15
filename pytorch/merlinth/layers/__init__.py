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
except:
    print('padconv layers could not be loaded. Optox might not be installed.')
# from merlinth.layers.convolutional.complex_conv3d import ComplexPadConv2Dt
# from merlinth.layers.pad import real_pad2d
# from merlinth.layers.pad import real_pad2d_transpose
# from merlinth.layers.pad import real_pad3d
# from merlinth.layers.pad import real_pad3d_transpose
# from merlinth.layers.pad import complex_pad2d
# from merlinth.layers.pad import complex_pad2d_transpose
# from merlinth.layers.pad import complex_pad3d
# from merlinth.layers.pad import complex_pad3d_transpose


# from .complex_regularizer import *
# from .complex_conv2d import *
# from .complex_foe2d import *
# from .complex_foe3d import *
# from .complex_layer import *
# #from .complex_tdv import *
# from .unet import *
# from .complex_act import *
# from .foe2d import *