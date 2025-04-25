merlintf
========


.. currentmodule:: merlintf

Common Tools
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    complex
    utils

Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.layers.data_consistency
    merlintf.keras.layers.fft
    merlintf.keras.layers.mri
    merlintf.keras.layers.warp

Convolutions
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.layers.ComplexConv1D
    merlintf.keras.layers.ComplexConv2D
    merlintf.keras.layers.ComplexConv3D
    merlintf.keras.layers.ComplexConv2Dt
    merlintf.keras.layers.ComplexConv3Dt
    merlintf.keras.layers.ComplexConv1DTranspose
    merlintf.keras.layers.ComplexConv2DTranspose
    merlintf.keras.layers.ComplexConv3DTranspose
    merlintf.keras.layers.ComplexConv2DtTranspose
    merlintf.keras.layers.ComplexConv3DtTranspose
    merlintf.keras.layers.PadConv1D
    merlintf.keras.layers.PadConv2D
    merlintf.keras.layers.PadConv3D
    merlintf.keras.layers.PadConvScale2D
    merlintf.keras.layers.PadConvScale3D
    merlintf.keras.layers.PadConvScale2DTranspose
    merlintf.keras.layers.PadConvScale3DTranspose
    merlintf.keras.layers.Conv2Dt
    merlintf.keras.layers.Conv3Dt
    merlintf.keras.layers.Conv2DtTranspose
    merlintf.keras.layers.Conv3DtTranspose


Upsampling
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.layers.UpSampling1D
    merlintf.keras.layers.UpSampling2D
    merlintf.keras.layers.UpSampling3D
    merlintf.keras.layers.UpSampling2Dt
    merlintf.keras.layers.UpSampling3Dt


Zeropadding
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.layers.ZeroPadding1D
    merlintf.keras.layers.ZeroPadding2D
    merlintf.keras.layers.ZeroPadding3D
    merlintf.keras.layers.ZeroPadding2Dt
    merlintf.keras.layers.ZeroPadding3Dt


Cropping
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.layers.Cropping1D
    merlintf.keras.layers.Cropping2D
    merlintf.keras.layers.Cropping3D
    merlintf.keras.layers.Cropping2Dt
    merlintf.keras.layers.Cropping3Dt


Pooling
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.layers.MagnitudeMaxPooling
    merlintf.keras.layers.MagnitudeMaxPool1D
    merlintf.keras.layers.MagnitudeMaxPool2D
    merlintf.keras.layers.MagnitudeMaxPool2Dt
    merlintf.keras.layers.MagnitudeMaxPool3D
    merlintf.keras.layers.MagnitudeMaxPool3Dt
    merlintf.keras.layers.MagnitudeAveragePooling
    merlintf.keras.layers.MagnitudeAveragePool1D
    merlintf.keras.layers.MagnitudeAveragePool2D
    merlintf.keras.layers.MagnitudeAveragePool2Dt
    merlintf.keras.layers.MagnitudeAveragePool3D
    merlintf.keras.layers.MagnitudeAveragePool3Dt

Padding
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.layers.Padding
    merlintf.keras.layers.Padding1D
    merlintf.keras.layers.Padding1DTranspose
    merlintf.keras.layers.Padding2D
    merlintf.keras.layers.Padding2DTranspose
    merlintf.keras.layers.Padding3D
    merlintf.keras.layers.Padding3DTranspose


Activation
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.layers.Activation.cReLU
    merlintf.keras.layers.Activation.ModReLU
    merlintf.keras.layers.Activation.cPReLU
    merlintf.keras.layers.Activation.ModPReLU
    merlintf.keras.layers.Activation.cStudentT
    merlintf.keras.layers.Activation.ModStudentT
    merlintf.keras.layers.Activation.cStudentT2
    merlintf.keras.layers.Activation.ModStudentT2
    merlintf.keras.layers.Activation.Identity
    merlintf.keras.layers.Activation.get
    merlintf.keras.layers.Activation.Cardioid
    merlintf.keras.layers.Activation.Cardioid2


Normalization
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.layers.ComplexBatchNormalization
    merlintf.keras.layers.ComplexInstanceNormalization


Models
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.models.cnn
    merlintf.keras.models.foe
    merlintf.keras.models.unet


Optimizers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.optimizers.blockadam


Losses
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    merlintf.keras.losses.loss_complex_mse
    merlintf.keras.losses.loss_complex_mse_2D
    merlintf.keras.losses.loss_complex_mse_3D
    merlintf.keras.losses.loss_complex_mse_2Dt
    merlintf.keras.losses.loss_complex_mse_3Dt
    merlintf.keras.losses.loss_abs_mse
    merlintf.keras.losses.loss_abs_mse_2D
    merlintf.keras.losses.loss_abs_mse_3D
    merlintf.keras.losses.loss_abs_mse_2Dt
    merlintf.keras.losses.loss_abs_mse_3Dt
    merlintf.keras.losses.loss_complex_mae
    merlintf.keras.losses.loss_complex_mae_2D
    merlintf.keras.losses.loss_complex_mae_3D
    merlintf.keras.losses.loss_complex_mae_2Dt
    merlintf.keras.losses.loss_complex_mae_3Dt
    merlintf.keras.losses.loss_abs_mae
    merlintf.keras.losses.loss_abs_mae_2D
    merlintf.keras.losses.loss_abs_mae_3D
    merlintf.keras.losses.loss_abs_mae_2Dt
    merlintf.keras.losses.loss_abs_mae_3Dt


Utils
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures: