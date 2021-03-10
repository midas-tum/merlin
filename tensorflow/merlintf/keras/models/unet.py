import tensorflow as tf
import tensorflow_addons as tfa
import merlintf
import numpy as np

import unittest
#import numpy as np
import optotf.keras.pad

__all__ = ['Real2chUNet',
           'MagUNet',
           'ComplexUNet']

class UNet(tf.keras.Model):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                       activation='relu', use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='UNet', use_padding=False, residual_output_add=False, **kwargs):
        """
        Abstract class for 2D/2D+t/3D/3D+t/4D UNet model
        input parameter:
        dim                         [string] operating dimension
        filters                     [integer, tuple] number of filters at the base level, dyadic increase
        kernel_size                 [integer, tuple] kernel size
        pool_size                   [integer, tuple] downsampling/upsampling operator size
        num_layer_per_level         number of convolutional layers per encocer/decoder level
        num_level                   amount of encoder/decoder stages (excluding bottleneck layer), network depth
        activation                  activation function
        use_bias                    apply bias for convolutional layer
        normalization               use normalization layers: BN (batch), IN (instance), none
        downsampling                downsampling operation: mp (max-pooling), st (stride)
        upsampling                  upsampling operation: us (upsampling), tc (transposed convolution)
        name                        specific identifier for network
        use_padding                 use padding on input and cropping on output in case of not matching, True: forces eager mode execution as dynamic shape extraction in graph mode not supported (tested for TF <=2.4)
        residual_output_add         adding aliased input to output for residual learning
        """
        super().__init__(name=name)

        # validate input dimensions
        self.kernel_size = merlintf.keras.utils.validate_input_dimension(dim, kernel_size)
        self.pool_size = merlintf.keras.utils.validate_input_dimension(dim, pool_size)

        self.dim = dim
        self.num_level = num_level
        self.num_layer_per_level = num_layer_per_level
        self.filters = filters
        self.use_bias = use_bias
        self.activation = activation
        self.normalization = normalization
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.use_padding = use_padding
        self.residual_output_add = residual_output_add
        if not use_padding:
            print('UNet: Check if input padding/output cropping is needed. No input shape specified, potentially switching to eager mode execution. Please provide input_shape by calling:')
            print('model.is_padding_needed(input_shape)')
            self.padding = self.is_padding_needed()

    def create_layers(self, **kwargs):
        # ------------- #
        # create layers #
        # ------------- #
        self.ops = []
        # encoder
        stage = []
        for ilevel in range(self.num_level):
            level = []
            for ilayer in range(self.num_layer_per_level):
                level.append(self.conv_layer(self.filters * (2**ilevel), self.kernel_size,
                                           strides=self.strides[ilayer],
                                           use_bias=self.use_bias,
                                           activation=self.activation,
                                           padding='same', **kwargs))
                level.append(callCheck(self.norm_layer, **kwargs))
                level.append(callCheck(self.activation_layer, **kwargs))

            if self.downsampling == 'mp':
                level.append(callCheck(self.down_layer, pool_size=self.pool_size, **kwargs))
            else:
                level.append(callCheck(self.down_layer, **kwargs))
            stage.append(level)
        self.ops.append(stage)

        # bottleneck
        stage = []
        for ilayer in range(self.num_layer_per_level):
            stage.append(self.conv_layer(self.filters * (2 ** (self.num_level)), self.kernel_size,
                                       strides=self.strides[ilayer],
                                       use_bias=self.use_bias,
                                       activation=self.activation,
                                       padding='same', **kwargs))
            stage.append(callCheck(self.norm_layer, **kwargs))
            stage.append(callCheck(self.activation_layer, **kwargs))
        if self.upsampling == 'us':
            stage.append(self.up_layer(self.pool_size, **kwargs))
        elif self.upsampling == 'tc':
            stage.append(self.up_layer(self.filters * (2 ** (self.num_level-1)), self.kernel_size,
                                       strides=self.pool_size,
                                       use_bias=self.use_bias,
                                       activation=self.activation,
                                       padding='same', **kwargs))
        self.ops.append(stage)

        # decoder
        stage = []
        for ilevel in range(self.num_level-1, -1, -1):
            level = []
            for ilayer in range(self.num_layer_per_level):
                level.append(self.conv_layer(self.filters * (2 ** ilevel), self.kernel_size,
                                           strides=1,
                                           use_bias=self.use_bias,
                                           activation=self.activation,
                                           padding='same', **kwargs))
                level.append(callCheck(self.norm_layer, **kwargs))
                level.append(callCheck(self.activation_layer, **kwargs))

            if ilevel > 0:
                if self.upsampling == 'us':
                    level.append(self.up_layer(self.pool_size, **kwargs))
                elif self.upsampling == 'tc':
                    level.append(self.up_layer(self.filters * (2 ** (ilevel-1)), self.kernel_size,
                                          strides=self.pool_size,
                                          use_bias=self.use_bias,
                                          activation=self.activation,
                                          padding='same', **kwargs))
            stage.append(level)
        self.ops.append(stage)

        # output convolution
        if self.residual_output_add:
            stage = []
            stage.append(self.conv_layer(self.out_cha, 1, strides=1,
                                               use_bias=self.use_bias,
                                               activation=None,
                                               padding='same', **kwargs))
            stage.append(self.activation_layer_last)
            self.ops.append(stage)
        else:
            self.ops.append(self.conv_layer(self.out_cha, 1, strides=1,
                                               use_bias=self.use_bias,
                                               activation=self.activation_last,
                                               padding='same', **kwargs))

    def is_padding_needed(self, in_shape=None):
        # in_shape (excluding batch and channel dimension!)
        if in_shape is None:  # input shape not specified or dynamically varying
            self.use_padding = True
            self.pad = None
        else:  # input shape specified
            self.pad = self.calculate_padding(in_shape)
            if np.all(np.asarray(self.pad) == 0):
                self.use_padding = False
            else:
                self.use_padding = True
        if self.use_padding:
            print('Enabling input padding and output cropping!')
            print('!!! Compile model with model.compile(run_eagerly=True) !!!')
        return self.use_padding

    def calculate_padding(self, in_shape):
        in_shape = np.asarray(in_shape)
        n_dim = merlintf.keras.utils.get_ndim(self.dim)
        if len(in_shape) > n_dim:
            in_shape = in_shape[:n_dim]
        factor = np.power(self.pool_size, self.num_level)
        paddings = np.ceil(in_shape / factor) * factor - in_shape
        pad = []
        for idx in range(n_dim):
            pad_top = paddings[idx].astype(np.int) // 2
            pad_bottom = paddings[idx].astype(np.int) - pad_top
            # pad.extend([pad_top, pad_bottom])  # for optox, TODO: check if reversed order of paddings
            pad.append((pad_top, pad_bottom))
        return tuple(pad)

    def calculate_padding_tensor(self, tensor):
        # calculate pad size
        # ATTENTION: input shape calculation with tf.keras.fit() ONLY possible in eager mode because of NoneType defined shapes! -> Force eager mode execution
        imshape = tensor.get_shape().as_list()
        if tf.keras.backend.image_data_format() == 'channels_last':  # default
            imshapenp = np.array(imshape[1:len(self.pool_size)+1]).astype(float)
        else:  # channels_first
            imshapenp = np.array(imshape[2:len(self.pool_size)+2]).astype(float)

        return self.calculate_padding(imshapenp)

    def call(self, inputs):
        if self.use_padding:
            if self.pad is None:  # input shape cannot be determined or fixed before compile
                pad = self.calculate_padding_tensor(inputs)
            else:
                pad = self.pad  # local variable to avoid permanent storage of fixed pad value in case of dynamic input shapes
            # xin = merlintf.keras.layers.pad(len(self.pool_size), inputs, pad, 'symmetric')  # symmetric padding via optox
            xin = self.pad_layer(pad)(inputs)
            x = xin  # xin needed for residual add forward
        else:
            x = inputs
        xforward = []
        # encoder
        for ilevel in range(self.num_level):
            for iop, op in enumerate(self.ops[0][ilevel]):
                if iop == len(self.ops[0][ilevel]) - 1:
                    xforward.append(x)
                if op is not None:
                    x = op(x)

        # bottleneck
        for op in self.ops[1]:
            if op is not None:
                x = op(x)

        # decoder
        for ilevel in range(self.num_level - 1, -1, -1):
            x = tf.keras.layers.concatenate([x, xforward[ilevel]])
            for op in self.ops[2][self.num_level - 1 - ilevel]:
                if op is not None:
                    x = op(x)

        # output convolution
        if self.residual_output_add:
            x = self.ops[3][0](x)
            x = tf.keras.layers.Add()([x, xin])
            x = self.ops[3][1](x)
        else:
            x = self.ops[3](x)
        if self.use_padding:
            x = self.crop_layer(pad)(x)
        return x

class RealUNet(UNet):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                 activation='relu', use_bias=True,
                 normalization='none', downsampling='mp', upsampling='tc',
                 name='RealUNet', use_padding=False, residual_output_add=False, **kwargs):
        """
        Builds the real-valued 2D/2D+t/3D/3D+t/4D UNet model (abstract class)
        """
        super().__init__(dim, filters, kernel_size, pool_size, num_layer_per_level, num_level, activation, use_bias, normalization, downsampling, upsampling, name, use_padding, residual_output_add)

        # get correct conv and input padding/output cropping operator
        if dim == '2D':
            self.conv_layer = tf.keras.layers.Conv2D
            self.pad_layer = optotf.keras.pad.Pad2d
            self.crop_layer = tf.keras.layers.Cropping2D
        elif dim == '3D':
            self.conv_layer = tf.keras.layers.Conv3D
            self.pad_layer = optotf.keras.pad.Pad3d
            self.crop_layer = tf.keras.layers.Cropping3D
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        # output convolution
        if 'activation_last' in kwargs:
            self.activation_last = kwargs.get('activation_last')
        else:
            self.activation_last = activation
        if residual_output_add:
            self.activation_layer_last = tf.keras.layers.Activation(self.activation_last)

        # get normalization operator
        if normalization == 'BN':
            self.norm_layer = tf.keras.layers.BatchNormalization
            self.activation_layer = tf.keras.layers.Activation(activation)
            self.activation = ''
        elif normalization == 'IN':
            self.norm_layer = tfa.layers.InstanceNormalization
            self.activation_layer = tf.keras.layers.Activation(activation)
            self.activation = ''
        elif normalization == 'none':
            self.norm_layer = None
            self.activation_layer = None
            self.activation = activation
        else:
            raise RuntimeError(f"Normalization for {normalization} not implemented!")

        # get downsampling operator
        if downsampling == 'mp':
            if dim == '2D':
                self.down_layer = tf.keras.layers.MaxPool2D
            elif dim == '3D':
                self.down_layer = tf.keras.layers.MaxPool3D
            else:
                raise RuntimeError(f"MaxPooling for dim={dim} not implemented!")
            self.strides = [1] * num_layer_per_level
        elif downsampling == 'st':
            self.down_layer = None
            self.strides = [1] * (num_layer_per_level - 1) + [2]
        else:
            raise RuntimeError(f"Downsampling operation {downsampling} not implemented!")

        # get upsampling operator
        if upsampling == 'us':
            if dim == '2D':
                self.up_layer = tf.keras.layers.UpSampling2D
            elif dim == '3D':
                self.up_layer = tf.keras.layers.UpSampling3D
            else:
                raise RuntimeError(f"Upsampling for dim={dim} not implemented!")
        elif upsampling == 'tc':
            if dim == '2D':
                self.up_layer = tf.keras.layers.Conv2DTranspose
            elif dim == '3D':
                self.up_layer = tf.keras.layers.Conv3DTranspose
            else:
                raise RuntimeError(f"Transposed convlutions for dim={dim} not implemented!")
        else:
            raise RuntimeError(f"Upsampling operation {upsampling} not implemented!")

class Real2chUNet(RealUNet):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                       activation='relu', use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='Real2chUNet', use_padding=False, residual_output_add=False, **kwargs):
        """
        Builds the real-valued 2-channel (real/imag or mag/pha in channel dim) 2D/2D+t/3D/3D+t/4D UNet model
        """
        super().__init__(dim, filters, kernel_size, pool_size, num_layer_per_level, num_level, activation, use_bias, normalization, downsampling, upsampling, name, use_padding, residual_output_add, **kwargs)
        self.out_cha = 2
        super().create_layers(**kwargs)

    def call(self, inputs):
        x = merlintf.complex2real(inputs)
        x = super().call(x)
        return merlintf.real2complex(x)

class MagUNet(RealUNet):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                       activation='relu', use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='MagUNet', use_padding=False, residual_output_add=False, **kwargs):
        """
        Builds the magnitude-based 2D/2D+t/3D/3D+t/4D UNet model (working on real or complex-valued input)
        """
        super().__init__(dim, filters, kernel_size, pool_size, num_layer_per_level, num_level, activation, use_bias, normalization, downsampling, upsampling, name, use_padding, residual_output_add, **kwargs)
        self.out_cha = 1
        super().create_layers(**kwargs)

    def call(self, inputs):
        x = merlintf.complex_abs(inputs)
        return super().call(x)

class ComplexUNet(UNet):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                       activation='ModReLU', use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='ComplexUNet', use_padding=False, residual_output_add=False, **kwargs):
        """
        Builds the complex-valued 2D/2D+t/3D/3D+t/4D UNet model
        """
        super().__init__(dim, filters, kernel_size, pool_size, num_layer_per_level, num_level, activation, use_bias, normalization, downsampling, upsampling, name, use_padding, residual_output_add)

        # get correct conv operator
        self.conv_layer = merlintf.keras.layers.ComplexConvolution(dim)
        if dim == '2D':
            self.pad_layer = optotf.keras.pad.Pad2d
        elif dim == '3D':
            self.pad_layer = optotf.keras.pad.Pad3d
        self.crop_layer = merlintf.keras.layers.Cropping(dim)

        # output convolution
        if 'activation_last' in kwargs:
            self.activation_last = kwargs.get('activation_last')
        else:
            self.activation_last = activation
        if residual_output_add:
            self.activation_layer_last = tf.keras.layers.Activation(self.activation_last)
        self.out_cha = 1

        # get normalization operator
        if normalization == 'BN':
            self.norm_layer = merlintf.keras.layers.ComplexBatchNormalization
            self.activation_layer = merlintf.keras.layers.Activation(activation)
            self.activation = ''
        elif normalization == 'IN':
            self.norm_layer = merlintf.keras.layers.ComplexInstanceNormalization
            self.activation_layer = merlintf.keras.layers.Activation(activation)
            self.activation = ''
        elif normalization == 'none':
            self.norm_layer = None
            self.activation_layer = None
            self.activation = activation
        else:
            raise RuntimeError(f"Normalization for {normalization} not implemented!")

        # get downsampling operator
        if downsampling == 'mp':
            self.down_layer = merlintf.keras.layers.MagnitudeMaxPooling(dim)
            self.strides = [1] * num_layer_per_level
        elif downsampling == 'st':
            self.down_layer = None
            self.strides = [1] * (num_layer_per_level-1) + [2]
        else:
            raise RuntimeError(f"Downsampling operation {downsampling} not implemented!")

        # get upsampling operator
        if upsampling == 'us':
            self.up_layer = merlintf.keras.layers.UpSampling(dim)  # TODO check if working for complex
        elif upsampling == 'tc':
            self.up_layer = merlintf.keras.layers.ComplexConvolutionTranspose(dim)
        else:
            raise RuntimeError(f"Upsampling operation {upsampling} not implemented!")

        super().create_layers(**kwargs)

def callCheck(fhandle, **kwargs):
    if fhandle is not None:
        return fhandle(**kwargs)
    else:
        return fhandle

class UNetTest(unittest.TestCase):
    def test_UNet_2chreal_2d(self):
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='2chreal', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='2chreal', complex_input=True)

    def test_UNet_mag_2d(self):
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='mag', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='mag', complex_input=True)

    def test_UNet_complex_2d(self):
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='complex', complex_input=False)
        self._test_UNet('2D', 64, (3, 3), (2, 2), network='complex', complex_input=True)

    def test_UNet_2chreal_3d(self):
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='2chreal', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='2chreal', complex_input=True)
        self._test_UNet('3D', 32, (1, 3, 3), (1, 2, 2), network='2chreal', complex_input=True)

    def test_UNet_mag_3d(self):
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='mag', complex_input=False)
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='mag', complex_input=True)
        self._test_UNet('3D', 32, (1, 3, 3), (1, 2, 2), network='mag', complex_input=True)

    def test_UNet_mag_3d_padding(self):
        self._test_UNet('3D', 32, (3, 3, 3), (2, 2, 2), network='mag', complex_input=True, D=20, M=32, N=32) # padding required

    #def test_UNet_complex_3d(self):
    #    self._test_UNet('3D', 32, (3, 3, 3), network='complex', complex_input=False)
    #    self._test_UNet('3D', 32, (3, 3, 3), network='complex', complex_input=True)

    def _test_UNet(self, dim, filters, kernel_size, down_size=(2,2,2), network='complex', complex_input=True, D=30, M=32, N=32):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental_run_functions_eagerly(False)

        nBatch = 2

        if network == 'complex':
            model = ComplexUNet(dim, filters, kernel_size, down_size)
        elif network =='2chreal':
            model = Real2chUNet(dim, filters, kernel_size, down_size)
        else:
            model = MagUNet(dim, filters, kernel_size, down_size)

        if dim == '2D':
            if complex_input:
                x = merlintf.random_normal_complex((nBatch, M, N, 1), dtype=tf.float32)
            else:
                x = tf.random.normal((nBatch, M, N, 1), dtype=tf.float32)
        elif dim == '3D' or dim == '2Dt':
            if complex_input:
                x = merlintf.random_normal_complex((nBatch, D, M, N, 1), dtype=tf.float32)
            else:
                x = tf.random.normal((nBatch, D, M, N, 1), dtype=tf.float32)
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')

        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)

if __name__ == "__main__":
    unittest.main()