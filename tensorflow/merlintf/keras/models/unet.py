import tensorflow as tf
import tensorflow_addons as tfa
import merlintf
import numpy as np

__all__ = ['Real2chUNet',
           'MagUNet',
           'ComplexUNet']

class UNet(tf.keras.Model):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                       activation='relu', activation_last='relu', kernel_size_last=1, use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='UNet', padding='none', **kwargs):
        """
        Abstract class for 2D/2D+t/3D/3D+t/4D UNet model
        input parameter:
        dim                         [string] operating dimension
        filters                     [integer, tuple] number of filters at the base level, dyadic increase
        kernel_size                 [integer, tuple] kernel size
        pool_size                   [integer, tuple] downsampling/upsampling operator size
        num_layer_per_level         [integer] number of convolutional layers per encocer/decoder level
        num_level                   [integer] amount of encoder/decoder stages (excluding bottleneck layer), network depth
        activation                  [string] activation function
        activation_last             [string] activation function of last layer
        kernel_size_last            [integer, tuple] kernel size in last layer
        use_bias                    [bool] apply bias for convolutional layer
        normalization               [string] use normalization layers: BN (batch), IN (instance), none|None
        downsampling                downsampling operation: mp (max-pooling), st (stride)
        upsampling                  upsampling operation: us (upsampling), tc (transposed convolution)
        name                        specific identifier for network
        padding                     [string] padding on input and cropping on output in case of not matching with demanded pool_size and num_layer: zero (zero-padding layer), reflect (optotf), symmetric (optotf), replicate (optotf), none|None (in_shape check required), force_none (avoid padding and eager execution)
                                    if padding is used: forces eager mode execution as dynamic shape extraction in graph mode not supported (tested for TF <=2.4)
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
        self.activation_last = activation_last
        self.kernel_size_last = merlintf.keras.utils.validate_input_dimension(dim, kernel_size_last)
        self.normalization = normalization
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.padding = padding
        self.kwargs = kwargs
        if 'in_shape' in kwargs:
            self.use_padding = self.is_padding_needed(kwargs.get('in_shape'))
        else:
            self.use_padding = self.is_padding_needed()  # in_shape at init time not known

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.use_padding = self.is_padding_needed(input_shape)
        self.create_layers()

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
                                           strides=1,
                                           use_bias=self.use_bias,
                                           activation=self.activation,
                                           padding='same', **kwargs))
                level.append(callCheck(self.norm_layer, **kwargs))
                level.append(self.activation_layer if not isinstance(self.activation_layer, str) else merlintf.keras.layers.Activation(self.activation_layer))  # workaround to have trainable activations (complex-valued UNet)

            if self.downsampling == 'mp':
                level.append(callCheck(self.down_layer, pool_size=self.pool_size, **kwargs))
            else:
                level.append(self.down_layer(self.filters * (2**ilevel), self.kernel_size,
                                           strides=self.pool_size,
                                           use_bias=self.use_bias,
                                           activation=self.activation,
                                           padding='same', **kwargs))
            stage.append(level)
        self.ops.append(stage)

        # bottleneck
        stage = []
        for ilayer in range(self.num_layer_per_level):
            stage.append(self.conv_layer(self.filters * (2 ** (self.num_level)), self.kernel_size,
                                       strides=1,
                                       use_bias=self.use_bias,
                                       activation=self.activation,
                                       padding='same', **kwargs))
            stage.append(callCheck(self.norm_layer, **kwargs))
            stage.append(self.activation_layer if not isinstance(self.activation_layer, str) else merlintf.keras.layers.Activation(self.activation_layer))  # workaround to have trainable activations (complex-valued UNet)
        if self.upsampling == 'us':
            stage.append(self.conv_layer(self.filters * (2 ** (self.num_level - 1)), 1,
                                       strides=1,
                                       use_bias=self.use_bias,
                                       activation=self.activation,
                                       padding='same', **kwargs))
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
                level.append(self.activation_layer if not isinstance(self.activation_layer, str) else merlintf.keras.layers.Activation(self.activation_layer))  # workaround to have trainable activations (complex-valued UNet)

            if ilevel > 0:
                if self.upsampling == 'us':
                    level.append(self.conv_layer(self.filters * (2 ** (ilevel - 1)), 1,
                                                 strides=1,
                                                 use_bias=self.use_bias,
                                                 activation=self.activation,
                                                 padding='same', **kwargs))
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
        self.ops.append(self.conv_layer(self.out_cha, self.kernel_size_last, strides=1,
                                           use_bias=self.use_bias,
                                           activation=self.activation_last,
                                           padding='same', **kwargs))

    def is_padding_needed(self, in_shape=None):
        # in_shape (excluding batch and channel dimension!)
        if not self.padding.lower() == 'none' and in_shape is None:
            print('merlintf.keras.models.unet: Check if input padding/output cropping is needed. No input shape specified, potentially switching to eager mode execution. Please provide input_shape by calling: model.is_padding_needed(input_shape)')
        if in_shape is None:  # input shape not specified or dynamically varying
            self.use_padding = True
            self.pad = None
            self.optotf_pad = None
        else:  # input shape specified
            self.pad, self.optotf_pad = self.calculate_padding(in_shape)
            if np.all(np.asarray(self.pad) == 0):
                self.use_padding = False
            else:
                self.use_padding = True
        if self.padding.lower() == 'force_none':
            self.use_padding = False
            self.pad = None
            self.optotf_pad = None
        if self.use_padding:
            if self.padding.lower() == 'none':
                self.padding = 'zero'  # default padding
            #print('Safety measure: Enabling input padding and output cropping!')
            #print('!!! Compile model with model.compile(run_eagerly=True) !!!')
        return self.use_padding

    def calculate_padding(self, in_shape):
        in_shape = np.asarray(in_shape)
        n_dim = merlintf.keras.utils.get_ndim(self.dim)
        if len(in_shape) > n_dim+1:  # assumed in_shape: batch x X x Y x ...
            in_shape = in_shape[1:n_dim+1]
        factor = np.power(self.pool_size, self.num_level)
        paddings = np.ceil(in_shape / factor) * factor - in_shape
        pad = []
        optotf_pad = []
        for idx in range(n_dim):
            pad_top = paddings[idx].astype(np.int) // 2
            pad_bottom = paddings[idx].astype(np.int) - pad_top
            optotf_pad.extend([pad_top, pad_bottom])
            pad.append((pad_top, pad_bottom))
        return tuple(pad), optotf_pad[::-1]

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
                pad, optotf_pad = self.calculate_padding_tensor(inputs)
            else:
                pad = self.pad  # local variable to avoid permanent storage of fixed pad value in case of dynamic input shapes
                optotf_pad = self.optotf_pad
            if self.padding.lower() == 'zero':
                x = self.pad_layer(pad)(inputs)
            else:
                x = self.pad_layer(inputs, optotf_pad, self.padding)
            # x = merlintf.keras.layers.pad(len(self.pool_size), inputs, optotf_pad, 'symmetric')  # symmetric padding via optox
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
        x = self.ops[3](x)
        if self.use_padding:
            x = self.crop_layer(pad)(x)
        return x

class RealUNet(UNet):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                 activation='relu', activation_last='relu', kernel_size_last=1, use_bias=True,
                 normalization='none', downsampling='mp', upsampling='tc',
                 name='RealUNet',  padding='none', **kwargs):
        """
        Builds the real-valued 2D/2D+t/3D/3D+t/4D UNet model (abstract class)
        """
        super().__init__(dim, filters, kernel_size, pool_size, num_layer_per_level, num_level, activation, activation_last, kernel_size_last, use_bias, normalization, downsampling, upsampling, name, padding, **kwargs)

        # get correct conv and input padding/output cropping operator
        if dim == '2D':
            self.conv_layer = tf.keras.layers.Conv2D
            if self.padding.lower() == 'zero':
                self.pad_layer = tf.keras.layers.ZeroPadding2D
            else:
                self.pad_layer = merlintf.keras.layers.Padding2D
            self.crop_layer = tf.keras.layers.Cropping2D
        elif dim == '2Dt':
            self.conv_layer = merlintf.keras.layers.Conv2Dt
            if self.padding.lower() == 'zero':
                self.pad_layer = tf.keras.layers.ZeroPadding3D
            else:
                self.pad_layer = merlintf.keras.layers.Padding3D
            self.crop_layer = tf.keras.layers.Cropping3D
        elif dim == '3D':
            self.conv_layer = tf.keras.layers.Conv3D
            if self.padding.lower() == 'zero':
                self.pad_layer = tf.keras.layers.ZeroPadding3D
            else:
                self.pad_layer = merlintf.keras.layers.Padding3D
            self.crop_layer = tf.keras.layers.Cropping3D
        elif dim == '3Dt':
            self.conv_layer = merlintf.keras.layers.Conv3Dt
            if self.padding.lower() == 'zero':
                self.pad_layer = merlintf.keras.layers.ZeroPadding4D
            else:
                self.pad_layer = merlintf.keras.layers.Padding4D
            self.crop_layer = merlintf.keras.layers.Cropping4D
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        # output convolution
        #self.activation_layer_last = tf.keras.layers.Activation(self.activation_last)

        # get normalization operator
        if normalization == 'BN':
            self.norm_layer = tf.keras.layers.BatchNormalization
            self.activation_layer = tf.keras.layers.Activation(activation)
            self.activation = None
        elif normalization == 'IN':
            self.norm_layer = tfa.layers.InstanceNormalization
            self.activation_layer = tf.keras.layers.Activation(activation)
            self.activation = None
        elif normalization.lower() == 'none':
            self.norm_layer = None
            self.activation_layer = None
            self.activation = activation
        else:
            raise RuntimeError(f"Normalization for {normalization} not implemented!")

        # get downsampling operator
        n_dim = merlintf.keras.utils.get_ndim(self.dim)
        if downsampling == 'mp':
            if dim == '2D':
                self.down_layer = merlintf.keras.layers.MagnitudeMaxPool2D
            elif dim == '2Dt':
                self.down_layer = merlintf.keras.layers.MagnitudeMaxPool2Dt  # internally resorts to 3D pooling
            elif dim == '3D':
                self.down_layer = merlintf.keras.layers.MagnitudeMaxPool3D
            elif dim == '3Dt':
                self.down_layer = merlintf.keras.layers.MagnitudeMaxPool3Dt
            else:
                raise RuntimeError(f"MaxPooling for dim={dim} not implemented!")
            self.strides = [1] * num_layer_per_level
        elif downsampling == 'st':
            self.down_layer = self.conv_layer
            self.strides = [[1] * n_dim] * num_layer_per_level + [list(self.pool_size)]
        else:
            raise RuntimeError(f"Downsampling operation {downsampling} not implemented!")

        # get upsampling operator
        if upsampling == 'us':
            if dim == '2D':
                self.up_layer = tf.keras.layers.UpSampling2D
            elif dim == '2Dt':
                self.up_layer = tf.keras.layers.UpSampling3D
            elif dim == '3D':
                self.up_layer = tf.keras.layers.UpSampling3D
            elif dim == '3Dt':
                self.up_layer = merlintf.keras.layers.UpSampling4D
            else:
                raise RuntimeError(f"Upsampling for dim={dim} not implemented!")
        elif upsampling == 'tc':
            if dim == '2D':
                self.up_layer = tf.keras.layers.Conv2DTranspose
            elif dim == '2Dt':
                self.up_layer = merlintf.keras.layers.Conv2DtTranspose
            elif dim == '3D':
                self.up_layer = tf.keras.layers.Conv3DTranspose
            elif dim == '3Dt':
                self.up_layer = merlintf.keras.layers.Conv3DtTranspose
            else:
                raise RuntimeError(f"Transposed convlutions for dim={dim} not implemented!")
        else:
            raise RuntimeError(f"Upsampling operation {upsampling} not implemented!")

class Real2chUNet(RealUNet):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                       activation='relu', activation_last=None, kernel_size_last=1, use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='Real2chUNet',  padding='none', **kwargs):
        """
        Builds the real-valued 2-channel (real/imag or mag/pha in channel dim) 2D/2D+t/3D/3D+t/4D UNet model
        """
        super().__init__(dim, filters, kernel_size, pool_size, num_layer_per_level, num_level, activation, activation_last, kernel_size_last, use_bias, normalization, downsampling, upsampling, name, padding, **kwargs)
        self.out_cha = 2
        #super().create_layers(**kwargs)

    def call(self, inputs):
        x = merlintf.complex2real(inputs)
        x = super().call(x)
        return merlintf.real2complex(x)

class MagPhaUNet(RealUNet):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                       activation='relu', activation_last=None, kernel_size_last=1, use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='MagPhaUNet',  padding='none', **kwargs):
        """
        Builds the real-valued 2-channel (real/imag or mag/pha in channel dim) 2D/2D+t/3D/3D+t/4D UNet model
        """
        super().__init__(dim, filters, kernel_size, pool_size, num_layer_per_level, num_level, activation, activation_last, kernel_size_last, use_bias, normalization, downsampling, upsampling, name, padding, **kwargs)
        self.out_cha = 2
        #super().create_layers(**kwargs)

    def call(self, inputs):
        x = merlintf.complex2magpha(inputs)
        x = super().call(x)
        return merlintf.magpha2complex(x)

class MagUNet(RealUNet):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                       activation='relu', activation_last='relu', kernel_size_last=1, use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='MagUNet',  padding='none', **kwargs):
        """
        Builds the magnitude-based 2D/2D+t/3D/3D+t/4D UNet model (working on real or complex-valued input)
        """
        super().__init__(dim, filters, kernel_size, pool_size, num_layer_per_level, num_level, activation, activation_last, kernel_size_last, use_bias, normalization, downsampling, upsampling, name, padding, **kwargs)
        self.out_cha = 1
        #super().create_layers(**kwargs)

    def call(self, inputs):
        x = merlintf.complex_abs(inputs)
        return super().call(x)

class ComplexUNet(UNet):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                       activation='ModReLU', activation_last='ModReLU', kernel_size_last=1, use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='ComplexUNet',  padding='none', **kwargs):
        """
        Builds the complex-valued 2D/2D+t/3D/3D+t/4D UNet model
        """
        super().__init__(dim, filters, kernel_size, pool_size, num_layer_per_level, num_level, activation, activation_last, kernel_size_last, use_bias, normalization, downsampling, upsampling, name, padding, **kwargs)

        # get correct conv operator
        self.conv_layer = merlintf.keras.layers.ComplexConvolution(dim)
        if self.padding.lower() == 'zero':
            self.pad_layer = merlintf.keras.layers.ZeroPadding(dim)
        else:
            if self.dim == '2D':
                self.pad_layer = merlintf.keras.layers.Padding2D
            if self.dim == '2Dt':
                self.pad_layer = merlintf.keras.layers.Padding3D
            elif self.dim == '3D':
                self.pad_layer = merlintf.keras.layers.Padding3D
            if self.dim == '3Dt':
                self.pad_layer = merlintf.keras.layers.Padding4D
            else:
                raise RuntimeError(f"Padding for {dim} and {self.padding} not implemented!")

        self.crop_layer = merlintf.keras.layers.Cropping(dim)

        # output convolution
        #self.activation_layer_last = merlintf.keras.layers.Activation(self.activation_last)
        self.out_cha = 1

        # get normalization operator
        if normalization == 'BN':
            self.norm_layer = merlintf.keras.layers.ComplexBatchNormalization
            self.activation_layer = self.activation  # workaround: put in str so that during creation a new act layer is created each time (since they can be trainable)
            self.activation = None
        elif normalization == 'IN':
            self.norm_layer = merlintf.keras.layers.ComplexInstanceNormalization
            self.activation_layer = self.activation
            self.activation = None
        elif normalization.lower() == 'none':
            self.norm_layer = None
            self.activation_layer = None
            self.activation = activation
        else:
            raise RuntimeError(f"Normalization for {normalization} not implemented!")

        # get downsampling operator
        n_dim = merlintf.keras.utils.get_ndim(self.dim)
        if downsampling == 'mp':
            self.down_layer = merlintf.keras.layers.MagnitudeMaxPooling(dim)
            self.strides = [1] * num_layer_per_level
        elif downsampling == 'st':
            self.down_layer = self.conv_layer
            self.strides = [[1] * n_dim] * num_layer_per_level + [list(self.pool_size)]
        else:
            raise RuntimeError(f"Downsampling operation {downsampling} not implemented!")

        # get upsampling operator
        if upsampling == 'us':
            self.up_layer = merlintf.keras.layers.UpSampling(dim)  # TODO check if working for complex
        elif upsampling == 'tc':
            self.up_layer = merlintf.keras.layers.ComplexConvolutionTranspose(dim)
        else:
            raise RuntimeError(f"Upsampling operation {upsampling} not implemented!")

        #super().create_layers(**kwargs)


def callCheck(fhandle, **kwargs):
    def isempty(x):
        return False if x else True
    if fhandle is not None:
        if kwargs is None or isempty(kwargs):
            return fhandle()
        else:
            return fhandle(**kwargs)
    else:
        return fhandle
