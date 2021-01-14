import tensorflow as tf

import numpy as np
from keras import backend as K
#from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, PReLU, Conv3DTranspose, \
    Conv2D, Conv2DTranspose, MaxPooling2D, GlobalMaxPool2D, UpSampling2D, \
    Input, BatchNormalization, Activation, Dense, Dropout, Lambda, Layer, Concatenate, Add, Subtract

from keras.models import Model, load_model
#from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.merge import concatenate, add

#K.set_image_data_format = 'channels_first'


class UNet:
    def __init__(self, input_shape, kspace_input_shape, mask_shape, sensemap_shape, lambdaval, range_scale,
                 n_stages=3, depth=4, n_conv_per_level=2, n_dim=3,
                 pool_size=(2, 2, 2, 1), strides=(1, 1, 1, 1), kernel=(5, 5, 5, 5),
                 deconvolution=True, n_base_filters=2,
                 normalization=1, activation='relu', activation_last_layer="sigmoid", padding='same', complex_conv=False,
                 conv_separable=True, convolution_4D='looped', reuse=True,
                 prescale_kspace=True, input_type='numpy', real_imag=True, is_complex=True, network_type = 'unet'):
        """
        Builds the 2D/3D/3D+t/4D cascaded UNet as keras model
        input parameter:
        input_shape, kspace_input_shape, mask_shape, sensemap_shape, lambdaval      input, kspace, sampling mask, sense map and regularization lambda as input
        range_scale                                                                 dynamic scaling range of input data
        pool_size, strides, kernel                                                  network parameter, dimension (X,Y,Z,time)
        n_stages                                                                    amount of UNet stages (cascaded UNets) = unrolled iterations
        depth                                                                       network depth of UNet
        n_conv_per_level                                                            amount of convolutional filters inside one UNet level
        n_dim                                                                       input dimensionality: 2 (XxY), 3 (XxYxZ), 4 (XxYxZxTime)
        deconvolution                                                               use transpose convolution (True) or upsampling (False)
        n_base_filters                                                              kernels in 1st conv layer, dyadic increase
        normalization                                                               normalization: 0=none, 1=batch normalization, 2=instance normalization
        activation                                                                  activation functions
        activation_last_layer                                                       activation function in last layer
        padding                                                                     padding of conv layer
        complex_conv                                                                perform complex-valued convolution (True) or real-valued (False)
        conv_separable                                                              use 3D+1D convolution
        convolution_4D                                                              implementation type of 4D convolution
        reuse                                                                       kernel weight sharing (4D convolution)
        prescale_kspace                                                             same dynamic range of k-space in DC
        input_type                                                                  data processing format
        real_imag                                                                   real and imaginary (True) or magnitude and phase (False)
        is_complex                                                                  complex-valued (True) or real-valued (False) input

        :return: N-D UNet Model
        """
        # save memory and store only one mask per freq-line
        mask_shape = mask_shape.copy()
        mask_shape[3] = 1
        if n_dim == 4:  # channels x (complex) x time x X x Y x Z
            self.input_shape = input_shape
            self.kspace_input_shape = kspace_input_shape
            self.mask_shape = mask_shape
            self.sensemap_shape = sensemap_shape
            self.pool_size = [pool_size[idx] for idx in [3, 0, 1, 2]]
            self.strides = [strides[idx] for idx in [3, 0, 1, 2]]
            self.kernel = [kernel[idx] for idx in [3, 0, 1, 2]]
            self.n_filter_last_layer = 1
            self.conv_separable = conv_separable  # use separable 3D+1D convolution
            self.kernel_out = [1, 1, 1, 1]
        elif n_dim == 3:  # channels x (complex) x X x Y x Z
            self.input_shape = input_shape
            self.kspace_input_shape = kspace_input_shape
            self.mask_shape = mask_shape
            self.sensemap_shape = sensemap_shape
            self.pool_size = [pool_size[idx] for idx in [0, 1, 2]]
            self.strides = [strides[idx] for idx in [0, 1, 2]]
            self.kernel = [kernel[idx] for idx in [0, 1, 2]]
            self.n_filter_last_layer = 1
            self.conv_separable = conv_separable  # use separable 3D+1D convolution
            self.kernel_out = [1, 1, 1]
        elif n_dim == 2:  # channels x (complex) x X x Y
            self.input_shape = input_shape
            self.kspace_input_shape = kspace_input_shape
            self.mask_shape = mask_shape
            self.sensemap_shape = sensemap_shape
            self.pool_size = [pool_size[idx] for idx in [0, 1]]
            self.strides = [strides[idx] for idx in [0, 1]]
            self.kernel = [kernel[idx] for idx in [0, 1]]
            self.n_filter_last_layer = 1
            self.conv_separable = conv_separable  # use separable 3D+1D convolution
            self.kernel_out = [1, 1]

        self.lambdaval = lambdaval
        self.n_stages = n_stages        # #cascaded UNets
        self.depth = depth              # depth of UNet
        self.n_dim = np.floor(n_dim)    # imaging dimensionality
        self.n_dim_datacon = n_dim

        self.deconvolution = deconvolution
        if isinstance(n_base_filters, list):  # list of channels per level in each stage
            self.n_base_filters = n_base_filters
        else:  # scalar -> prepare by multiplication of 2
            self.n_base_filters = list()
            for i in range(n_conv_per_level):
                self.n_base_filters.append(n_base_filters * (2**i))    # The number of filters that the first layer in the convolution network will have. Following  layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required to train the model.
        self.n_conv_per_level = n_conv_per_level  # number of convolutional filters per level in encoder/decoder stage
        self.normalization = normalization  # 0 = none, 1 = batch normalization, 2 = instance normalization
        self.activation = activation                                # activation of internal layers
        self.activation_last_layer = activation_last_layer          # activation of last layer
        self.padding = padding
        self.reuse = reuse                  # reuse same 3D conv filter along all temp states
        self.complex_conv = complex_conv    # use complex convolutions
        self.convolution_4D = convolution_4D   # 4D convolutions (full or looped)
        self.prescale_kspace = prescale_kspace  # pre-scale kspace in data generator
        self.input_type = input_type  # input type of data (numpy) or (tfrecord)
        #self.range_scale = tf.constant(range_scale, dtype=tf.float32)
        self.range_scale = range_scale
        self.real_imag = real_imag   # real/imag or abs/pha processing
        self.is_complex = is_complex  # input is complex-valued (dimension 1)
        self.network_type = network_type  # network type
        print('Building a %s' % network_type)
        self.data_format = 'channels_first'
        K.set_image_data_format = self.data_format

    def cascaded_net(self, intensor=None):
        #if self.input_type == 'numpy':
        inputs = Input(self.input_shape)
        kspace_in = Input(self.kspace_input_shape)
        mask_in = Input(self.mask_shape)

        if not self.prescale_kspace:
            scale_orig = Input((1, 2))
            scale_net = Input((1, 2))
        else:
            scale_orig = None
            scale_net = None
        lambdaval = Input((1, 1), name='lambda_in')  # channel and scalar dim

        current_in = inputs

        # cascaded stages
        for istage in range(self.n_stages):
            print('== building stage %d/%d' % (istage+1, self.n_stages))
            if self.network_type == 'unet':
                outnet = self.unet_3D(current_in, istage)
            print('-- unet %d finished' % (istage+1))
            if (self.lambdaval > 0) & (istage < self.n_stages-1):
                outdata = self.data_consistency(outnet, kspace_in, mask_in, scale_orig, scale_net, None, lambdaval)
                print('-- data consistency finished')
                current_out = outdata
                current_in = outdata
            else:
                current_out = outnet
                current_in = outnet

        #current_out = Activation('sigmoid')(current_out)
        if not self.prescale_kspace:
            model = Model(inputs=[inputs, kspace_in, mask_in, lambdaval, scale_orig, scale_net], outputs=current_out) # make them all keras layers, inside must be one layer which is not!!!!
        else:
            model = Model(inputs=[inputs, kspace_in, mask_in, lambdaval], outputs=current_out)
        print('== Model generation done!')
        return model

    def data_consistency(self, inputs, kspace_input, mask, scale_orig, scale_net, sensemap, lambdaval):
        # data consistency
        # CG, Proximal, ...
        return inputs

    def unet_3D(self, inputs, stage=0):
        #inputs = Input(input_shape)
        current_layer = inputs
        levels = list()

        # add levels with max pooling
        print('Encoder level ', end='', flush=True)
        for layer_depth in range(self.depth):
            print('%d..' % layer_depth, end='', flush=True)
            layers_in_level = list()
            for i_conv_per_level in range(self.n_conv_per_level):
                current_layer = self.create_convolution_block(input_layer=current_layer, n_filters=self.n_base_filters[i_conv_per_level]*(2**layer_depth), stage=stage, level=layer_depth, filter_bank=i_conv_per_level)
                layers_in_level.append(current_layer)

            if layer_depth < self.depth - 1:
                current_layer = self.create_down_sampling(current_layer, stage, layer_depth, self.pool_size)
                layers_in_level.append(current_layer)

            levels.append(layers_in_level)

        print('')
        print('Decoder level ', end='', flush=True)
        # add levels with up-convolution or up-sampling
        for layer_depth in range(self.depth-2, -1, -1):
            print('%d..' % layer_depth, end='', flush=True)
            layers_in_level = list()
            try:
                n_filters = levels[layer_depth][-1]._keras_shape[1]
            except:
                n_filters = levels[layer_depth][-1].get_shape().as_list()[1]
            up_convolution = self.create_up_convolution(current_layer, n_filters=n_filters, stage=stage, level=layer_depth)
            layers_in_level.append(up_convolution)
            concat = concatenate([up_convolution, levels[layer_depth][-2]], axis=1)
            layers_in_level.append(concat)

            current_layer = concat
            for i_conv_per_level in range(self.n_conv_per_level):
                try:
                    n_filters = levels[layer_depth][-1]._keras_shape[1]
                except:
                    n_filters = levels[layer_depth][-1].get_shape().as_list()[1]
                current_layer = self.create_convolution_block(input_layer=current_layer, n_filters=n_filters, stage=stage, level=layer_depth, filter_bank=self.n_conv_per_level+i_conv_per_level)
                layers_in_level.append(current_layer)

            levels.append(layers_in_level)

        print('')
        if self.n_conv_per_level > 1:
            final_convolution = self.create_convolution_block(current_layer, n_filters=self.n_filter_last_layer, kernel=self.kernel_out, stage=stage, level=0, filter_bank=2*self.n_conv_per_level+1, activation=self.activation_last_layer)
        else:
            final_convolution = current_layer
            #act = Activation(self.activation_last_layer)(final_convolution)

        return final_convolution

    def create_convolution_block(self, input_layer, n_filters, stage, level, filter_bank=0, kernel=self.kernel, normalization=1, activation=self.activation):

        # prepare some layers
        expandLayer = Lambda(lambda x: K.expand_dims(x, axis=1))
        stackLayer = Lambda(lambda x: K.concatenate(x, axis=1))
        layer_mag_pha = Lambda(lambda x: to_mag_pha(x, self.n_dim))
        layer_real_imag = Lambda(lambda x: to_real_imag(x))
        if self.n_dim == 2:
            sliceLayerReal = Lambda(lambda x: x[:, 0, :, :])
            sliceLayerImag = Lambda(lambda x: x[:, 1, :, :])
        elif self.n_dim == 3:
            sliceLayerReal = Lambda(lambda x: x[:, 0, :, :, :])
            sliceLayerImag = Lambda(lambda x: x[:, 1, :, :, :])
        elif self.n_dim == 3:
            sliceLayerReal = Lambda(lambda x: x[:, 0, :, :, :, :])
            sliceLayerImag = Lambda(lambda x: x[:, 1, :, :, :, :])

        if self.n_dim == 2:
            if self.is_complex:
                if self.complex_conv:
                    input_real = sliceLayerReal(input_layer)
                    input_imag = sliceLayerImag(input_layer)
                    conv_real_spatial = Conv2D(n_filters, kernel, padding=self.padding, strides=self.strides,
                                               data_format=self.data_format, input_shape=self.input_shape)
                    conv_imag_spatial = Conv2D(n_filters, kernel, padding=self.padding, strides=self.strides,
                                               data_format=self.data_format)
                    layer_real = Subtract()(
                        [conv_real_spatial(input_real), conv_imag_spatial(input_imag)])
                    layer_imag = Add()(
                        [conv_imag_spatial(input_real), conv_real_spatial(input_imag)])
                    layer_real = expandLayer(layer_real)
                    layer_imag = expandLayer(layer_imag)
                    layer = stackLayer([layer_real, layer_imag])
                else:
                    input_real = sliceLayerReal(input_layer)
                    layer_real = Conv2D(n_filters, kernel, padding=self.padding, strides=self.strides,
                                        data_format=self.data_format, input_shape=self.input_shape)(input_real)
                    layer_real = expandLayer(layer_real)
                    input_imag = sliceLayerImag(input_layer)
                    layer_imag = Conv2D(n_filters, kernel, padding=self.padding, strides=self.strides,
                                        data_format=self.data_format)(input_imag)
                    layer_imag = expandLayer(layer_imag)
                    layer = stackLayer([layer_real, layer_imag])
            else:
                 layer = Conv2D(n_filters, kernel, padding=self.padding, strides=self.strides, input_shape=self.input_shape)(input_layer)
        elif self.n_dim == 3:
            if self.is_complex:
                if self.complex_conv:
                    input_real = sliceLayerReal(input_layer)
                    input_imag = sliceLayerImag(input_layer)
                    conv_real_spatial = Conv3D(n_filters, kernel, padding=self.padding, strides=self.strides,
                                        data_format=self.data_format, input_shape=self.input_shape)
                    conv_imag_spatial = Conv3D(n_filters, kernel, padding=self.padding, strides=self.strides,
                                        data_format=self.data_format)
                    layer_real = Subtract()(
                        [conv_real_spatial(input_real), conv_imag_spatial(input_imag)])
                    layer_imag = Add()(
                        [conv_imag_spatial(input_real), conv_real_spatial(input_imag)])
                    layer_real = expandLayer(layer_real)
                    layer_imag = expandLayer(layer_imag)
                    layer = stackLayer([layer_real, layer_imag])
                else:  # real-valued conv (of complex data)
                    input_real = sliceLayerReal(input_layer)
                    layer_real = Conv3D(n_filters, kernel, padding=self.padding, strides=self.strides, data_format=self.data_format, input_shape=self.input_shape)(input_real)
                    layer_real = expandLayer(layer_real)
                    input_imag = sliceLayerImag(input_layer)
                    layer_imag = Conv3D(n_filters, kernel, padding=self.padding, strides=self.strides, data_format=self.data_format)(input_imag)
                    layer_imag = expandLayer(layer_imag)
                    layer = stackLayer([layer_real, layer_imag])
            else:
                layer = Conv3D(n_filters, kernel, padding=self.padding, strides=self.strides, input_shape=self.input_shape)(input_layer)

        elif self.n_dim == 4:
            if self.is_complex:
                if self.complex_conv:
                    input_real = sliceLayerReal(input_layer)
                    input_imag = sliceLayerImag(input_layer)
                    conv_real_spatial = Conv4D(n_filters, kernel, padding=self.padding, strides=self.strides,
                                        data_format=self.data_format, input_shape=self.input_shape)
                    conv_imag_spatial = Conv4D(n_filters, kernel, padding=self.padding, strides=self.strides,
                                        data_format=self.data_format)
                    layer_real = Subtract()(
                        [conv_real_spatial(input_real), conv_imag_spatial(input_imag)])
                    layer_imag = Add()(
                        [conv_imag_spatial(input_real), conv_real_spatial(input_imag)])
                    layer_real = expandLayer(layer_real)
                    layer_imag = expandLayer(layer_imag)
                    layer = stackLayer([layer_real, layer_imag])
                else:  # real-valued conv (of complex data)
                    input_real = sliceLayerReal(input_layer)
                    layer_real = Conv4D(n_filters, kernel, padding=self.padding, strides=self.strides, data_format=self.data_format, input_shape=self.input_shape)(input_real)
                    layer_real = expandLayer(layer_real)
                    input_imag = sliceLayerImag(input_layer)
                    layer_imag = Conv4D(n_filters, kernel, padding=self.padding, strides=self.strides, data_format=self.data_format)(input_imag)
                    layer_imag = expandLayer(layer_imag)
                    layer = stackLayer([layer_real, layer_imag])
            else:
                layer = Conv4D(input_layer, n_filters, kernel, padding=self.padding, strides=self.strides, stage=stage, level=level, filter_bank=filter_bank)

        if normalization == 1:  # batch normalization
            if self.is_complex:
                if self.real_imag:
                    input_mag_pha = layer_mag_pha(layer)
                    layer = ComplexBatchNormalization(axis=1)(input_mag_pha)
                    layer = layer_real_imag(layer)
                else:
                    layer = ComplexBatchNormalization(axis=1)(layer)
            else:
                layer = BatchNormalization(axis=1)(layer)
        elif normalization == 2:
            # TODO: complex
            try:
                from keras_contrib.layers.normalization import InstanceNormalization
            except ImportError:
                raise ImportError("Install keras_contrib in order to use instance normalization."
                                  "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
            layer = InstanceNormalization(axis=1)(layer)

        # activation
        if self.is_complex:
            input_magreal = sliceLayerReal(layer)
            input_phaimag = sliceLayerImag(layer)
            layer_magreal = Activation(activation)(input_magreal)
            layer_phaimag = Activation(activation)(input_phaimag)
            layer = stackLayer([expandLayer(layer_magreal), expandLayer(layer_phaimag)])
        else:
            layer = Activation(activation)(layer)

        return layer

    def create_up_convolution(self, input_layer, n_filters, stage, level, filter_bank=0):
        if self.deconvolution:
            # prepare some layers
            expandLayer = Lambda(lambda x: K.expand_dims(x, axis=1))
            stackLayer = Lambda(lambda x: K.concatenate(x, axis=1))
            layer_mag_pha = Lambda(lambda x: to_mag_pha(x, self.n_dim))
            layer_real_imag = Lambda(lambda x: to_real_imag(x))
            if self.n_dim == 2:
                sliceLayerReal = Lambda(lambda x: x[:, 0, :, :])
                sliceLayerImag = Lambda(lambda x: x[:, 1, :, :])
            elif self.n_dim == 3:
                sliceLayerReal = Lambda(lambda x: x[:, 0, :, :, :])
                sliceLayerImag = Lambda(lambda x: x[:, 1, :, :, :])

            if self.n_dim == 2:
                if self.is_complex:
                    if self.complex_conv:
                        input_real = sliceLayerReal(input_layer)
                        input_imag = sliceLayerImag(input_layer)
                        conv_real_spatial = Conv2DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                                   data_format=self.data_format, input_shape=self.input_shape)
                        conv_imag_spatial = Conv2DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                                   data_format=self.data_format)
                        layer_real = Subtract()(
                            [conv_real_spatial(input_real), conv_imag_spatial(input_imag)])
                        layer_imag = Add()(
                            [conv_imag_spatial(input_real), conv_real_spatial(input_imag)])
                        layer_real = expandLayer(layer_real)
                        layer_imag = expandLayer(layer_imag)
                        layer = stackLayer([layer_real, layer_imag])
                    else:
                        input_real = sliceLayerReal(input_layer)
                        layer_real = Conv2DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                            data_format=self.data_format, input_shape=self.input_shape)(input_real)
                        layer_real = expandLayer(layer_real)
                        input_imag = sliceLayerImag(input_layer)
                        layer_imag = Conv2DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                            data_format=self.data_format)(input_imag)
                        layer_imag = expandLayer(layer_imag)
                        layer = stackLayer([layer_real, layer_imag])
                else:
                    layer = Conv2DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                   input_shape=self.input_shape)(input_layer)
            elif self.n_dim == 3:
                if self.is_complex:
                    if self.complex_conv:
                        input_real = sliceLayerReal(input_layer)
                        input_imag = sliceLayerImag(input_layer)
                        conv_real_spatial = Conv3DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                                   data_format=self.data_format, input_shape=self.input_shape)
                        conv_imag_spatial = Conv3DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                                   data_format=self.data_format)
                        layer_real = Subtract()(
                            [conv_real_spatial(input_real), conv_imag_spatial(input_imag)])
                        layer_imag = Add()(
                            [conv_imag_spatial(input_real), conv_real_spatial(input_imag)])
                        layer_real = expandLayer(layer_real)
                        layer_imag = expandLayer(layer_imag)
                        layer = stackLayer([layer_real, layer_imag])
                    else:  # real-valued conv (of complex data)
                        input_real = sliceLayerReal(input_layer)
                        layer_real = Conv3DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                            data_format=self.data_format, input_shape=self.input_shape)(input_real)
                        layer_real = expandLayer(layer_real)
                        input_imag = sliceLayerImag(input_layer)
                        layer_imag = Conv3DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                            data_format=self.data_format)(input_imag)
                        layer_imag = expandLayer(layer_imag)
                        layer = stackLayer([layer_real, layer_imag])
                else:
                    layer = Conv3DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                   input_shape=self.input_shape)(input_layer)

            elif self.n_dim == 4:
                if self.is_complex:
                    if self.complex_conv:
                        input_real = sliceLayerReal(input_layer)
                        input_imag = sliceLayerImag(input_layer)
                        conv_real_spatial = Conv4DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                                   data_format=self.data_format, input_shape=self.input_shape)
                        conv_imag_spatial = Conv4DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                                   data_format=self.data_format)
                        layer_real = Subtract()(
                            [conv_real_spatial(input_real), conv_imag_spatial(input_imag)])
                        layer_imag = Add()(
                            [conv_imag_spatial(input_real), conv_real_spatial(input_imag)])
                        layer_real = expandLayer(layer_real)
                        layer_imag = expandLayer(layer_imag)
                        layer = stackLayer([layer_real, layer_imag])
                    else:  # real-valued conv (of complex data)
                        input_real = sliceLayerReal(input_layer)
                        layer_real = Conv4DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                            data_format=self.data_format, input_shape=self.input_shape)(input_real)
                        layer_real = expandLayer(layer_real)
                        input_imag = sliceLayerImag(input_layer)
                        layer_imag = Conv4DTranspose(n_filters, kernel, padding=self.padding, strides=self.strides,
                                            data_format=self.data_format)(input_imag)
                        layer_imag = expandLayer(layer_imag)
                        layer = stackLayer([layer_real, layer_imag])
                else:
                    layer = Conv4DTranspose(input_layer, n_filters, kernel, padding=self.padding, strides=self.strides,
                                   stage=stage, level=level, filter_bank=filter_bank)
        else:
            if self.is_complex:
                # TODO
            else:
                if self.n_dim == 2:
                    layer = UpSampling2D(size=self.pool_size)(input_layer)
                elif self.n_dim == 3:
                    layer = UpSampling3D(size=self.pool_size)(input_layer)
                elif self.n_dim == 4:
                    layer = UpSampling4D(input_layer, stage, level, pool_size=self.pool_size)

        return layer

    def create_down_sampling(self, input_layer, stage, level, pool_size):
        if self.n_dim == 2:
            if self.is_complex:
                current_layer = ComplexMaxPooling2D(pool_size=pool_size, data_format=self.data_format)(input_layer)
            else:
                current_layer = MaxPooling2D(pool_size=pool_size, data_format=self.data_format)(input_layer)
        elif self.n_dim == 3:
            if self.is_complex:
                current_layer = ComplexMaxPooling3D(pool_size=pool_size, data_format=self.data_format)(input_layer)
            else:
                current_layer = MaxPooling3D(pool_size=pool_size, data_format=self.data_format)(input_layer)
        elif self.n_dim == 4:
            if self.is_complex:
                current_layer = ComplexMaxPooling4D(pool_size=pool_size, data_format=self.data_format)(input_layer)
            else:
                current_layer = MaxPooling4D(pool_size=pool_size, data_format=self.data_format)(input_layer)

        return current_layer


############################
# 2D UNET simple
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def unet_2D(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model