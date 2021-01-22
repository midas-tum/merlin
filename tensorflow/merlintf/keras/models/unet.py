import tensorflow as tf
import merlintf

import unittest
import numpy as np

class ComplexUNet(tf.keras.Model):
    def __init__(self, dim='2D', nf=64, ksz=3, num_layer_per_level=2, num_level=4,
                       activation='ModReLU', use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='ComplexUNet', **kwargs):
        """
        Builds the 2D/2D+t/3D/3D+t/4D UNet model
        input parameter:
        dim                         [string] operating dimension
        nf                          [integer, tuple] number of filters at the base level, dyadic increase
        ksz                         [integer, tuple] kernel size
        num_layer_per_level         number of convolutional layers per encocer/decoder level
        num_level                   amount of encoder/decoder stages (excluding bottleneck layer), network depth
        activation                  activation function
        use_bias                    apply bias for convolutional layer
        normalization               use normalization layers: BN (batch), IN (instance), none
        downsampling                downsampling operation: mp (max-pooling), st (stride)
        upsampling                  upsampling operation: us (upsampling), tc (transposed convolution)
        """
        super().__init__(name=name)

        # get correct conv operator
        conv_layer = merlintf.keras.layers.ComplexConvolution(dim)

        # get normalitzation operator
        activation_last = activation
        if normalization == 'BN':
            norm_layer = merlintf.keras.layers.ComplexBatchNormalization
            activation_layer = merlintf.keras.layers.Activation(activation)
            activation = ''
        elif normalization == 'IN':
            norm_layer = merlintf.keras.layers.ComplexInstanceNormalization
            activation_layer = merlintf.keras.layers.Activation(activation)
            activation = ''
        elif normalization == 'none':
            norm_layer = None
            activation_layer = None
        else:
            raise RuntimeError(f"Normalization for {normalization} not implemented!")

        # get downsampling operator
        if downsampling == 'mp':
            down_layer = merlintf.keras.layers.MagnitudeMaxPooling(dim)
            strides = [1] * num_layer_per_level
        elif downsampling == 'st':
            down_layer = None
            strides = [1] * (num_layer_per_level-1) + [2]
        else:
            raise RuntimeError(f"Downsampling operation {downsampling} not implemented!")

        # get upsampling operator
        if upsampling == 'us':
            up_layer = merlintf.keras.layers.UpSampling(dim)  # TODO check if working for complex
        elif upsampling == 'tc':
            up_layer = merlintf.keras.layers.ComplexConvolutionTranspose(dim)
        else:
            raise RuntimeError(f"Upsampling operation {upsampling} not implemented!")

        self.num_level = num_level
        self.num_layer_per_level = num_layer_per_level
        # create layers
        self.ops = []
        # encoder
        stage = []
        for ilevel in range(num_level):
            level = []
            for ilayer in range(num_layer_per_level):
                level.append(conv_layer(nf * (2**ilevel), ksz,
                                           stride=strides[ilayer],
                                           use_bias=use_bias,
                                           activation=activation,
                                           padding='same'))
                level.append(norm_layer)
                level.append(activation_layer)

            level.append(down_layer)
            stage.ops.append(level)
        self.ops.append(stage)

        # bottleneck
        stage = []
        for ilayer in range(num_layer_per_level):
            stage.append(conv_layer(nf * (2 ** (num_level+1)), ksz,
                                       stride=strides[ilayer],
                                       use_bias=use_bias,
                                       activation=activation,
                                       padding='same'))
            stage.append(norm_layer)
            stage.append(activation_layer)
        stage.append(up_layer)
        self.ops.append(stage)

        # decoder
        stage = []
        for ilevel in range(num_level, -1, -1):
            level = []
            for ilayer in range(num_layer_per_level):
                level.append(conv_layer(nf * (2 ** ilevel), ksz,
                                           stride=1,
                                           use_bias=use_bias,
                                           activation=activation,
                                           padding='same'))
                level.append(norm_layer)
                level.append(activation_layer)

            if ilevel > 0:
                level.append(up_layer)
            stage.append(level)
        self.ops.append(stage)

        # output convolution
        self.ops.append(conv_layer(1, 1, stride=1,
                                           use_bias=use_bias,
                                           activation=activation_last,
                                           padding='same')))

        def call(self, inputs):
            x = inputs
            xforward = []
            # encoder
            for ilevel in range(self.num_level):
                for iop, op in enumerate(self.ops[0][ilevel]):
                    if op is not None:
                        x = op(x)
                    if iop == len(self.ops[0][ilevel])-1:
                        xforward.append(x)

            # bottleneck
            for op in self.ops[1]
                if op is not None:
                    x = op(x)

            # decoder
            for ilevel in range(self.num_level, -1, -1):
                x = tf.keras.layer.concatenate([x, xforward[ilevel]])
                for op in self.ops[2][self.num_level-ilevel]:
                    if op is not None:
                        x = op(x)

            # output convolution
            x = self.ops[3](x)
            return x


class ComplexUNetTest(unittest.TestCase):
    def test_UNet_complex_2d(self):
        self._test_UNet_complex('2D', 64, (3, 3))

    def test_FoE_real_3d(self):
        self._test_UNet_complex('3D', 32, (3, 5, 5))

    def _test_UNet_complex(self, dim, nf, ksz):
        nBatch = 5
        D = 20
        M = 128
        N = 128
        nf_in = 10
        nw = 31

        model = ComplexUNet(dim, nf, ksz)

        if dim == '2D':
            x = tf.random.normal((nBatch, M, N, 1), dtype=tf.float32)
        elif dim == '3D' or dim == '2Dt':
            x = tf.random.normal((nBatch, D, M, N, 1), dtype=tf.float32)
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')

        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)


if __name__ == "__main__":
    unittest.test()