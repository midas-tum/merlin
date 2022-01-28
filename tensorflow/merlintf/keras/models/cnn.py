import tensorflow as tf
import merlintf

class ComplexCNN(tf.keras.Model):
    def __init__(self, dim='2D', filters=64, kernel_size=3, num_layer=5,
                       activation='ModReLU', use_bias=True,
                       name='ComplexCNN', **kwargs):
        super().__init__(name=name)
        # get correct conv operator
        if dim == '2D':
            conv_layer = merlintf.keras.layers.ComplexConv2D
        elif dim == '3D':
            conv_layer = merlintf.keras.layers.ComplexConv3D
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        # create layers
        self.ops = []
        for _ in range(num_layer-1):
            self.ops.append(conv_layer(filters, kernel_size,
                                        use_bias=use_bias,
                                        activation=activation,
                                        padding='same', **kwargs
                                        ))
        self.ops.append(conv_layer(1, kernel_size,
                                    use_bias=False,
                                    padding='same',
                                    activation=None, **kwargs))

    def call(self, inputs):
        x = inputs
        for op in self.ops:
            x = op(x)
        return x

class Real2chCNN(tf.keras.Model):
    def __init__(self, dim='2D', filters=64, kernel_size=3, num_layer=5,
                       activation='relu', use_bias=True,
                       name='Real2chCNN', **kwargs):
        super().__init__(name=name)
        # get correct conv operator
        if dim == '2D':
            conv_layer = tf.keras.layers.Conv2D
        elif dim == '3D':
            conv_layer = tf.keras.layers.Conv3D
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        kernel_initializer = kwargs.pop('kernel_initializer', 'glorot_uniform')

        # create layers
        self.ops = []
        for _ in range(num_layer-1):
            self.ops.append(conv_layer(filters, kernel_size,
                                        use_bias=use_bias,
                                        activation=activation,
                                        kernel_initializer=kernel_initializer,
                                        padding='same', **kwargs))

        self.ops.append(conv_layer(2, kernel_size,
                                    use_bias=False,
                                    padding='same',
                                    activation=None,
                                    kernel_initializer=kernel_initializer, **kwargs))

    def call(self, inputs):
        x = merlintf.complex2real(inputs)
        for op in self.ops:
            x = op(x)
        return merlintf.real2complex(x)

class Real2chCNNTest(unittest.TestCase):
    def test_cnn_real2ch_2d(self):
        self._test_cnn_real2ch('2D', 3)

    def test_cnn_real2ch_2d_2(self):
        self._test_cnn_real2ch('2D', (3,5))

    def test_cnn_real2ch_3d(self):
        self._test_cnn_real2ch('3D', (3, 5, 5))

    def _test_cnn_real2ch(self, dim, kernel_size):
        nBatch = 5
        D = 20
        M = 128
        N = 128

        config = {
            'dim': dim,
            'filters': 64,
            'kernel_size': kernel_size,
            'num_layer': 5,
            'activation': 'relu'
        }

        model = Real2chCNN(**config)

        if dim == '2D':
            x = merlintf.random_normal_complex((nBatch, M, N, 1))
        elif dim == '3D' or dim == '2Dt':
            x = merlintf.random_normal_complex((nBatch, D, M, N, 1))
        else:
            raise RuntimeError(f'No implementation for dim {dim} available!')
        
        Kx = model(x)
        self.assertTrue(Kx.shape == x.shape)
s