import tensorflow as tf
import merlintf
  
class ComplexCNN(tf.keras.Model):
    def __init__(self, dim=2, nf=64, ksz=3, num_layer=5, 
                       activation='ModReLU', use_bias=True,
                       name='ComplexCNN', **kwargs):
        super().__init__(name=name)
        # get correct conv operator
        if dim == 2:
            conv_layer = merlintf.keras.layers.ComplexConv2D
        elif dim == 3:
            conv_layer = merlintf.keras.layers.ComplexConv3D
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        # create layers
        self.ops = []
        for _ in range(num_layer-1):
            self.ops.append(conv_layer(nf, ksz,
                                        use_bias=use_bias,
                                        activation=activation,
                                        padding='same',
                                        ))
        self.ops.append(conv_layer(1, ksz,
                                    use_bias=False,
                                    padding='same',
                                    activation=None))

    def call(self, inputs):
        x = inputs
        for op in self.ops:
            x = op(x)
        return x

class Real2chCNN(tf.keras.Model):
    def __init__(self, dim=2, nf=64, ksz=3, num_layer=5, 
                       activation='relu', use_bias=True,
                       name='Real2chCNN', **kwargs):
        super().__init__(name=name)
        # get correct conv operator
        if dim == 2:
            conv_layer = tf.keras.layers.Conv2D
        elif dim == 3:
            conv_layer = tf.keras.layers.Conv3D
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        # create layers
        self.ops = []
        for _ in range(num_layer):
            self.ops.append(conv_layer(nf, ksz,
                                        use_bias=use_bias,
                                        activation=activation,
                                        padding='same',))

        self.ops.append(conv_layer(2, ksz,
                                    use_bias=False,
                                    padding='same',
                                    activation=None))

    def call(self, inputs):
        x = merlintf.complex2real(inputs)
        for op in self.ops:
            x = op(x)
        return merlintf.real2complex(x)