import torch
import torch.nn as nn
import torch.nn.functional as F
from merlinth.layers.complex_act import ModReLU

class UNet(nn.Module):
    def __init__(self, dim='2D', filters=64, kernel_size=3, pool_size=2, num_layer_per_level=2, num_level=4,
                       activation='relu', activation_last='relu', kernel_size_last=1, use_bias=True,
                       normalization='none', downsampling='mp', upsampling='tc',
                       name='UNet', n_channels=[2, 2], **kwargs):
        """
        Class for 2D UNet model
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
        n_channels                  [list] input and output channel of tensors, default: [2, 2]
        """
        super(UNet, self).__init__()
        if dim.upper() != '2D':
            raise ValueError(f'Dimension {dim} not supported!')

        self.name = name
        self.dim = dim
        self.num_level = num_level
        self.num_layer_per_level = num_layer_per_level
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.use_bias = use_bias
        self.activation = activation
        self.activation_last = activation_last
        self.kernel_size_last = kernel_size_last
        self.normalization = normalization
        self.downsampling = downsampling
        self.upsampling = upsampling

        # get activation operator
        activations = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'linear': nn.Linear,
            'modrelu': ModReLU,
            'none': None
        }
        activations_arg = {
            'relu': {'inplace': True},
            'leakyrelu': {'negative_slope': 0.1, 'inplace': True},
            'sigmoid': {'inplace': True},
            'tanh': {'inplace': True},
            'linear': {'in_features': 2, 'out_features': 2, 'bias': True},
            'modrelu': {'num_parameters': 32, 'bias_init': 0.1, 'requires_grad': True}
        }

        self.activation_layer = activations[activation.lower()]
        self.activation_layer_args = activations_arg[activation.lower()]

        self.activation_layer_last = activations[activation_last.lower()]
        self.activation_layer_args_last = activations_arg[activation_last.lower()]

        # get normalization operator
        normalizations = {
            'bn': nn.BatchNorm2d,
            'in': nn.InstanceNorm2d,
            'none': None
        }
        self.norm_layer = normalizations[normalization.lower()]

        # get downsampling operator
        n_dim = 2  # TODO: 2D hard-coded atm
        if downsampling == 'mp':
            self.down_layer = nn.MaxPool2d
            self.strides = [1] * num_layer_per_level
        elif downsampling == 'st':
            self.down_layer = None
            self.strides = [[1] * n_dim] * (num_layer_per_level - 1) + [list(self.pool_size)]
        else:
            raise RuntimeError(f"Downsampling operation {downsampling} not implemented!")

        # get upsampling operator
        if upsampling == 'us':
            self.us = True
        elif upsampling == 'tc':
            self.us = False
        else:
            raise RuntimeError(f"Upsampling operation {upsampling} not implemented!")

        in_channel = n_channels[0]
        self.ops = []
        self.order = []

        # encoder
        stage = []
        for ilevel in range(self.num_level):
            level = []
            level.append('c')
            self.ops.append(ConvStage(in_channel, self.filters * (2 ** ilevel), self.kernel_size,
                                   stride=self.strides,
                                   padding='same',
                                   num_layer_per_level=self.num_layer_per_level,
                                   norm_layer=self.norm_layer,
                                   activation_layer=self.activation_layer,
                                   activation_layer_args=self.activation_layer_args))
            in_channel = int(self.filters * (2 ** ilevel))
            level.append('d')
            self.ops.append(Down(self.pool_size, self.down_layer))
            stage.append(level)
        self.order.append(stage)

        # bottleneck
        stage = []
        stage.append('c')
        self.ops.append(ConvStage(in_channel, self.filters * (2 ** (self.num_level)), self.kernel_size,
                                   stride=self.strides,
                                   padding='same',
                                   num_layer_per_level=self.num_layer_per_level,
                                   norm_layer=self.norm_layer,
                                   activation_layer=self.activation_layer,
                                   activation_layer_args=self.activation_layer_args))
        in_channel = int(self.filters * (2 ** self.num_level))

        stage.append('u')
        self.ops.append(Up(in_channel, kernel_size=self.kernel_size, stride=self.pool_size, padding='same', bilinear=self.upsampling=='us'))
        #if self.upsampling == 'tc':
        #    in_channel = int(in_channel / 2)
        self.order.append(stage)

        # decoder
        stage = []
        for ilevel in range(self.num_level - 1, -1, -1):
            level = []
            if self.upsampling == 'us':
                level.append('c')
                self.ops.append(ConvStage(in_channel, self.filters * (2 ** ilevel) // 2, self.kernel_size,
                                   stride=self.strides,
                                   padding='same',
                                   num_layer_per_level=self.num_layer_per_level,
                                   norm_layer=self.norm_layer,
                                   activation_layer=self.activation_layer,
                                   activation_layer_args=self.activation_layer_args))
                in_channel = int(self.filters * (2 ** ilevel) // 2)
            else:
                level.append('c')
                self.ops.append(ConvStage(in_channel, self.filters * (2 ** ilevel), self.kernel_size,
                                   stride=self.strides,
                                   padding='same',
                                   num_layer_per_level=self.num_layer_per_level,
                                   norm_layer=self.norm_layer,
                                   activation_layer=self.activation_layer,
                                   activation_layer_args=self.activation_layer_args))
                in_channel = int(self.filters * (2 ** ilevel))
            if ilevel > 0:
                level.append('u')
                self.ops.append(Up(in_channel, kernel_size=self.kernel_size, stride=self.pool_size, padding='same', bilinear=self.upsampling=='us'))
            stage.append(level)
        self.order.append(stage)

        # output convolution
        self.order.append('c')
        self.ops.append(ConvStage(in_channel, n_channels[1], self.kernel_size_last,
                                   stride=1,
                                   padding='same',
                                   num_layer_per_level=1,
                                   norm_layer=self.norm_layer,
                                   activation_layer=self.activation_layer_last,
                                   activation_layer_args=self.activation_layer_args_last))
        self.ops = nn.ModuleList(self.ops)

    def forward(self, x):
        xforward = []
        iteridx = iter(self.ops)
        # encoder
        for ilevel in range(self.num_level):
            for iop, _ in enumerate(self.order[0][ilevel]):
                if iop == len(self.order[0][ilevel]) - 1:
                    xforward.append(x)
                op = next(iteridx)
                #if op is not None:
                x = op(x)

        # bottleneck
        for iop, _ in enumerate(self.order[1]):
            op = next(iteridx)
            #if op is not None:
            if iop == len(self.order[1]) - 1:
                x = op(x, xforward[-1])
            else:
                x = op(x)

        # decoder
        for ilevel in range(self.num_level - 1, -1, -1):
            for iop, _ in enumerate(self.order[2][self.num_level - 1 - ilevel]):
                op = next(iteridx)
                #if op is not None:
                if ilevel > 0 and iop == len(self.order[2][self.num_level - 1 - ilevel]) - 1:
                    x = op(x, xforward[ilevel-1])
                else:
                    x = op(x)

        # output convolution
        op = next(iteridx)
        x = op(x)

        return x


class ConvStage(nn.Module):
    """(convolution => normalization => activation)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', num_layer_per_level=2, norm_layer=None, activation_layer=None, activation_layer_args=None):
        super().__init__()

        self.level = []
        for ilayer in range(num_layer_per_level):
            if isinstance(stride, list):
                stridecurr = stride[ilayer]
            else:
                stridecurr = stride
            self.level.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stridecurr, padding=padding))
            in_channels = out_channels
            if norm_layer is not None:
                self.level.append(norm_layer(out_channels))
            if activation_layer is not None:
                if "in_features" in activation_layer_args:
                    activation_layer_args["in_features"] = in_channels
                    activation_layer_args["out_features"] = out_channels
                self.level.append(activation_layer(**activation_layer_args))
        self.level = nn.ModuleList(self.level)

    def forward(self, x):
        for op in self.level:
            x = op(x)
        return x


class Down(nn.Module):
    """Downscaling"""

    def __init__(self, pool_size=2, down_layer=None):
        super().__init__()
        self.maxpool_conv = down_layer
        if self.maxpool_conv is not None:
            self.maxpool_conv = down_layer(pool_size)

    def forward(self, x):
        if self.maxpool_conv is None:
            return x
        else:
            return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, kernel_size=3, stride=2, padding='same', bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            #self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride)
            #self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return torch.cat([x2, x1], dim=1)

'''
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
'''