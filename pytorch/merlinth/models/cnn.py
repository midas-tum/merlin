import torch
import merlinth
from merlinth.layers import ComplexConv2d, ComplexConv3d
from merlinth.layers.complex_act import cReLU

__all__ = ['Real2chCNN', 'ComplexCNN']

class Real2chCNN(torch.nn.Module):
    def __init__(
        self,
        dim="2D",
        input_dim=1,
        filters=64,
        kernel_size=3,
        num_layer=5,
        activation="relu",
        use_bias=True,
        normalization=None,
        **kwargs,
    ):
        super().__init__()
        # get correct conv operator
        if dim == "2D":
            conv_layer = torch.nn.Conv2d
        elif dim == "3D":
            conv_layer = torch.nn.Conv3d
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        if activation == "relu":
            act_layer = torch.nn.ReLU

        padding = kernel_size // 2
        # create layers
        self.ops = []
        self.ops.append(
            conv_layer(
                input_dim * 2,
                filters,
                kernel_size,
                padding=padding,
                bias=use_bias,
                **kwargs,
            )
        )
        if normalization is not None:
            self.ops.append(normalization(num_features=filters, affine=True))
        self.ops.append(act_layer(inplace=True))

        for _ in range(num_layer - 2):
            self.ops.append(
                conv_layer(
                    filters,
                    filters,
                    kernel_size,
                    padding=padding,
                    bias=use_bias,
                    **kwargs,
                )
            )
            if normalization is not None:
                self.ops.append(normalization(num_features=filters, affine=True))
            self.ops.append(act_layer(inplace=True))

        self.ops.append(
            conv_layer(
                filters,
                input_dim * 2,
                kernel_size,
                bias=False,
                padding=padding,
                **kwargs,
            )
        )
        self.ops = torch.nn.Sequential(*self.ops)
        self.apply(self.weight_initializer)

    def weight_initializer(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # equivalent to tf.layers.xavier_initalization()
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, inputs):
        if merlinth.iscomplex(inputs):
            x = merlinth.complex2real(inputs)
        else:
            x = inputs
        x = self.ops(x)
        if merlinth.iscomplex(inputs):
            return merlinth.real2complex(x)
        else:
            return x  # data already in real2channel format

class ComplexCNN(merlinth.layers.module.ComplexModule):
    def __init__(
        self,
        dim="2D",
        input_dim=1,
        filters=64,
        kernel_size=3,
        num_layer=5,
        activation="relu",
        use_bias=True,
        normalization=None,
        weight_std=False,
        **kwargs,
    ):
        super().__init__()
        # get correct conv operator
        if dim == "2D":
            conv_layer = ComplexConv2d
        elif dim == "3D":
            conv_layer = ComplexConv3d
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        if activation == "relu":
            act_layer = cReLU

        padding = kernel_size // 2
        # create layers
        self.ops = []
        self.ops.append(
            conv_layer(
                input_dim,
                filters,
                kernel_size,
                padding=padding,
                bias=use_bias,
                weight_std=weight_std,
                **kwargs,
            )
        )
        if normalization is not None:
            self.ops.append(normalization())

        self.ops.append(act_layer())

        for _ in range(num_layer - 2):
            self.ops.append(
                conv_layer(
                    filters,
                    filters,
                    kernel_size,
                    padding=padding,
                    bias=use_bias,
                    **kwargs,
                )
            )
            if normalization is not None:
                self.ops.append(normalization())
            self.ops.append(act_layer())

        self.ops.append(
            conv_layer(
                filters,
                input_dim,
                kernel_size,
                bias=False,
                padding=padding,
                **kwargs,
            )
        )
        self.ops = torch.nn.Sequential(*self.ops)
        self.apply(self.weight_initializer)

    def weight_initializer(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # equivalent to tf.layers.xavier_initalization()
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, x):
        x = self.ops(x)
        return x
