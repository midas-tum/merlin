import torch
import merlinth
import six
import sys

try:
    import optoth.pad
except:
    print('optoth could not be imported')


def get(identifier):
    return Padding(identifier)


def Padding(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'Padding' + (str(identifier).upper() if len(identifier) == 2 else str(identifier[0:2]).upper() + str(identifier[-1]))
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret padding function identifier: {}'.format(identifier))

def PaddingTranspose(identifier):
    if isinstance(identifier, six.string_types):
        identifier = 'Padding' + (str(identifier).upper() if len(identifier) == 2 else str(identifier[0:2]).upper() + str(identifier[-1]))
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError('Could not interpret padding function identifier: {}'.format(identifier))

def deserialize(op):
    if op == 'Padding1D':
        return Padding1D
    elif op == 'Padding2D':
        return Padding2D
    elif op == 'Padding3D':
        return Padding3D
    elif op == 'Padding4D':
        return Padding4D
    elif op == 'Padding1DTranspose':
        return Padding1DTranspose
    elif op == 'Padding2DTranspose':
        return Padding2DTranspose
    elif op == 'Padding3DTranspose':
        return Padding3DTranspose
    elif op == 'Padding4DTranspose':
        return Padding4DTranspose
    else:
        raise ValueError('Unknown padding operation: {}'.format(op))

def flatten(object):
    for item in object:
        if isinstance(item, (list, tuple, set)):
            yield from flatten(item)
        else:
            yield item

class _Padding(torch.nn.Module):
    def __init__(self, pad, mode, value, optox, rank, transpose):
        super(_Padding, self).__init__()
        self.mode = mode
        self.optox = optox and (True if 'optoth.pad' in sys.modules else False)
        self.value = value
        self.pad = pad
        self.rank = rank
        self.transpose = transpose

    def forward(self, x):
        if self.optox:
            if self.transpose:
                if self.rank == 1:
                    return optoth.pad.pad1d_transpose(x, flatten(self.pad)[::-1], self.mode, self.value)
                elif self.rank == 2:
                    return optoth.pad.pad2d_transpose(x, flatten(self.pad)[::-1], self.mode, self.value)
                elif self.rank == 3:
                    return optoth.pad.pad3d_transpose(x, flatten(self.pad)[::-1], self.mode, self.value)
                elif self.rank == 4:
                    axis = 2  # 'channels_first', default Pytorch
                    # xyz padding
                    shape_in = inputs.shape
                    x_list = torch.split(inputs, shape_in[axis], axis=axis)
                    x = torch.concat(x_list, axis=0)
                    x = torch.squeeze(x, axis=axis)
                    x = optoth.pad.pad3d_transpose(x, flatten(self.pad)[::-1][0:3], mode=self.mode)
                    x_list = torch.split(x, shape_in[axis], axis=0)
                    x = torch.stack(x_list, axis=axis)
                    # t padding
                    axis += 1
                    shape_in = x.shape
                    x_list = torch.split(x, shape_in[axis], axis=axis)
                    x = torch.concat(x_list, axis=0)
                    x = torch.squeeze(x, axis=axis)
                    x = optoth.pad.pad3d_transpose(x, [flatten(self.pad)[::-1][-1], 0, 0, 0, 0], mode=self.mode)
                    x_list = torch.split(x, shape_in[axis], axis=0)
                    return torch.stack(x_list, axis=axis)
            else:
                if self.rank == 1:
                    return optoth.pad.pad1d(x, flatten(self.pad)[::-1], self.mode, self.value)
                elif self.rank == 2:
                    return optoth.pad.pad2d(x, flatten(self.pad)[::-1], self.mode, self.value)
                elif self.rank == 3:
                    return optoth.pad.pad3d(x, flatten(self.pad)[::-1], self.mode, self.value)
                elif self.rank == 4:
                    axis = 2  # 'channels_first', default Pytorch
                    # xyz padding
                    shape_in = inputs.shape
                    x_list = torch.split(inputs, shape_in[axis], axis=axis)
                    x = torch.concat(x_list, axis=0)
                    x = torch.squeeze(x, axis=axis)
                    x = optoth.pad.pad3d(x, flatten(self.pad)[::-1][0:3], mode=self.mode)
                    x_list = torch.split(x, shape_in[axis], axis=0)
                    x = torch.stack(x_list, axis=axis)
                    # t padding
                    axis += 1
                    shape_in = x.shape
                    x_list = torch.split(x, shape_in[axis], axis=axis)
                    x = torch.concat(x_list, axis=0)
                    x = torch.squeeze(x, axis=axis)
                    x = optoth.pad.pad3d(x, [flatten(self.pad)[::-1][-1], 0, 0, 0, 0], mode=self.mode)
                    x_list = torch.split(x, shape_in[axis], axis=0)
                    return torch.stack(x_list, axis=axis)
        else:
            return torch.nn.functional.pad(x, self.pad, self.mode, self.value)

class Padding1D(_Padding):
    def __init__(self, pad=1, mode='symmetric', value=0, optox=False):
        super(Padding1D, self).__init__(pad, mode, value, optox, 1, False)

class Padding2D(_Padding):
    def __init__(self, pad=(1, 1), mode='symmetric', value=0, optox=False):
        super(Padding2D, self).__init__(pad, mode, value, optox, 2, False)

class Padding3D(_Padding):
    def __init__(self, pad=(1, 1, 1), mode='symmetric', value=0, optox=False):
        super(Padding3D, self).__init__(pad, mode, value, optox, 3, False)

class Padding4D(_Padding):
    def __init__(self, pad=(1, 1, 1, 1), mode='symmetric', value=0, optox=False):
        super(Padding4D, self).__init__(pad, mode, value, optox, 4, False)

class Padding1DTranspose(_Padding):
    def __init__(self, pad=1, mode='symmetric', value=0, optox=False):
        super(Padding1DTranspose, self).__init__(pad, mode, value, optox, 1, True)

class Padding2DTranspose(_Padding):
    def __init__(self, pad=(1, 1), mode='symmetric', value=0, optox=False):
        super(Padding2DTranspose, self).__init__(pad, mode, value, optox, 2, True)

class Padding3DTranspose(_Padding):
    def __init__(self, pad=(1, 1, 1), mode='symmetric', value=0, optox=False):
        super(Padding3DTranspose, self).__init__(pad, mode, value, optox, 3, True)

class Padding4DTranspose(_Padding):
    def __init__(self, pad=(1, 1, 1, 1), mode='symmetric', value=0, optox=False):
        super(Padding4DTranspose, self).__init__(pad, mode, value, optox, 4, True)

# Aliases
Padding2Dt = Padding3D
Padding3Dt = Padding4D
Padding2DtTranspose = Padding3DTranspose
Padding3DtTranspose = Padding4DTranspose