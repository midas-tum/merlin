import torch

__all__ = ['ComplexL2Loss', 'ComplexL2Loss_fun']

class ComplexL2Loss_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return torch.sum(z**2)

    @staticmethod
    def backward(ctx, grad_out):
        z = ctx.saved_tensors[0]
        return  2 * z


class ComplexL2Loss(torch.nn.Module):
    def __init__(self):
        super(ComplexL2Loss, self).__init__()

    def forward(self, z):
        assert z.shape[-1] == 2
        return ComplexL2Loss_fun.apply(z)

