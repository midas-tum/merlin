import torch
from merlinth.layers.complex_cg import CGClass


class DCGD(torch.nn.Module):
    def __init__(self, A, AH, weight_init=1.0, weight_scale=1.0, requires_grad=True, **kwargs):
        super().__init__()

        self.A = A
        self.AH = AH

        self.weight_scale = weight_scale
        self.weight_init = weight_init
        self._weight = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)*weight_init)
        self._weight.requires_grad_(requires_grad)
        self._weight.proj = lambda: self._weight.data.clamp_(1e-4, 1000)

    @property
    def weight(self):
        return self._weight * self.weight_scale

    def forward(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        stepsize = self.weight * scale
        return x - stepsize * self.AH(self.A(x, *constants) - y, *constants)

    def __repr__(self):
        return f'DCGD(lambda_init={self.weight_init:.4g}, weight_scale={self.weight_scale}, requires_grad={self._weight.requires_grad})'

class DCPM(torch.nn.Module):
    def __init__(self, A, AH, weight_init=1.0, weight_scale=1.0, requires_grad=True, **kwargs):
        super().__init__()

        self.A = A
        self.AH = AH

        self.weight_scale = weight_scale
        self.weight_init = weight_init
        self._weight = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)*weight_init)
        self._weight.requires_grad_(requires_grad)
        self._weight.proj = lambda: self._weight.data.clamp_(1e-4, 1000)

        max_iter = kwargs.get('max_iter', 10)
        tol = kwargs.get('tol', 1e-10)
        self.prox = CGClass(A, AH, max_iter=max_iter, tol=tol)

    @property
    def weight(self):
        return self._weight * self.weight_scale

    def forward(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        lambdaa = 1.0 / torch.max(self.weight * scale, torch.ones_like(self.weight)*1e-9)
        return self.prox(lambdaa, x, y, *constants)

    def __repr__(self):
        return f'DCPD(lambda_init={self.weight_init:.4g}, weight_scale={self.weight_scale}, requires_grad={self._weight.requires_grad})'

class itSENSE(torch.nn.Module):
    def __init__(self, A, AH, weight=0.0, **kwargs):
        super().__init__()

        self.A = A
        self.AH = AH

        self.weight = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)*weight)
        self.weight.requires_grad_(False)

        self.max_iter = kwargs.get('max_iter', 10)
        self.tol = kwargs.get('tol', 1e-10)
        self.op = CGClass(A, AH, max_iter=self.max_iter, tol=self.tol)

    def forward(self, inputs):
        y = inputs[0]
        constants = inputs[1:]
        x = torch.zeros_like(self.AH(y, *constants))
        return self.op(self.weight, x, y, *constants)

    def __repr__(self):
        return f'itSENSE(lambda={self.weight:.4g}, max_iter={self.max_iter}, tol={self.tol})'
