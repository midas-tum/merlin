
import torch
import merlinth.mytorch as mytorch

class ComplexCG(torch.autograd.Function):
    @staticmethod
    def complexDot(data1, data2):
        mult = mytorch.complex.complex_mult_conj(data1, data2)
        re, im = torch.unbind(mult, dim=-1)
        return torch.stack([torch.sum(re),
                            torch.sum(im)], -1)

    @staticmethod
    def solve(x0, M, tol, max_iter):
        x = torch.zeros_like(x0)
        r = x0.clone()
        p = x0.clone()
        rTr = r.pow(2).sum()

        it = 0
        while rTr > tol and it < max_iter:
            it += 1
            q = M(p)
            alpha = rTr / ComplexCG.complexDot(p, q)[...,0]
            x += alpha * p
            r -= alpha * q

            rTrNew = r.pow(2).sum()

            beta = rTrNew / rTr

            p = r.clone() + beta * p
            rTr = rTrNew.clone()
        return x

    @staticmethod
    def forward(ctx, A, AH, max_iter, tol, lambdaa, x, y, *constants):
        ctx.tol = tol
        ctx.max_iter = max_iter
        ctx.A = A
        ctx.AH = AH

        def M(p):
            return AH(A(p, *constants), *constants) + lambdaa * p

        rhs = AH(y, *constants) + lambdaa * x
        ctx.save_for_backward(x, rhs, *constants, lambdaa)

        return ComplexCG.solve(rhs, M, ctx.tol, ctx.max_iter)

    @staticmethod
    def backward(ctx, grad_x):
        x, rhs, *constants, lambdaa = ctx.saved_tensors

        def M(p):
            return ctx.AH(ctx.A(p, *constants), *constants) + lambdaa * p

        Qe  = ComplexCG.solve(grad_x, M, ctx.tol, ctx.max_iter)
        QQe = ComplexCG.solve(Qe,     M, ctx.tol, ctx.max_iter)

        grad_x = lambdaa * Qe

        grad_lambdaa = mytorch.complex.complex_dotp(Qe, x)[...,0].sum() - \
                       mytorch.complex.complex_dotp(QQe, rhs)[...,0].sum()

        return None, None, None, None, grad_lambdaa, grad_x, None, None, None

class CGClass(torch.nn.Module):
    def __init__(self, A, AH, max_iter=10, tol=1e-10):
        super().__init__()
        self.A = A
        self.AH = AH
        self.max_iter = max_iter
        self.tol = tol

        self.cg = ComplexCG
        #ComplexCG.set_ops(self.cg, A, AH)
    
    def forward(self, lambdaa, x, y, *constants):
        out = torch.zeros_like(x)

        for n in range(x.shape[0]):
            cg_out = self.cg.apply(self.A, self.AH, self.max_iter, self.tol, lambdaa, x[n], y[n], *[c[n] for c in constants])
            out[n] = cg_out
        return out