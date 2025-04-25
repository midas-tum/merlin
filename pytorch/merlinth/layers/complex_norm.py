import torch
from .complex_act import Identity

__all__ = [#'ComplexInstanceNormalization1d',
           'ComplexInstanceNormalization2d',
           'ComplexInstanceNormalization3d',
           'get_normalization']

def get_normalization(normalization):
    if normalization == 'instance':
        return ComplexInstanceNormalization3d()
    elif normalization == 'no':
        return Identity()
    else:
        raise ValueError("Options for normalization: ['no', 'instance']")

class _ComplexInstanceNormalizationCovariance(torch.nn.Module):
    def __init__(self):
        super(_ComplexInstanceNormalizationCovariance, self).__init__()

    def whiten2x2(self, tensor,
                        nugget=1e-5):
        """Solve R M R = I for R and a given 2x2 matrix M = [[a, b], [c, d]].
        Arguments
        ---------
        tensor : torch.tensor
            The input data expected to be at least 3d, with shape [B, F, ..., 2],
            where `B` is the batch dimension, `F` -- the channels/features,
            `...` -- the spatial dimensions (if present). The leading dimension
            `2` represents real and imaginary components (stacked).
        training : bool, default=True
            Determines whether to update running feature statistics, if they are
            provided, or use them instead of batch computed statistics. If `False`
            then `running_mean` and `running_cov` MUST be provided.
        momentum : float, default=0.1
            The weight in the exponential moving average used to keep track of the
            running feature statistics.
        nugget : float, default=1e-05
            The ridge coefficient to stabilise the estimate of the real-imaginary
            covariance.
        Details
        -------
        Using (tril) L L^T = V seems to 'favour' the first dimension (re), so
        Trabelsi et al. (2018) used explicit 2x2 root of M: such R that M = RR.
        For M = [[a, b], [c, d]] we have the following facts:
            (1) inv M = \frac1{ad - bc} [[d, -b], [-c, a]]
            (2) \sqrt{M} = \frac1{t} [[a + s, b], [c, d + s]]
                for s = \sqrt{ad - bc}, t = \sqrt{a + d + 2 s}
                det \sqrt{M} = t^{-2} (ad + s(d + a) + s^2 - bc) = s
        Therefore `inv \sqrt{M} = [[p, q], [r, s]]`, where
            [[p, q], [r, s]] = \frac1{t s} [[d + s, -b], [-c, a + s]]
        """
        # assume tensor is B x F x ... x
        # tail shape for broadcasting ? x 1 x F
        axes = tuple(range(2, tensor.dim()))
        print(tensor.shape, axes)

        # 1. compute batch mean [F] and center the batch
        mean = tensor.mean(dim=axes, keepdims=True)
        tensor = tensor - mean

        # 2. per feature real-imaginary 2x2 covariance matrix
        # faster than doing mul and then mean. Stabilize by a small ridge.
        var = torch.view_as_real(tensor).var(dim=axes, unbiased=False, keepdims=True) + nugget
        cov_uu, cov_vv = var.real(), var.imag()

        # has to mul-mean here anyway (naïve) : reduction axes shifted left.
        cov_vu = cov_uv = (tensor.real() * tensor.imag()).mean(dim=axes, keepdims=True)

        # 3. get R = [[p, q], [r, s]], with E R c c^T R^T = R M R = I
        # (unsure if intentional, but the inv-root in Trabelsi et al. (2018) uses
        # numpy `np.sqrt` instead of `K.sqrt` so grads are not passed through
        # properly, i.e. constants, [complex_standardization](bn.py#L56-57).
        sqrdet = torch.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
        # torch.det uses svd, so may yield -ve machine zero

        denom = sqrdet * torch.sqrt(cov_uu + 2 * sqrdet + cov_vv)
        p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
        r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom

        # 4. apply Q to x (manually)
        out = torch.stack([
            tensor.real() * p + tensor.imag() * r,
            tensor.real() * q + tensor.imag() * s,
        ], dim=-1)
        return torch.view_as_complex(out) # , torch.cat([p, q, r, s], dim=0).reshape(2, 2, -1)

    def forward(self, x):
        return self.whiten2x2(x)

class ComplexInstanceNorm_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, eps):
        # assume tensor is B x F x ... x 2
        # tail shape for broadcasting ? x 1 x F x [*1]
        axes = tuple(range(2, z.dim()))

        # 1. compute batch mean [2 x F] and center the batch
        mean = z.mean(dim=axes, keepdims=True)
        zhat = z - mean

        # 2. per feature real-imaginary 2x2 covariance matrix
        # faster than doing mul and then mean. Stabilize by a small ridge.
        #var = torch.mean(mytorch.complex.complex_mult_conj(zhat, zhat), dim=axes, keepdim=True) + eps
        var = torch.mean(zhat.conj() * zhat, dim=axes, keepdim=True) + eps
        #var = var[...,0].unsqueeze_(-1)
        std = torch.sqrt(var)
        zhat = zhat / std

        ctx.save_for_backward(zhat, std)        

        # 4. normalize
        return zhat
        
    @staticmethod
    def backward(ctx, grad_in):
        zhat = ctx.saved_tensors[0]
        std = ctx.saved_tensors[1]
        axes = tuple(range(2, zhat.dim()))

        # 1. compute batch mean [2 x F] and center the batch
        # grad_inH = grad_in.conj()
        # zhatH = zhat.conj()

        # part2a = mytorch.complex.complex_mult(grad_inH, zhat).mean(dim=axes, keepdim=True)#[...,0].unsqueeze_(-1)
        # part2b = mytorch.complex.complex_mult(grad_in, zhatH).mean(dim=axes, keepdim=True)#[...,0].unsqueeze_(-1)
        # part2 = mytorch.complex.complex_mult(part2a + part2b, zhat)
        part2a = torch.mean(grad_in.conj() * zhat, dim=axes, keepdim=True)
        part2b = torch.mean(grad_in * zhat.conj(), dim=axes, keepdim=True) 
        part2 = (part2a + part2b) * zhat

        part3 = grad_in.mean(dim=axes, keepdim=True)

        return (grad_in - part2/2 - part3) / std, None

class _ComplexInstanceNormalization(torch.nn.Module):
    def forward(self, z, eps=1e-5):
        return ComplexInstanceNorm_fun().apply(z, 1e-5)
# class ComplexInstanceNormalization1d(_ComplexInstanceNormalization):
#     """Complex-valued batch normalization for 2D or 3D data.
#     See torch.nn.BatchNorm1d for details.
#     """
#     def _check_input_dim(self, input):
#         if input.dim() != 2 and input.dim() != 3:
#             raise ValueError('expected 2D or 3D input (got {}D input)'
#                              .format(input.dim()))

class ComplexInstanceNormalization2d(_ComplexInstanceNormalization):
    """Complex-valued batch normalization for 4D data.
    See torch.nn.BatchNorm2d for details.
    """
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class ComplexInstanceNormalization3d(_ComplexInstanceNormalization):
    """Complex-valued batch normalization for 5D data.
    See torch.nn.BatchNorm3d for details.
    """
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
