
import torch

__all__ = ['ComplexRegularizer',
            'ComplexRegularizerWrapper3d',
          'ComplexRegularizerWrapper3dBatched',
           'ComplexRegularizerWrapper',
           'PseudoComplexRegularizerWrapper']

class ComplexRegularizer(torch.nn.Module):
    """
    Basic regularization function
    """

    def __init__(self):
        super(ComplexRegularizer, self).__init__()

    def forward(self, x):
        return self.grad(x)

    def energy(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

    def get_theta(self):
        """
        return all parameters of the regularization
        """
        return self.named_parameters()

    def get_vis(self):
        raise NotImplementedError

class ComplexRegularizerWrapper(torch.nn.Module):
    """
    Basic regularization function
    """

    def __init__(self, R):
        super(ComplexRegularizerWrapper, self).__init__()

        self.R = R

    def forward(self, x):
        return self.grad(x)

    def energy(self, x):
        # re-shape data from [nBatch, nSmaps, nFE, nPE, 2]
        # to [nBatch*nSmaps, 1, nFE, nPE, 2]
        shp = x.shape
        x = x.view(shp[0] * shp[1], 1, *shp[2:])

        # apply denoising
        return self.R.energy(x)

    def grad(self, x):
        # re-shape data from [nBatch, nSmaps, nFE, nPE, 2]
        # to [nBatch*nSmaps, 1, nFE, nPE, 2]
        shp = x.shape
        x = x.view(shp[0] * shp[1], 1, *shp[2:])

        # apply reg
        x = self.R.grad(x)

        # re-shape data from [nBatch, nSmaps, nFE, nPE, 2]
        # to [nBatch*nSmaps, 1, nFE, nPE, 2]
        x = x.view(*shp)
        return x

    def get_theta(self):
        """
        return all parameters of the regularization
        """
        return self.R.named_parameters()

    def get_vis(self):
        return self.R.get_vis()

class ComplexRegularizerWrapper3dBatched(torch.nn.Module):
    """
    Basic regularization function
    """

    def __init__(self, R):
        super(ComplexRegularizerWrapper3dBatched, self).__init__()

        self.R = R

    def forward(self, x):
        # reshape 5d tensor [batch_size, D, H, W, 2] to 6d tensor [batch_size, channels, D, H, W, 2]
        shp = x.shape
        x = x.view(x.shape[0], 1, *x.shape[1:])

        # apply reg
        x = self.R(x)

        # re-shape data to original size
        x = x.view(*shp)
        return x

    def energy(self, x):
        # reshape 5d tensor [batch_size, D, H, W, 2] to 6d tensor [batch_size, channels, D, H, W, 2]
        x = x.view(1, 1, *x.shape)

        # apply denoising
        return self.R.energy(x)

    def grad(self, x):
        # reshape 5d tensor [batch_size, D, H, W, 2] to 6d tensor [batch_size, channels, D, H, W, 2]
        shp = x.shape
        x = x.view(x.shape[0], 1, *x.shape[1:])

        # apply reg
        x = self.R.grad(x)

        # re-shape data to original size
        x = x.view(*shp)
        return x

    def get_theta(self):
        """
        return all parameters of the regularization
        """
        return self.R.named_parameters()

    def get_vis(self):
        return self.R.get_vis()

class ComplexRegularizerWrapper3d(torch.nn.Module):
    """
    Basic regularization function
    """

    def __init__(self, R):
        super(ComplexRegularizerWrapper3d, self).__init__()

        self.R = R

    def forward(self, x):
        # reshape 3d tensor [D, H, W, 2] to 5d tensor [batch_size, channels, D, H, W, 2]
        shp = x.shape
        x = x.view(1, 1, *x.shape)

        # apply reg
        x = self.R(x)

        # re-shape data from [nBatch, nSmaps, nFE, nPE, 2]
        # to [nBatch*nSmaps, 1, nFE, nPE, 2]
        x = x.view(*shp)
        return x

    def energy(self, x):
        # reshape 3d tensor [D, H, W, 2] to 5d tensor [batch_size, channels, D, H, W, 2]
        x = x.view(1, 1, *x.shape)

        # apply denoising
        return self.R.energy(x)

    def grad(self, x):
        # reshape 3d tensor [D, H, W, 2] to 5d tensor [batch_size, channels, D, H, W, 2]
        shp = x.shape
        x = x.view(1, 1, *x.shape)

        # apply reg
        x = self.R.grad(x)

        # re-shape data from [nBatch, nSmaps, nFE, nPE, 2]
        # to [nBatch*nSmaps, 1, nFE, nPE, 2]
        x = x.view(*shp)
        return x

    def get_theta(self):
        """
        return all parameters of the regularization
        """
        return self.R.named_parameters()

    def get_vis(self):
        return self.R.get_vis()

class PseudoComplexRegularizerWrapper(ComplexRegularizerWrapper):
    def energy(self, x):
        # re-shape data from [nBatch, nSmaps, nFE, nPE, 2]
        # to [nBatch*nSmaps, 2, nFE, nPE]
        shp = x.shape
        x = x.view(shp[0] * shp[1], *shp[2:]).permute((0, 3, 1, 2))

        # apply denoising
        return self.R.energy(x)

    def grad(self, x):
        # re-shape data from [nBatch, nSmaps, nFE, nPE, 2]
        # to [nBatch*nSmaps, 2, nFE, nPE]
        shp = x.shape
        x = x.view(shp[0] * shp[1], *shp[2:]).permute((0, 3, 1, 2))

        # apply reg
        x = self.R.grad(x)

        # re-shape data from [nBatch, nSmaps, 2, nFE, nPE]
        # to [nBatch*nSmaps, 1, nFE, nPE, 2]
        x = x.permute((0, 2, 3, 1)).view(*shp)
        return x
