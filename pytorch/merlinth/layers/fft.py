import torch

def fft2(x, dim=(-2,-1)):
    return torch.fft.fft2(x, dim=dim, norm='ortho')

def ifft2(X, dim=(-2,-1)):
    return torch.fft.ifft2(X, dim=dim, norm='ortho')

def fft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(fft2(torch.fft.ifftshift(x, dim), dim), dim)

def ifft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(ifft2(torch.fft.ifftshift(x, dim), dim), dim)
