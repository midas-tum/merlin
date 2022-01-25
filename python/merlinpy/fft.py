import numpy as np

def IFFT2c(x, axes=(-2,-1)):
    xshape = [x.shape[ax] for ax in axes]
    return np.sqrt(np.prod(xshape)) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)

def FFT2c(x, axes=(-2,-1)):
    xshape = [x.shape[ax] for ax in axes]
    return 1 / np.sqrt(np.prod(xshape)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)

def FFTNc(x, axes=(0,1)):
    for ax in axes:
        x = 1 / np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x

def IFFTNc(x, axes=(0,1)):
    for ax in axes:
        x = np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x

def IFFT2(x, axes=(-2,-1)):
    xshape = [x.shape[ax] for ax in axes]
    return np.sqrt(np.prod(xshape)) * np.fft.ifft2(x, axes=axes)

def FFT2(x, axes=(-2,-1)):
    xshape = [x.shape[ax] for ax in axes]
    return np.sqrt(np.prod(xshape)) * np.fft.fft2(x, axes=axes)

def FFTN(x, axes=(0,1)):
    for ax in axes:
        x = 1 / np.sqrt(x.shape[ax]) * np.fft.fft(x, axis=ax)
    return x

def IFFTN(x, axes=(0,1)):
    for ax in axes:
        x = np.sqrt(x.shape[ax]) * np.fft.ifft(x, axis=ax)
    return x