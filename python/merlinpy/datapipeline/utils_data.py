import numpy as np

# N-dimensional FFT (considering ifftshift/fftshift operations)
def fftnc(x, axes=(0, 1)):
    # return 1 / np.sqrt(np.prod(x.shape[axes])) * np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
    for ax in axes:
        x = 1 / np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x

# N-dimensional IFFT (considering ifftshift/fftshift operations)
def ifftnc(x, axes=(0, 1)):
    # return np.sqrt(np.prod(x.shape[axes])) * np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
    for ax in axes:
        x = np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x

# centered zero-padding
def zpad(x, s, mode='constant'):
    # x: input data
    # s: desired size
    # mode: padding mode (constant, ...)
    if type(s) is not np.ndarray:
        s = np.asarray(s, dtype='f')

    if type(x) is not np.ndarray:
        x = np.asarray(x, dtype='f')

    m = np.asarray(np.shape(x), dtype='f')
    if len(m) < len(s):
        m = [m, np.ones(1, len(s) - len(m))]

    if np.sum(m == s) == len(m):
        return x

    idx = list()
    diff = list()
    for n in range(np.size(s)):

        if np.remainder(s[n], 2) == 0:
            idx.append(np.arange(np.floor(s[n] / 2) + 1 + np.ceil(-m[n] / 2) - 1, np.floor(s[n] / 2) + np.ceil(m[n] / 2)))
        else:
            idx.append(np.arange(np.floor(s[n] / 2) + np.ceil(-m[n] / 2) - 1, np.floor(s[n] / 2) + np.ceil(m[n] / 2) - 1))

        if s[n] != m[n]:
            diff.append( ( int(np.abs(idx[n][0])), int(np.abs(s[n]-1 - idx[n][-1])) ) )
        else:
            diff.append( (0, 0) )

    padval = tuple(i for i in diff)
    return np.pad(x, padval, mode=mode)

# centered cropping
def crop(x, s):
    # x: input data
    # s: desired size
    if type(s) is not np.ndarray:
        s = np.asarray(s, dtype='f')

    if type(x) is not np.ndarray:
        x = np.asarray(x, dtype='f')

    m = np.asarray(np.shape(x), dtype='f')
    if len(m) < len(s):
        m = [m, np.ones(1, len(s) - len(m))]

    if np.sum(m == s) == len(m):
        return x

    idx = list()
    for n in range(np.size(s)):
        if np.remainder(s[n], 2) == 0:
            idx.append(list(np.arange(np.floor(m[n] / 2) + 1 + np.ceil(-s[n] / 2) - 1, np.floor(m[n] / 2) + np.ceil(s[n] / 2), dtype=np.int)))
        else:
            idx.append(list(np.arange(np.floor(m[n] / 2) + np.ceil(-s[n] / 2) - 1, np.floor(m[n] / 2) + np.ceil(s[n] / 2) - 1), dtype=np.int))

    index_arrays = np.ix_(*idx)
    return x[index_arrays]