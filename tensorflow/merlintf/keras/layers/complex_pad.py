import tensorflow as tf
import optotf.pad

def complex_pad(rank, x, pad, mode):
    if rank == 2:
        return complex_pad2d(x, pad, mode)
    elif rank == 3:
        return complex_pad3d(x, pad, mode)
    else:
        raise ValueError("Complex pad does only exist for 2D and 3D")

def complex_pad_transpose(rank, x, pad, mode):
    if rank == 2:
        return complex_pad2d_transpose(x, pad, mode)
    elif rank == 3:
        return complex_pad3d_transpose(x, pad, mode)
    else:
        raise ValueError("Complex pad does only exist for 2D and 3D")

def complex_pad2d(x, pad, mode='symmetric'):
    xp_re = optotf.pad.pad2d(tf.math.real(x), pad, mode=mode)
    xp_im = optotf.pad.pad2d(tf.math.imag(x), pad, mode=mode)

    return tf.complex(xp_re, xp_im)

def complex_pad2d_transpose(x, pad, mode='symmetric'):
    xp_re = optotf.pad.pad2d_transpose(tf.math.real(x), pad, mode=mode)
    xp_im = optotf.pad.pad2d_transpose(tf.math.imag(x), pad, mode=mode)

    return tf.complex(xp_re, xp_im)

def complex_pad3d(x, pad, mode='symmetric'):
    xp_re = optotf.pad.pad3d(tf.math.real(x), pad, mode=mode)
    xp_im = optotf.pad.pad3d(tf.math.imag(x), pad, mode=mode)

    return tf.complex(xp_re, xp_im)

def complex_pad3d_transpose(x, pad, mode='symmetric'):
    xp_re = optotf.pad.pad3d_transpose(tf.math.real(x), pad, mode=mode)
    xp_im = optotf.pad.pad3d_transpose(tf.math.imag(x), pad, mode=mode)

    return tf.complex(xp_re, xp_im)