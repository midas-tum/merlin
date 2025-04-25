# import tensorflow as tf
# import optotf.pad
# import merlintf

# """
# Interface to optox padding layers 
# """

# # main functions
# def Pad2D(x, pad, mode):
#     return pad(2, x, pad, mode)

# def Pad1D(x, pad, mode):
#     return pad(1, x, pad, mode)

# def Pad3D(x, pad, mode):
#     return pad(3, x, pad, mode)

# def Pad4D(x, pad, mode):
#     return pad(4, x, pad, mode)

# def Pad1DTranspose(x, pad, mode):
#     return pad_transpose(1, x, pad, mode)

# def Pad2DTranspose(x, pad, mode):
#     return pad_transpose(2, x, pad, mode)

# def Pad3DTranspose(x, pad, mode):
#     return pad_transpose(3, x, pad, mode)

# def Pad4DTranspose(x, pad, mode):
#     return pad_transpose(4, x, pad, mode)

# def pad(rank, x, pad, mode):
#     if merlintf.iscomplex(x):
#         return complex_pad(rank, x, pad, mode)
#     else:
#         return real_pad(rank, x, pad, mode)

# def pad_transpose(rank, x, pad, mode):
#     if merlintf.iscomplex(x):
#         return complex_pad_transpose(rank, x, pad, mode)
#     else:
#         return real_pad_transpose(rank, x, pad, mode)

# # real-valued padding
# def real_pad(rank, x, pad, mode):
#     if rank == 1:
#         return real_pad1d(x, pad, mode)
#     elif rank == 2:
#         return real_pad2d(x, pad, mode)
#     elif rank == 3:
#         return real_pad3d(x, pad, mode)
#     else:
#         raise ValueError("Real pad does only exist for 1D, 2D and 3D")

# def real_pad_transpose(rank, x, pad, mode):
#     if rank == 1:
#         return real_pad1d_transpose(x, pad, mode)
#     elif rank == 2:
#         return real_pad2d_transpose(x, pad, mode)
#     elif rank == 3:
#         return real_pad3d_transpose(x, pad, mode)
#     else:
#         raise ValueError("Real pad does only exist for 1D, 2D and 3D")

# def real_pad1d(x, pad, mode='symmetric'):
#     return optotf.pad.pad1d(x, pad, mode=mode)

# def real_pad1d_transpose(x, pad, mode='symmetric'):
#     return optotf.pad.pad1d_transpose(x, pad, mode=mode)

# def real_pad2d(x, pad, mode='symmetric'):
#     return optotf.pad.pad2d(x, pad, mode=mode)

# def real_pad2d_transpose(x, pad, mode='symmetric'):
#     return optotf.pad.pad2d_transpose(x, pad, mode=mode)

# def real_pad3d(x, pad, mode='symmetric'):
#     return optotf.pad.pad3d(x, pad, mode=mode)

# def real_pad3d_transpose(x, pad, mode='symmetric'):
#     return optotf.pad.pad3d_transpose(x, pad, mode=mode)

# # complex-valued padding
# def complex_pad(rank, x, pad, mode):
#     if rank == 1:
#         return complex_pad1d(x, pad, mode)
#     elif rank == 2:
#         return complex_pad2d(x, pad, mode)
#     elif rank == 3:
#         return complex_pad3d(x, pad, mode)
#     else:
#         raise ValueError("Complex pad does only exist for 1D, 2D and 3D")

# def complex_pad_transpose(rank, x, pad, mode):
#     if rank == 1:
#         return complex_pad1d_transpose(x, pad, mode)
#     elif rank == 2:
#         return complex_pad2d_transpose(x, pad, mode)
#     elif rank == 3:
#         return complex_pad3d_transpose(x, pad, mode)
#     else:
#         raise ValueError("Complex pad does only exist for 1D, 2D and 3D")

# def complex_pad1d(x, pad, mode='symmetric'):
#     xp_re = optotf.pad.pad1d(tf.math.real(x), pad, mode=mode)
#     xp_im = optotf.pad.pad1d(tf.math.imag(x), pad, mode=mode)

#     return tf.complex(xp_re, xp_im)

# def complex_pad1d_transpose(x, pad, mode='symmetric'):
#     xp_re = optotf.pad.pad1d_transpose(tf.math.real(x), pad, mode=mode)
#     xp_im = optotf.pad.pad1d_transpose(tf.math.imag(x), pad, mode=mode)

#     return tf.complex(xp_re, xp_im)

# def complex_pad2d(x, pad, mode='symmetric'):
#     xp_re = optotf.pad.pad2d(tf.math.real(x), pad, mode=mode)
#     xp_im = optotf.pad.pad2d(tf.math.imag(x), pad, mode=mode)

#     return tf.complex(xp_re, xp_im)

# def complex_pad2d_transpose(x, pad, mode='symmetric'):
#     xp_re = optotf.pad.pad2d_transpose(tf.math.real(x), pad, mode=mode)
#     xp_im = optotf.pad.pad2d_transpose(tf.math.imag(x), pad, mode=mode)

#     return tf.complex(xp_re, xp_im)

# def complex_pad3d(x, pad, mode='symmetric'):
#     xp_re = optotf.pad.pad3d(tf.math.real(x), pad, mode=mode)
#     xp_im = optotf.pad.pad3d(tf.math.imag(x), pad, mode=mode)

#     return tf.complex(xp_re, xp_im)

# def complex_pad3d_transpose(x, pad, mode='symmetric'):
#     xp_re = optotf.pad.pad3d_transpose(tf.math.real(x), pad, mode=mode)
#     xp_im = optotf.pad.pad3d_transpose(tf.math.imag(x), pad, mode=mode)

#     return tf.complex(xp_re, xp_im)


# # Aliases
# Pad2Dt = Pad3D
# Pad2DtTranspose = Pad3DTranspose
# Pad3Dt = Pad4D
# Pad3DtTranspose = Pad4DTranspose