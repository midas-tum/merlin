import tensorflow as tf
try:
    import optotf.keras.warp
except:
    print('optotf could not be imported')

class WarpForward(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.W = optotf.keras.warp.Warp(channel_last=False)

    def call(self, x, u):
        # we assume that the input does not have any channel dimension
        # x [batch, frames, M, N]
        # u [batch, frames, frames_all, M, N, 2]
        out_shape = tf.shape(u)[:-1]
        M, N = tf.shape(u)[-3:-1]
        x = tf.repeat(tf.expand_dims(x, -3), repeats=tf.shape(u)[-4], axis=-3)
        x = tf.reshape(x, (-1, 1, M, N)) # [batch, frames * frames_all, 1, M, N]
        u = tf.reshape(u, (-1, M, N, 2)) # [batch, frames * frames_all, M, N, 2]
        Wx = self.W(x, u)
        return tf.reshape(Wx, out_shape)

class WarpAdjoint(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.WH = optotf.keras.warp.WarpTranspose(channel_last=False)

    def call(self, x, u):
        # we assume that the input does not have any channel dimension
        # x [batch, frames, frames_all, M, N]
        # u [batch, frames, frames_all, M, N, 2]
        out_shape = tf.shape(u)[:-1]
        M, N = tf.shape(u)[-3:-1]
        x = tf.reshape(x, (-1, 1, M, N)) # [batch * frames * frames_all, 1, M, N]
        u = tf.reshape(u, (-1, M, N, 2)) # [batch * frames * frames_all, M, N, 2]
        x_warpT = self.WH(x, u, x)
        x_warpT = tf.reshape(x_warpT, out_shape)
        x_warpT = tf.math.reduce_sum(x_warpT, -3)
        return x_warpT

