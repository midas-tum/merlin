import tensorflow.keras.backend as K
import tensorflow as tf

# assumed input dimensions
# 2D: [batch, channel, height, width] or [batch, height, width, channel]
# 2Dt: [batch, channel, time, height, width] or [batch, time, height, width, channel]
# 3D: [batch, channel, depth, height, width] or [batch, depth, height, width, channel]
# 3Dt: [batch, channel, time, depth, height, width] or [batch, time, depth, height, width, channel]

'''
Complex MSE Loss
'''
@tf.function
def loss_complex_mse(y_true, y_pred, axis=(2,3,4)):
    if tf.keras.backend.image_data_format() == 'channels_first':
        axismean = (0, 1)
    elif tf.keras.backend.image_data_format() == 'channels_last':
        axismean = (0, -1)
    diff = (y_true - y_pred)
    return K.mean(K.sum(tf.math.real(tf.math.conj(diff) * diff), axis=axis), axis=axismean)

@tf.function
def loss_complex_mse_2D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_complex_mse(y_true, y_pred, axis=(2,3))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_complex_mse(y_true, y_pred, axis=(1,2))

@tf.function
def loss_complex_mse_3D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_complex_mse(y_true, y_pred, axis=(2,3,4))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_complex_mse(y_true, y_pred, axis=(1,2,3))

@tf.function
def loss_complex_mse_2Dt(y_true, y_pred):
    return loss_complex_mse_3D(y_true, y_pred)

@tf.function
def loss_complex_mse_4D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_complex_mse(y_true, y_pred, axis=(2,3,4,5))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_complex_mse(y_true, y_pred, axis=(1,2,3,4))

@tf.function
def loss_complex_mse_3Dt(y_true, y_pred):
    return loss_complex_mse_4D(y_true, y_pred)

'''
Abs MSE Loss
'''
@tf.function
def loss_abs_mse(y_true, y_pred, axis=(2,3,4)):
    if tf.keras.backend.image_data_format() == 'channels_first':
        axismean = (0, 1)
    elif tf.keras.backend.image_data_format() == 'channels_last':
        axismean = (0, -1)
    diff = (merlintf.complex_abs(y_true) - merlintf.complex_abs(y_pred))
    return K.mean(K.sum(tf.math.real(tf.math.conj(diff) * diff), axis=axis), axis=axismean)

@tf.function
def loss_abs_mse_2D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_abs_mse(y_true, y_pred, axis=(2, 3))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_abs_mse(y_true, y_pred, axis=(1, 2))

@tf.function
def loss_abs_mse_3D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_abs_mse(y_true, y_pred, axis=(2, 3, 4))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_abs_mse(y_true, y_pred, axis=(1, 2, 3))

@tf.function
def loss_abs_mse_2Dt(y_true, y_pred):
    return loss_abs_mse_3D(y_true, y_pred)

@tf.function
def loss_abs_mse_4D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_abs_mse(y_true, y_pred, axis=(2, 3, 4, 5))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_abs_mse(y_true, y_pred, axis=(1, 2, 3, 4))

@tf.function
def loss_abs_mse_3Dt(y_true, y_pred):
    return loss_abs_mse_4D(y_true, y_pred)

'''
Complex MAE Loss
'''
@tf.function
def loss_complex_mae(y_true, y_pred, axis=(2,3,4)):
    if tf.keras.backend.image_data_format() == 'channels_first':
        axismean = (0, 1)
    elif tf.keras.backend.image_data_format() == 'channels_last':
        axismean = (0, -1)
    diff = (y_true - y_pred)
    return K.mean(K.sum(tf.sqrt(tf.math.real(tf.math.conj(diff) * diff) + 1e-9), axis=axis), axis=axismean)

@tf.function
def loss_complex_mae_2D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_complex_mae(y_true, y_pred, axis=(2, 3))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_complex_mae(y_true, y_pred, axis=(1, 2))

@tf.function
def loss_complex_mae_3D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_complex_mae(y_true, y_pred, axis=(2, 3, 4))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_complex_mae(y_true, y_pred, axis=(1, 2, 3))

@tf.function
def loss_complex_mae_2Dt(y_true, y_pred):
    return loss_complex_mae_3D(y_true, y_pred)

@tf.function
def loss_complex_mae_4D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_complex_mae(y_true, y_pred, axis=(2, 3, 4, 5))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_complex_mae(y_true, y_pred, axis=(1, 2, 3, 4))

@tf.function
def loss_complex_mae_3Dt(y_true, y_pred):
    return loss_complex_mae_4D(y_true, y_pred)

'''
Abs MAE Loss
'''
@tf.function
def loss_abs_mae(y_true, y_pred, axis=(2,3,4)):
    if tf.keras.backend.image_data_format() == 'channels_first':
        axismean = (0, 1)
    elif tf.keras.backend.image_data_format() == 'channels_last':
        axismean = (0, -1)
    diff = (merlintf.complex_abs(y_true) - merlintf.complex_abs(y_pred))
    return K.mean(K.sum(tf.sqrt(tf.math.real(tf.math.conj(diff) * diff) + 1e-9), axis=axis), axis=axismean)

@tf.function
def loss_abs_mae_2D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_abs_mae(y_true, y_pred, axis=(2, 3))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_abs_mae(y_true, y_pred, axis=(1, 2))

@tf.function
def loss_abs_mae_3D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_abs_mae(y_true, y_pred, axis=(2, 3, 4))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_abs_mae(y_true, y_pred, axis=(1, 2, 3))

@tf.function
def loss_abs_mae_2Dt(y_true, y_pred):
    return loss_abs_mae_3D(y_true, y_pred)

@tf.function
def loss_abs_mae_4D(y_true, y_pred):
    if tf.keras.backend.image_data_format() == 'channels_first':
        return loss_abs_mae(y_true, y_pred, axis=(2, 3, 4, 5))
    elif tf.keras.backend.image_data_format() == 'channels_last':
        return loss_abs_mae(y_true, y_pred, axis=(1, 2, 3, 4))

@tf.function
def loss_abs_mae_3Dt(y_true, y_pred):
    return loss_abs_mae_4D(y_true, y_pred)