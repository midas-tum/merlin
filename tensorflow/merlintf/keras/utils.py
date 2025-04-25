import os
import datetime
import tensorflow as tf
import merlinpy
import gc

center_crop = merlinpy.center_crop
#TODO do we really want to have merlinpy dependency here?

class ToKerasIO():
    def __init__(self, input_keys, output_keys):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, sample):
        inputs = []
        outputs = []
        for key in self.input_keys:
            inputs.append(sample[key])
        for key in self.output_keys:
            outputs.append(sample[key])
        return inputs, outputs

class AddKerasChannelDim(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = sample[key][..., None]
        return sample

def get_ndim(dim):
    if dim == '2D':
        n_dim = 2  # (x,y)
    elif dim == '3D':
        n_dim = 3  # (x,y,z)
    elif dim == '2Dt':
        n_dim = 3  # (t,x,y)
    elif dim == '3Dt':
        n_dim = 4  # (t,x,y,z)
    return n_dim

def validate_input_dimension(dim, param):
    n_dim = get_ndim(dim)
    if isinstance(param, tuple) or isinstance(param, list):
        if not len(param) == n_dim:
            raise RuntimeError("Parameter dimensions {} do not match requested dimensions {}!".format(len(param), n_dim))
        else:
            return param
    else:
        return tuple([param for _ in range(n_dim)])

def get_callbacks(validation_generator, model, logdir, flip_images = False):
    # Reshape the image for the Summary API.
    inputs, outputs = validation_generator.__getitem__(0)
    if isinstance(inputs, list):
      noisy = inputs[0]
    else:
      noisy = inputs
    if isinstance(outputs, list):
      target = outputs[0]
    else:
      target = outputs

    noisy = center_crop(noisy, target.shape[-3:-1], channel_last=True)

    def log_images(epoch, logs):
      prediction = model.predict(inputs)
      prediction = center_crop(prediction, target.shape[-3:-1], channel_last=True)

      # Creates a file writer for the log directory.
      file_writer = tf.summary.create_file_writer(logdir)
      # Using the file writer, log the reshaped image.
      def process(x):
        if flip_images:
          return tf.image.flip_left_right(tf.image.flip_up_down(x))
        else:
          return x

      with file_writer.as_default():
        tf.summary.image("Validation predict", process(tf.abs(prediction)), step=epoch)
        tf.summary.image("Validation noisy", process(tf.abs(noisy)), step=epoch)
        tf.summary.image("Validation target", process(tf.abs(target)), step=epoch)

    class GCCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()

    gc_callback = GCCallback()
    img_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_images)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    return [img_callback, tensorboard_callback, gc_callback]
