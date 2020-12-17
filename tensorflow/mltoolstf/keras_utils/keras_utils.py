
import os
import datetime
import tensorflow as tf
import mltools
import gc

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

    noisy = mltools.utils.center_crop(noisy, target.shape[-3:-1], channel_last=True)

    def log_images(epoch, logs):
      prediction = model.predict(inputs)
      prediction = mltools.utils.center_crop(prediction, target.shape[-3:-1], channel_last=True)

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
