import tensorflow as tf
import datetime
import os


def get_callbacks(validation_generator, model):
    # Add logging directory
    logdir = os.path.join("logs", model.name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Reshape the image for the Summary API.
    inputs, target = validation_generator.__getitem__(0)

    if isinstance(inputs, tuple) or isinstance(inputs, list):
      noisy = inputs[0]
    else:
      noisy = inputs

    def log_images(epoch, logs):
      # Creates a file writer for the log directory.
      file_writer = tf.summary.create_file_writer(logdir)
      # Using the file writer, log the reshaped image.
      with file_writer.as_default():
        tf.summary.image("Validation predict", tf.abs(model.predict(inputs)), step=epoch)
        tf.summary.image("Validation noisy", tf.abs(noisy), step=epoch)
        tf.summary.image("Validation target", tf.abs(target), step=epoch)

    img_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_images)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    return [img_callback, tensorboard_callback]
