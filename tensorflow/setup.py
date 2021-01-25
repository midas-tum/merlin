from setuptools import setup
import subprocess
import os
import tensorflow as tf

# If precompiled tensorflow isused, one has to destinguish between "tensorflow" and "tensorflow-gpu"
tfCPU = not subprocess.call(["pip","-q","show","tensorflow"] )
tfGPU = not subprocess.call(["pip","-q","show","tensorflow-gpu"] )
if tfCPU:
  tfstr = "tensorflow == {}".format(tf.version.VERSION)
if tfGPU:
  tfstr = "tensorflow-gpu == {}".format(tf.version.VERSION)
if (tfGPU and tfCPU) or not (tfGPU or tfCPU):
  tfstr = ""
  assert False, "\n\nunexpected error, is tensorflow or tensorflow-gpu installed with pip?\n\n"
  exit(1)
print ("=>required tensorflow for pip: %s\n"% tfstr)


# define requirements
REQUIRED_PACKAGES = [
    tfstr, # tensorflow or tensorflow-gpu
]

setup(
    name='merlintf',
    version='0.2.0',
    author="Kerstin Hammernik, Thomas Kuestner",
    author_email="k.hammernik@imperial.ac.uk, thomas.kuestner@med.uni-tuebingen.de",
    packages=["merlintf",
              "merlintf.optim",
              "merlintf.keras",
              "merlintf.keras.layers",
              "merlintf.keras.layers.convolutional",
              "merlintf.keras.models",
              "merlintf.keras.optimizers",
             ],
    package_dir={"merlintf": os.path.join('.', "merlintf"),
                 "merlintf.optim": os.path.join('.', "merlintf/optim"),
                 "merlintf.keras": os.path.join('.', "merlintf/keras"),
                 "merlintf.keras.layers": os.path.join('.', "merlintf/keras/layers"),
                 "merlintf.keras.layers.convolutional": os.path.join('.', "merlintf/keras/layers/convolutional"),
                 "merlintf.keras.models": os.path.join('.', "merlintf/keras/models"),
                 "merlintf.keras.optimizers": os.path.join('.', "merlintf/keras/optimizers"),
    },
    install_requires=REQUIRED_PACKAGES
)