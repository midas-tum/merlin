from setuptools import setup
import subprocess
import os
import tensorflow as tf

# If precompiled tensorflow isused, one has to destinguish between "tensorflow" and "tensorflow-gpu"
tfCPU = not subprocess.call(["pip","-q","show","tensorflow"] )
tfGPU = not subprocess.call(["pip","-q","show","tensorflow-gpu"] )
tfNightly = not subprocess.call(["pip","-q","show","tf-nightly"] )
tfaddons = not subprocess.call(["pip","-q","show","tensorflow-addons"])

tfstra = ""
if tfCPU:
  tfstr = "tensorflow=={}".format(tf.version.VERSION)
  if not tfaddons:
    tfstra = "tensorflow-addons[tensorflow]"
if tfGPU:
  tfstr = "tensorflow-gpu=={}".format(tf.version.VERSION)
  if not tfaddons:
    tfstra = "tensorflow-addons[tensorflow-gpu]"
if not (tfGPU or tfCPU):
  tfstr = ""
  tfstra = ""
  assert False, "\n\nunexpected error, is tensorflow or tensorflow-gpu installed with pip?\n\n"
  exit(1)
print ("=>required tensorflow for pip: %s\n"% tfstr)
if not tfaddons:
    print("=>required tensorflow-addons for pip: %s\n" % tfstra)

# define requirements
if not tfaddons:
    REQUIRED_PACKAGES = [tfstr, # tensorflow or tensorflow-gpu
        tfstra,
    ]
else:
    REQUIRED_PACKAGES = [tfstr]

setup(
    name='merlintf',
    version='0.2.3',
    author="Kerstin Hammernik, Thomas Kuestner",
    author_email="k.hammernik@imperial.ac.uk, thomas.kuestner@med.uni-tuebingen.de",
    packages=["merlintf",
              "merlintf.optim",
              "merlintf.keras",
              "merlintf.keras.layers",
              "merlintf.keras.layers.convolutional",
              "merlintf.keras.models",
              "merlintf.keras.optimizers",
              "merlintf.test",
             ],
    package_dir={"merlintf": os.path.join('.', "merlintf"),
                 "merlintf.optim": os.path.join('.', "merlintf/optim"),
                 "merlintf.keras": os.path.join('.', "merlintf/keras"),
                 "merlintf.keras.layers": os.path.join('.', "merlintf/keras/layers"),
                 "merlintf.keras.layers.convolutional": os.path.join('.', "merlintf/keras/layers/convolutional"),
                 "merlintf.keras.models": os.path.join('.', "merlintf/keras/models"),
                 "merlintf.keras.optimizers": os.path.join('.', "merlintf/keras/optimizers"),
                 "merlintf.test": os.path.join('.', "merlintf/test"),
    },
    install_requires=REQUIRED_PACKAGES,
)