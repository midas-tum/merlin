from setuptools import setup
import subprocess
import os
try:
    import tensorflow as tf
except:
    print('Tensorflow not installed')

# If precompiled tensorflow is used, one has to destinguish between "tensorflow" and "tensorflow-gpu"
tfCPU = not subprocess.call(["pip","-q","show","tensorflow"])
tfGPU = not subprocess.call(["pip","-q","show","tensorflow-gpu"])
tfNightly = not subprocess.call(["pip","-q","show","tf-nightly"])
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
  tfstr = "tensorflow-gpu"
  if not tfaddons:
    tfstra = "tensorflow-addons[tensorflow-gpu]"

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
    author_email="merlin.midastum@gmail.com",
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
    license='MIT',
    url='https://github.com/midas-tum/merlin',
    description='Machine Enhanced Reconstruction Learning and Interpretation Networks (MERLIN) - merlintf',
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)