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
    name='mltoolstf',
    version='0.1.0',
    author="Kerstin Hammernik, Thomas Kuestner",
    author_email="k.hammernik@imperial.ac.uk, thomas.kuestner@med.uni-tuebingen.de",
    packages=["mltoolstf",
              "mltoolstf.optim",
              "mltoolstf.keras_utils",
              "mltoolstf.ddr_complex",
              #"mltoolstf.ddr_complex.recurrent"
             ],
    package_dir={"mltoolstf": os.path.join('.', "mltoolstf"),
                 "mltoolstf.optim": os.path.join('.', "mltoolstf/optim"),
                 "mltoolstf.keras_utils": os.path.join('.', "mltoolstf/keras_utils"),
                 "mltoolstf.ddr_complex": os.path.join('.', "mltoolstf/ddr_complex"),
                 #"mltoolstf.ddr_complex.recurrent": os.path.join('.', "mltoolstf/ddr_complex/recurrent"),
    },
    install_requires=REQUIRED_PACKAGES
)