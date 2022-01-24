from setuptools import setup
import os

setup(
    name='merlinth',
    version='0.3.0',
    author="Kerstin Hammernik, Thomas Kuestner",
    author_email="k.hammernik@imperial.ac.uk, thomas.kuestner@med.uni-tuebingen.de",
    packages=["merlinth",
              "merlinth.optim",
              "merlinth.layers",
              "merlinth.losses",
              "merlinth.layers.convolutional",
              "merlinth.models",
              ],
    package_dir={"merlinth": os.path.join('.', "merlinth"),
                 "merlinth.optim": os.path.join('.', "merlinth/optim"),
                 "merlinth.layers": os.path.join('.', "merlinth/layers"),
                 "merlinth.losses": os.path.join('.', "merlinth/losses"),
                 "merlinth.layers.convolutional": os.path.join('.', "merlinth/layers/convolutional"),
                 "merlinth.models": os.path.join('.', "merlinth/models"),
    },
    install_requires=[
        "numpy >= 1.15",
        "torch >= 1.0.0",
        "torchvision >= 0.2.1",
    ],
)