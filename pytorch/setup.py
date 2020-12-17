from setuptools import setup
import os

setup(
    name='merlinth',
    version='0.2.0',
    author="Kerstin Hammernik",
    author_email="k.hammernik@imperial.ac.uk",
    packages=["merlinth",
              "merlinth.optim",
              "merlinth.torch_utils",
              "merlinth.ddr_complex",
              "merlinth.mytorch",
              "merlinth.mytorch.loss",
              "merlinth.mytorch.optim",
              ],
    package_dir={"merlinth": os.path.join('.', "merlinth"),
                 "merlinth.optim": os.path.join('.', "merlinth/optim"),
                 "merlinth.torch_utils": os.path.join('.', "merlinth/torch_utils"),
                 "merlinth.ddr_complex": os.path.join('.', "merlinth/ddr_complex"),
                 "merlinth.mytorch": os.path.join('.', "merlinth/mytorch"),
                 "merlinth.mytorch.loss": os.path.join('.', "merlinth/mytorch/loss"),
                 "merlinth.mytorch.optim": os.path.join('.', "merlinth/mytorch/optim"),
    },
    install_requires=[
        "numpy >= 1.15",
        "torch >= 1.0.0",
        "torchvision >= 0.2.1",
    ],
)