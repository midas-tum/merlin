from setuptools import setup
import os

setup(
    name='mltoolsth',
    version='0.1.0',
    author="Kerstin Hammernik",
    author_email="k.hammernik@imperial.ac.uk",
    packages=["mltoolsth",
              "mltoolsth.optim",
              "mltoolsth.ddr_complex",
              "mltoolsth.mytorch",
              "mltoolsth.mytorch.loss",
              "mltoolsth.mytorch.optim",
              ],
    package_dir={"mltoolsth": os.path.join('.', "mltoolsth"),
                 "mltoolsth.optim": os.path.join('.', "mltoolsth/optim"),
                 "mltoolsth.ddr_complex": os.path.join('.', "mltoolsth/ddr_complex"),
                 "mltoolsth.mytorch": os.path.join('.', "mltoolsth/mytorch"),
                 "mltoolsth.mytorch.loss": os.path.join('.', "mltoolsth/mytorch/loss"),
                 "mltoolsth.mytorch.optim": os.path.join('.', "mltoolsth/mytorch/optim"),
    },
    install_requires=[
        "numpy >= 1.15",
        "torch >= 1.0.0",
        "torchvision >= 0.2.1",
    ],
)