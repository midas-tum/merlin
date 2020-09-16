from setuptools import setup
import os

setup(
    name='vntoolsth',
    version='0.1.0',
    author="Kerstin Hammernik",
    author_email="k.hammernik@imperial.ac.uk",
    packages=["vntoolsth",
              "vntoolsth.optim",
              "vntoolsth.ddr_complex"],
    package_dir={"vntoolsth": os.path.join('.', "vntoolsth"),
                 "vntoolsth.optim": os.path.join('.', "vntoolsth/optim"),
                 "vntoolsth.ddr_complex": os.path.join('.', "vntoolsth/ddr_complex"),
    },
    install_requires=[
        "numpy >= 1.15",
        "torch >= 1.0.0",
        "torchvision >= 0.2.1",
    ],
)