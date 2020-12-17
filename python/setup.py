from setuptools import setup
import os

setup(
    name='merlin',
    version='0.2.0',
    author="Kerstin Hammernik and Thomas Kuestner",
    author_email="k.hammernik@imperial.ac.uk and thomas.kuestner@med.uni-tuebingen.de",
    packages=["merlin",
              "merlin.fastmri",
              ],
    package_dir={"merlin": os.path.join('.', "merlin"),
                 "merlin.fastmri": os.path.join('.', "merlin/fastmri"),
    },
    install_requires=[
        "numpy >= 1.15",
        "xmltodict",
    ],
)