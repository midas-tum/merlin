from setuptools import setup
import os

setup(
    name='merlinpy',
    version='0.2.0',
    author="Kerstin Hammernik and Thomas Kuestner",
    author_email="k.hammernik@imperial.ac.uk and thomas.kuestner@med.uni-tuebingen.de",
    packages=["merlinpy",
              "merlinpy.fastmri",
              ],
    package_dir={"merlinpy": os.path.join('.', "merlinpy"),
                 "merlinpy.fastmri": os.path.join('.', "merlinpy/fastmri"),
    },
    install_requires=[
        "numpy >= 1.15",
        "xmltodict",
    ],
)