from setuptools import setup
import os

setup(
    name='mltools',
    version='0.1.0',
    author="Kerstin Hammernik",
    author_email="k.hammernik@imperial.ac.uk",
    packages=["mltools",
              "mltools.fastmri",
              ],
    package_dir={"mltools": os.path.join('.', "mltools"),
                 "mltools.fastmri": os.path.join('.', "mltools/fastmri"),
    },
    install_requires=[
        "numpy >= 1.15",
        "xmltodict",
    ],
)