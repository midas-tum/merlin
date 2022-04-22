from setuptools import setup
from setuptools.dist import Distribution
import os
from distutils.core import setup, Extension

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    name='merlinpy',
    version='0.3.0',
    author="Kerstin Hammernik, Thomas Kuestner",
    author_email="k.hammernik@imperial.ac.uk, thomas.kuestner@med.uni-tuebingen.de",
    packages=["merlinpy",
              "merlinpy.fastmri",
              "merlinpy.wandb",
              "merlinpy.recon",
              "merlinpy.losses",
              "merlinpy.datapipeline",
              "merlinpy.datapipeline.sampling",
              "merlinpy.datapipeline.sampling.VISTA",
              "merlinpy.test"
              ],
    package_dir={"merlinpy": os.path.join('.', "merlinpy"),
                 "merlinpy.fastmri": os.path.join('.', "merlinpy/fastmri"),
                 "merlinpy.wandb": os.path.join('.', "merlinpy/wandb"),
                 "merlinpy.recon": os.path.join('.', "merlinpy/recon"),
                 "merlinpy.losses": os.path.join('.', "merlinpy/losses"),
                 "merlinpy.datapipeline": os.path.join('.', "merlinpy/datapipeline"),
                 "merlinpy.datapipeline.sampling": os.path.join('.', "merlinpy/datapipeline/sampling"),
                 "merlinpy.datapipeline.sampling.VISTA": os.path.join('.', "merlinpy/datapipeline/sampling/VISTA"),
                 "merlinpy.test": os.path.join('.', "merlinpy/test"),
    },
    include_package_data=True,
    package_data={"merlinpy.datapipeline.sampling": ['*.so'],
    },
    install_requires=[
        "numpy >= 1.15",
        "xmltodict",
        "pyyaml",
        "pandas",
        "tqdm",
        "scipy",
        "scikit-image",
        "matplotlib",
        "joblib",
        "sphinx",
        "sphinxjp.themes.basicstrap",
        "sphinx-rtd-theme",
        "breathe",
    ],
    distclass=BinaryDistribution,
)
