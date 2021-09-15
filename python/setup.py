from setuptools import setup
from setuptools.dist import Distribution
import os

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    name='merlinpy',
    version='0.2.1',
    author="Kerstin Hammernik and Thomas Kuestner",
    author_email="k.hammernik@imperial.ac.uk and thomas.kuestner@med.uni-tuebingen.de",
    packages=["merlinpy",
              "merlinpy.fastmri",
              "merlinpy.wandb",
              "merlinpy.datapipeline",
              "merlinpy.datapipeline.sampling",
              "merlinpy.datapipeline.sampling.VISTA",
              ],
    package_dir={"merlinpy": os.path.join('.', "merlinpy"),
                 "merlinpy.fastmri": os.path.join('.', "merlinpy/fastmri"),
                 "merlinpy.wandb": os.path.join('.', "merlinpy/wandb"),
                 "merlinpy.datapipeline": os.path.join('.', "merlinpy/datapipeline"),
                 "merlinpy.datapipeline.sampling": os.path.join('.', "merlinpy/datapipeline/sampling"),
                 "merlinpy.datapipeline.sampling.VISTA": os.path.join('.', "merlinpy/datapipeline/sampling/VISTA"),
    },
    include_package_data=True,
    package_data={"merlinpy.datapipeline.sampling": ['*.so'],
    },
    install_requires=[
        "numpy >= 1.15",
        "xmltodict",
        "pyyaml",
        "pandas",
        "tqdm"
    ],
    distclass=BinaryDistribution,
)
