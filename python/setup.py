from setuptools import setup
import os
from distutils.core import setup, Extension

VD_CASPR_module = Extension('VD_CASPR_CINE', sources = ['merlinpy/datapipeline/sampling/CASPR/InterfacePythonCpp.cpp'], extra_compile_args=['-std=c++11'])
VDPD_module = Extension('VDPD', sources = ['merlinpy/datapipeline/sampling/PoissonDisc/InterfacePythonCpp.cpp'], extra_compile_args=['-std=c++11'])

setup(
    name='merlinpy',
    version='0.3.0',
    author="Kerstin Hammernik and Thomas Kuestner",
    author_email="k.hammernik@imperial.ac.uk and thomas.kuestner@med.uni-tuebingen.de",
    packages=["merlinpy",
              "merlinpy.fastmri",
              "merlinpy.wandb"
              ],
    package_dir={"merlinpy": os.path.join('.', "merlinpy"),
                 "merlinpy.fastmri": os.path.join('.', "merlinpy/fastmri"),
                 "merlinpy.wandb": os.path.join('.', "merlinpy/wandb"),
    },
    install_requires=[
        "numpy >= 1.15",
        "xmltodict",
        "pyyaml",
        "pandas",
        "tqdm"
    ],
    ext_modules=[VD_CASPR_module, VDPD_module],
)
