from setuptools import setup
from setuptools.dist import Distribution
import os
from distutils.core import setup, Extension
import shutil
import subprocess

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

currdir = os.getcwd()
compilepath = os.path.join('merlinpy', 'datapipeline', 'sampling', 'PoissonDisc')
os.chdir(compilepath)
subprocess.run(['python', 'setup_VDPD.py', 'build'])
os.chdir(currdir)
shutil.copyfile(os.path.join(compilepath, 'build', 'lib.linux-x86_64-cpython-38', os.listdir(os.path.join(compilepath, 'build', 'lib.linux-x86_64-cpython-38'))[0]), os.path.join('merlinpy', 'datapipeline', 'sampling', 'VDPDGauss.so'))

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
    package_data={"merlinpy.datapipeline.sampling": ['VDPDGauss.so'],
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
    license='MIT',
    url='https://github.com/midas-tum/merlin',
    description='Machine Enhanced Reconstruction Learning and Interpretation Networks (MERLIN) - merlinpy',
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
