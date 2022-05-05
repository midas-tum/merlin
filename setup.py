import os
import subprocess
import pathlib
from setuptools import setup

cwd = pathlib.Path().absolute()
os.chdir('python')
subprocess.run(['python', 'setup.py build'])
os.chdir(str(cwd))
subprocess.run(['pip install', './python'])
subprocess.run(['pip install', './pytorch'])
subprocess.run(['pip install', './tensorflow'])


setup(
    name='merlin',
    version='0.3.0',
    author="Kerstin Hammernik, Thomas Kuestner",
    author_email="k.hammernik@imperial.ac.uk, thomas.kuestner@med.uni-tuebingen.de",
    license='MIT',
    url='https://github.com/midas-tum/merlin',
    description='Machine Enhanced Reconstruction Learning and Interpretation Networks (MERLIN)',
    long_description=open('README.md').read(),
)