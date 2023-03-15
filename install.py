import os
import subprocess
import pathlib
from setuptools import setup

cwd = pathlib.Path().absolute()
os.chdir('python')
subprocess.run(['python', 'setup.py sampling'])
os.chdir(str(cwd))
#subprocess.run(['pip install', './python'])
subprocess.run(['pip install', './pytorch'])
subprocess.run(['pip install', './tensorflow'])