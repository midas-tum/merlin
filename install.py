import os
import subprocess
import pathlib
from setuptools import setup

cwd = pathlib.Path().absolute()
os.chdir('python')
subprocess.run(['python', 'setup.py sampling'])
os.chdir(str(cwd))
#subprocess.run(['pip install', './python'])
os.chdir('pytorch')
subprocess.run(['python', 'setup.py install'])
os.chdir(str(cwd))
os.chdir('tensorflow')
subprocess.run(['python', 'setup.py install'])