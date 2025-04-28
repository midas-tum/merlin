from setuptools import setup, Command
import subprocess
import os
import sys
import io
import shutil
try:
    import tensorflow as tf
except:
    print('Tensorflow not installed')


DESCRIPTION = 'Machine Enhanced Reconstruction Learning and Interpretation Networks (MERLIN) - merlintf'
VERSION = '0.4.3'
DEBUG = False

# Readme
currdir = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(os.path.dirname(currdir), 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            shutil.rmtree(os.path.join(currdir, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        if DEBUG:
            os.system('python -m twine upload --repository testpypi dist/*')
        else:
            os.system('python -m twine upload dist/*')
            self.status('Pushing git tags…')
            os.chdir(os.path.dirname(currdir))
            os.system('git tag merlintf-v{0}'.format(VERSION))
            os.system('git push --tags')

        sys.exit()

setup(
    name='merlintf-mri',
    version=VERSION,
    author="Kerstin Hammernik, Thomas Kuestner",
    author_email="merlin.midastum@gmail.com",
    packages=["merlintf",
              "merlintf.optim",
              "merlintf.keras",
              "merlintf.keras.layers",
              "merlintf.keras.layers.convolutional",
              "merlintf.keras.models",
              "merlintf.keras.optimizers",
              "merlintf.test",
             ],
    package_dir={"merlintf": os.path.join('.', "merlintf"),
                 "merlintf.optim": os.path.join('.', "merlintf/optim"),
                 "merlintf.keras": os.path.join('.', "merlintf/keras"),
                 "merlintf.keras.layers": os.path.join('.', "merlintf/keras/layers"),
                 "merlintf.keras.layers.convolutional": os.path.join('.', "merlintf/keras/layers/convolutional"),
                 "merlintf.keras.models": os.path.join('.', "merlintf/keras/models"),
                 "merlintf.keras.optimizers": os.path.join('.', "merlintf/keras/optimizers"),
                 "merlintf.test": os.path.join('.', "merlintf/test"),
    },
    install_requires=[
        "numpy >= 1.15",
        "tensorflow>=2.4",
        "tensorflow-addons[tensorflow]",
        "pkgconfig",
        "h5py",
    ],
    license='MIT',
    url='https://github.com/midas-tum/merlin',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    cmdclass={
        'upload': UploadCommand,
    }
)