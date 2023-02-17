from setuptools import setup, Command
from setuptools.dist import Distribution
import os
import sys
from distutils.core import setup, Extension
import shutil
import subprocess
import io
import importlib.util

# Package meta-data
DESCRIPTION = 'Machine Enhanced Reconstruction Learning and Interpretation Networks (MERLIN) - merlinpy'
VERSION = '0.3.9'

currdir = os.path.abspath(os.path.dirname(__file__))  #os.getcwd()
compilepath = os.path.join('merlinpy', 'datapipeline', 'sampling', 'PoissonDisc')

if len(sys.argv) > 1 and sys.argv[1] == 'sampling':
    INCLUDE_PACKAGE_DATA = True
    PACKAGE_DATA = {"merlinpy.datapipeline.sampling": ['VDPDGauss.so'],}
else:
    INCLUDE_PACKAGE_DATA = False
    PACKAGE_DATA = {}
# check if run via pip install or locally (enable build of sampling code)
# pip requires manylinux conform code, hence C++ sampling code need to be compiled for each GLIBC, Python and header versions
# -> opting for local workaround: build the sampling code yourself locally using "python3 setup.py

# Readme
try:
    with io.open(os.path.join(os.path.dirname(currdir), 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        if INCLUDE_PACKAGE_DATA:
            return True
        else:
            return False

class SamplingCommand(Command):
    """Support setup.py upload."""

    description = 'Build and install the package.'
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
        self.status('Build and install locally - for sampling code')
        try:
            self.status('Removing previous builds…')
            shutil.rmtree(os.path.join(currdir, 'dist'))
            shutil.rmtree(os.path.join(currdir, 'build'))
            shutil.rmtree(os.path.join(currdir, 'wheelhouse'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.chdir(compilepath)
        subprocess.run(['python3', 'setup_VDPD.py', 'build'])
        os.chdir(currdir)
        libpath = [p for p in os.listdir(os.path.join(compilepath, 'build')) if p.startswith('lib')][0]
        shutil.copyfile(
            os.path.join(compilepath, 'build', libpath, os.listdir(os.path.join(compilepath, 'build', libpath))[0]),
            os.path.join('merlinpy', 'datapipeline', 'sampling', 'VDPDGauss.so'))
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Installing locally')
        wheelfile = [f for f in os.listdir(os.path.join(currdir, 'dist')) if f.endswith('.whl')]
        os.system('pip install ' + os.path.join(currdir, 'dist', wheelfile[0]))

        sys.exit()

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and upload the package.'
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
        self.status('Build and upload wheel')
        try:
            self.status('Removing previous builds…')
            shutil.rmtree(os.path.join(currdir, 'dist'))
            shutil.rmtree(os.path.join(currdir, 'build'))
            shutil.rmtree(os.path.join(currdir, 'wheelhouse'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        '''
        self.status('Conform the Wheel to manylinux2010')
        namemodule = 'auditwheel'
        spec = importlib.util.find_spec(namemodule)
        if namemodule in sys.modules:
            print(f"{namemodule!r} already in sys.modules")
        elif spec is not None:
            # If you choose to perform the actual import ...
            module = importlib.util.module_from_spec(spec)
            sys.modules[namemodule] = module
            spec.loader.exec_module(module)
            print(f"{namemodule!r} has been imported")
        else:
            print(f"Installing the {namemodule!r} module")
            os.system('pip install ' + namemodule)
        os.system('auditwheel repair dist/merlinpy-' + VERSION + '-cp38-cp38-linux_x86_64.whl')
        '''

        self.status('Uploading the package to PyPI via Twine…')
        os.system('python -m twine upload --repository testpypi dist/*')

        self.status('Pushing git tags…')
        os.chdir(os.path.dirname(currdir))
        os.system('git tag merlinpy-v{0}'.format(VERSION))
        #os.system('git push --tags')

        sys.exit()

setup(
    name='merlinpy',
    version=VERSION,
    author="Kerstin Hammernik, Thomas Kuestner",
    author_email="merlin.midastum@gmail.com",
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
    include_package_data=INCLUDE_PACKAGE_DATA,
    package_data=PACKAGE_DATA,
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
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    cmdclass={
        'upload': UploadCommand,
        'sampling': SamplingCommand
    }
)
