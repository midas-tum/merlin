from setuptools import setup, find_packages
from setuptools.dist import Distribution
import os
import sys
import shutil
import subprocess

VERSION = '1.0.0'
DESCRIPTION = 'Machine Enhanced Reconstruction Learning and Interpretation Networks (MERLIN)'

currdir = os.path.abspath(os.path.dirname(__file__))
compilepath = os.path.join(currdir, 'python', 'merlinpy', 'datapipeline', 'sampling', 'PoissonDisc')
so_target_dir = os.path.join(currdir, 'python', 'merlinpy', 'datapipeline', 'sampling')
python_dir = os.path.join(currdir, 'python')

# Build PoissonDisc sampling C++ extension
def build_sampling():
    if os.path.exists(compilepath):
        try:
            os.chdir(compilepath)
            res = subprocess.run([sys.executable, 'setup_VDPD.py', 'build'])
            os.chdir(currdir)
            build_dir = os.path.join(compilepath, 'build')
            if os.path.exists(build_dir):
                libdirs = [p for p in os.listdir(build_dir) if p.startswith('lib')]
                if libdirs:
                    full_libpath = os.path.join(build_dir, libdirs[0])
                    for fname in os.listdir(full_libpath):
                        src = os.path.join(full_libpath, fname)
                        if os.path.isfile(src):
                            shutil.copyfile(src, os.path.join(so_target_dir, fname))
                            if fname.endswith(('.so', '.pyd', '.dll')):
                                base_so = os.path.join(so_target_dir, 'VDPDGauss.so')
                                if not os.path.exists(base_so):
                                    shutil.copyfile(src, base_so)
        except Exception as e:
            print(f"Warning: Failed to compile sampling code: {e}")
        finally:
            os.chdir(currdir)

# Stage optox C++ / CUDA extension binary modules into python/ directory
def stage_optox():
    sp_dirs = [p for p in sys.path if 'site-packages' in p and os.path.exists(p)]
    for sp in sp_dirs:
        ext_src = os.path.join(sp, '_ext')
        if os.path.exists(ext_src):
            for pkg in ['_ext', 'optoth', 'optotf', 'optopy', 'optox']:
                pkg_src = os.path.join(sp, pkg)
                pkg_dst = os.path.join(python_dir, pkg)
                if os.path.exists(pkg_src) and not os.path.exists(pkg_dst):
                    shutil.copytree(pkg_src, pkg_dst)
            
            ext_dst = os.path.join(python_dir, '_ext')
            if os.path.exists(ext_dst):
                py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
                for f in os.listdir(ext_dst):
                    if f.endswith('.so'):
                        src = os.path.join(ext_dst, f)
                        base_dst = os.path.join(ext_dst, f.split('.')[0] + '.so')
                        if src != base_dst and not os.path.exists(base_dst):
                            shutil.copyfile(src, base_dst)
                        for ver in ['38', '39', '310', '311', '312', '313', py_ver]:
                            ver_dst = os.path.join(ext_dst, f.replace('-39-', f'-{ver}-'))
                            if src != ver_dst and not os.path.exists(ver_dst):
                                shutil.copyfile(src, ver_dst)
            break

# Always run sampling compilation and optox staging
build_sampling()
stage_optox()

if len(sys.argv) > 1 and sys.argv[1] == 'sampling':
    sys.argv.pop(1)

try:
    with open(os.path.join(currdir, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

# Discover all packages from python/, pytorch/, and tensorflow/
pkgs = (find_packages(where='python') + 
        find_packages(where='pytorch') + 
        find_packages(where='tensorflow'))

pkg_dir = {
    '': 'python',
    'merlinth': 'pytorch/merlinth',
    'merlintf': 'tensorflow/merlintf',
}

setup(
    name='merlin',
    version=VERSION,
    author="Kerstin Hammernik, Thomas Kuestner",
    author_email="merlin.midastum@gmail.com",
    packages=pkgs,
    package_dir=pkg_dir,
    include_package_data=True,
    package_data={
        "": ["*.so", "*.pyd", "*.dll"],
        "merlinpy.datapipeline.sampling": ["*.so", "*.pyd", "*.dll", "*"],
        "_ext": ["*.so", "*.pyd", "*.dll", "*"],
        "optoth": ["*.so", "*.pyd", "*.dll", "*"],
        "optotf": ["*.so", "*.pyd", "*.dll", "*"],
        "optopy": ["*.so", "*.pyd", "*.dll", "*"],
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
)
