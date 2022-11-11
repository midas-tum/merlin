# Publish code

# PIP package
## Prerequisites

```
git clone https://github.com/NixOS/patchelf.git
cd patchelf
./bootstrap.sh
./configure
make 
make check
sudo make install
pip3 install auditwheel 
```

## Build
1. Build the wheel
```
python3 setup.py bdist_wheel
```
2. Repair the build to confirm with manylinux1
```
auditwheel repair dist/merlinpy-0.3.0-cp38-cp38-linux_x86_64.whl
```
3. Test the build upload
```
python -m twine upload --repository testpypi wheelhouse/*
```