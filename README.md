# MERLIN - Machine Enhanced Reconstruction Learning and Interpretation Networks

This repository contains machine learning (ML) tools for both pytorch and tensorflow:
- `optim`: Custom optimizer such as BlockAdam
- `ddr`: Data-driven regularizer following [this repository](https://github.com/VLOGroup/tdv), extended to complex-valued layers

## Requirements
```
git clone https://github.com/midas-tum/optox.git
cd optox
```
follow build instructions on the github.

## Installation
```
chmod 700 install.sh
./install.sh
```

## Verification
Run unittests to ensure proper working of sub-modules
```
cd tensorflow
python3 -m unittest merlintf.ddr_complex.complex_pool
```