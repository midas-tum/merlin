# MERLIN - Machine Enhanced Reconstruction Learning and Interpretation Networks
<!-- markdownlint-disable -->
<p align="center">
<a href="http://merlin-mri.rtfd.io/">
<img src="https://github.com/midas-tum/merlin/blob/3486af88e5370d49074883bc8cd1816687b8915f/notebooks/fig/MERLIN_logo.png" width="150" alt="MERLIN logo"> 
</a>
</p>
<!-- markdownlint-restore -->

This repository contains machine learning (ML) tools for PyTorch, TensorFlow and Python in three modules:
- `merlinth`: ML extensions to PyTorch
- `merlintf`: ML extensions to TensorFlow
- `merlinpy`: ML extensions to Python

If you use this code, please cite
```
@inproceedings{HammernikKuestner2022,
  title={Machine Enhanced Reconstruction Learning and Interpretation Networks (MERLIN)},
  author={Hammernik, K. and K{\"u}stner, T.},
  booktitle={Proceedings of the International Society for Magnetic Resonance in Medicine (ISMRM)},
  year={2022}
}
```

## Requirements
```
git clone https://github.com/midas-tum/optox.git
cd optox
python3 install.py --python --pytorch --tensorflow --gpunufft
```

follow build instructions on the github.

## Installation
### PyPi 
```
pip3 install merlinpy-mri merlinth-mri merlintf-mri
```
In case you want to use the sampling codes (C++), please use the direct way installation below for direct compilation according to your system setup. 
### Direct way

```
git clone https://github.com/midas-tum/merlin.git
python3 install.py
```

## Verification
Run unittests to ensure proper working of sub-modules
```
python3 -m unittest discover -s merlinpy.test
python3 -m unittest discover -s merlinth.test
python3 -m unittest discover -s merlintf.test
```

## Contents
**!!! Attention !!!** This package is work in progress and still under construction.
Major changes in structure will appear. If you experience any issues, if you have any feature requests or if you found any bugs, please let us know and raise an issue and/or pull request in github :)

Please watch the `Issues` space and look for the latest updates regularly! :)
### merlinth
```
merlinth
    |--- layers     # Data-driven regularizer following (https://github.com/VLOGroup/tdv), extended to complex-valued layers and similar setup as layers in `merlintf.keras`
        |-- Complex-valued convolutions
        |-- Complex-valued activations
        |-- Complex-valued pooling
        |-- Complex-valued normalization
        |-- FFT operations
        |-- Data consistency
        |-- ...
    |-- losses     # Common and custom loss functions
    |-- models     # Model zoo
        |-- Fields-of-Experts (FOE) regularizer
        |-- Total deep variation (TDV) regularizer
        |-- UNet
    |-- optim      # Custom optimizers such as BlockAdam
```

### merlintf
```
merlintf
    |-- keras
        |-- layers      # basic building blocks, focusing on complex valued operations
            |-- Complex-valued convolutions
            |-- Complex-valued activations
            |-- Complex-valued pooling
            |-- Complex-valued normalization
            |-- FFT operations
            |-- Data consistency
            |-- ...
        |-- models      # several layers are put together into networks for complex-valued processing (2-channel-real networks, complex networks)
            |-- Convolutional Neural Network
            |-- Fields-of-Experts (FOE) regularizer
            |-- Total deep variation (TDV) regularizer
            |-- UNet
        |-- optimizers       # custom optimizers    
    |-- optim                # custom optimizers
```

### merlinpy
```
merlinpy
    |-- datapipeline        # collection of datapipelines and transform functions
        |-- sampling        # subsampling codes and sampling trajectories
    |-- fastmri             # dataloader and processing related to fastMRI database
    |-- losses              # losses/metrics
    |-- recon               # conventional reconstructions
    |-- wandb               # logging via wandb.ai
```

## Common mistakes and best practices 
### writing own keras modules and layers
- `tf.keras.Model` cannot hold any trainable parameters. All trainable weights have to be defined in `tf.keras.layers.Layers`. Wrong implementation will cause weird behaviour when saving and re-loading the model weights!
- Do *not* define weights in the `__init__` function. Weights should be only 
created and initialized in the `def build(self, input_shape)` function of the `Layer`.
Wrong implementation will cause weird behaviour when saving and re-loading the model weights!
- The online documentation is a good orientation point to write own modules.
Make use of keras `Constraints` and `Initializers`.
