# MERLIN - Machine Enhanced Reconstruction Learning and Interpretation Networks

This repository contains machine learning (ML) tools for pytorch, tensorflow, python.
Tensorflow `merlintf.keras` contains following folders
- `layers` basic building blocks, focusing on complex valued operations
    - Complex-valued convolutions
    - Complex-valued activations
    - FFT operations
    - Data consistency
    - ...
- `models` several layers are put together into networks for complex-valued processing (2-channel-real networks, complex networks)
    - Convolutional Neural Network
    - Fields-of-Experts regularizer
    - UNet
- `optimizers` contains custom optimizers


Pytorch `merlinth` contains following folders (not restructured yet):
- `optim`: Custom optimizer such as BlockAdam
- `layers`: Data-driven regularizer following [this repository](https://github.com/VLOGroup/tdv), extended to complex-valued layers and similar setup as layers in `merintf.keras`

**!!! Attention !!!** This package is work in progress and still under construction.
Major changes in structure will appear. Especially, tensorflow/keras
building blocks are not fully tested yet! If you experience any issues, if you have any feature requests or if you found any bugs, please let us know and raise an issue and/or
pull request in github :)

Please watch the `Issues` space and look for the latest updates regularly! :)
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
python3 -m unittest merlintf.keras.layers.complex_pool
```

## Common mistakes and best practices in writing own keras modules and layers
- `tf.keras.Model` cannot hold any trainable parameters. All trainable weights have to be defined in `tf.keras.layers.Layers`. Wrong implementation will cause weird behaviour when saving and re-loading the model weights!
- Do *not* define weights in the `__init__` function. Weights should be only 
created and initialized in the `def build(self, input_shape)` function of the `Layer`.
Wrong implementation will cause weird behaviour when saving and re-loading the model weights!
- The online documentation is a good orientation point to write own modules.
Make use of keras `Constraints` and `Initializers`.
