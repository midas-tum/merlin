#!/bin/bash
cd python
python setup.py build
cd ..
pip install ./python
pip install ./pytorch
pip install ./tensorflow
