#!/bin/bash
python ./python/setup.py build
pip install ./python
pip install ./pytorch
pip install ./tensorflow
