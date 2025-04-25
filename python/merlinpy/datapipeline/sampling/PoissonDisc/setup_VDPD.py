#!/usr/bin/env python3
# encoding: utf-8

# to compile: python3 setup_VDPD.py build

from distutils.core import setup, Extension

VDPD_module = Extension('VDPD', sources = ['InterfacePythonCpp.cpp'], extra_compile_args=['-std=c++11'])

setup(name='VDPD',
      version='1.0',
      description='generate VD-PDsampling',
      author = 'Thomas Kuestner',
      author_email = 'thomas.kuestner@kcl.ac.uk',
      ext_modules=[VDPD_module])