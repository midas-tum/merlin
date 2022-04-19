.. MERLIN documentation master file, created by
   sphinx-quickstart on Wed Feb 23 15:33:15 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MERLIN's documentation!
==================================

Lore ipsum - add some info about merlin and define abbreviation

OVERVIEW
--------
The source files are organized as follows:

.. code-block::

    .
    ├── python/merlinpy          : python wrappers 
    ├── pytorch/merlinth         : pytorch wrappers
    └── tensorflow/merlintf      : tensorflow wrappers

.. toctree::
   :caption: USE MERLIN
   :titlesonly:
   :maxdepth: 1

   installation
   getting_started
   unittesting

.. toctree::
    :maxdepth: 2
    :caption: Python API

    api/merlinpy
    api/merlinth
    api/merlintf

.. toctree::
   :caption: About
   :titlesonly:
   :maxdepth: 1

   about

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
