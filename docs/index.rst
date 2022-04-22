.. MERLIN documentation master file, created by
   sphinx-quickstart on Wed Feb 23 15:33:15 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MERLIN's documentation!
==================================

Machine Enhanced Reconstruction Learning and Interpretation Networks (MERLIN) contains machine learning (ML) tools for
PyTorch, TensorFlow and Python in three modules: ``merlinpy``, ``merlinth``, ``merlintf``

OVERVIEW
--------
The source files are organized as follows:

.. code-block::

    .
    ├── python/merlinpy          : Python wrappers
    ├── pytorch/merlinth         : Pytorch wrappers
    └── tensorflow/merlintf      : Tensorflow wrappers



ACKNOWLEDGEMENT
---------------
If you use this code, please cite

.. parsed-literal::
   @inproceedings{HammernikKuestner2022,
     title={Machine Enhanced Reconstruction Learning and Interpretation Networks (MERLIN)},
     author={Hammernik, K. and K{\"u}stner, T.},
     booktitle={Proceedings of the International Society for Magnetic Resonance in Medicine (ISMRM)},
     year={2022}
   }


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
