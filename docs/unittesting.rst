Unittesting
===========

Python Wrappers
***************

To run all unittests for ``merlinpy`` use

.. code-block::

    python -m unittest discover merlinpy.test


Tensorflow/Keras Wrappers
******************************


To run all unittests for ``merlintf`` use

.. code-block::

    python -m unittest discover merlintf.test

To run a single unittest for, e.g., *complex activations*, use

.. code-block::

    python -m unittest merlintf.test.test_layers_complex_act

Torch Wrappers
***************


To run all unittests for ``merlinth`` use

.. code-block::

    python -m unittest discover merlinth.test

To run a single unittest for, e.g., *complex activations*, use

.. code-block::

    python -m unittest merlinth.test.test_layers_complex_act