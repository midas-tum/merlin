Installation
============

.. parsed-literal::
    chmod 700 install.sh
    ./install.sh


Documentation
***************

- Make sure to have Doxygen installed: ``sudo apt install doxygen``
- Make sure to have the requirements installed: ``pip install -r requirements.txt``
- Go to ``docs``
- Run ``make html``
- Go to ``_build/html``
- View the documentation: ``python -m http.server 8000``

Verification
============

Run unittests to ensure proper working of sub-modules

.. parsed-literal::
    python3 -m unittest discover -s merlinpy.test
    python3 -m unittest discover -s merlinth.test
    python3 -m unittest discover -s merlintf.test
