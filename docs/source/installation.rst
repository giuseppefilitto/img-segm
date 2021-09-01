Installation
=================

Supported python versions :
|python version|

First of all ensure to have the right python version installed.

This project use Tensorflow_, opencv-python_, numpy_: see the
requirements_ for more information.

Installation
------------

First of all, ensure to have the right python version and the package for the
lung extraction correctly installed

To install this package first of all you have to clone the repositories from GitHub:

.. code-block:: bash

  git clone https://github.com/giuseppefilitto/img-segm.git
  cd img-segm

The installation is managed by setup.py, which will install also the full dependency.
So, from the segmentation folder simply run

.. code-block:: bash

  pip install -r requirements.txt
  python setup.py install

Testing
-------

Testing routines use pytest_ and hypothesis_ packages. 
A full set of test is provided in testing_ directory.
You can run the full list of test with:

.. code-block:: bash

  python -m pytest

.. |python version| image:: https://img.shields.io/badge/python-3.5|3.6|3.7|3.8-blue.svg
.. _Tensorflow: https://www.tensorflow.org
.. _opencv-python: https://opencv.org
.. _numpy: https://numpy.org
.. _requirements: https://github.com/giuseppefilitto/img-segm/blob/main/requirements.txt
.. _pytest: https://docs.pytest.org/en/6.2.x/
.. _hypothesis: https://hypothesis.readthedocs.io/en/latest/
.. _testing: https://github.com/giuseppefilitto/img-segm/blob/master/testing

