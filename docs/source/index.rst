.. img-segm documentation master file, created by
   sphinx-quickstart on Wed Sep  1 10:28:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to img-segm's documentation!
====================================

Automatic pipeline for the segmentation of cancer areas on T2-weighted Magnetic Resonance Images of patient affected by colorectal cancer.

Colorectal cancer is a malignant neoplasm of the large intestine resulting from the uncontrolled proliferation of one of the cells making up the colorectal tract. 
Colorectal cancer is the second malignant tumor per number of deaths after the lung cancer and the third per number of new cases after the breast and lung cancer. 

In order to get information about diagnosis, therapy evaluation on colorectal cancer, analysis on radiological images can be performed through the application of dedicated algorithms.

In this scenario, the correct and fast identification of the cancer regions is a
fundamental task. 
Up to now this task is performed using manual or
semi-automatic techniques, which are time-consuming and
subjected to the operator expertise.

This project provides an automatic pipeline for the segmentation of
cancer areas on T2-weighted Magnetic Resonance Images of patient affected by colorectal cancer.




.. image :: ../../extras/imgs/11.png
    :width: 300
.. image :: ../../extras/imgs/11_cont.png
    :width: 300

**Example of segmentation** . **Left):** Original MRI scan of a patient affected by colorectal cancer. **Right):** Original MRI scan of a patient affected by colorectal cancer with identified tumor area.


Usage Example
=============

Once you have installed it, you can start to segment the images directly from your bash.
The input ``--dir`` is the path of the dir containing the DICOM series.
Please ensure that the folder contains only one series.
If the directory is a nested dir, the script will find automatically the sub-dir containing the DICOM series.

Quick start
-----------

.. code-block:: bash

   python -m MRIsegm --dir='/path/to/input/series/'

.. image :: ../../extras/imgs/example_quickstart.png



.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   installation
   usage
   MRIsegm



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
