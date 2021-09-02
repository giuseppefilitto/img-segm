Usage
=====

Once you have installed it, you can start to segment the images directly from your bash.
The input ``--dir`` is the path of the dir containing the DICOM series.
Please ensure that the folder contains only one series.
If the directory is a nested dir, the script will find automatically the sub-dir containing the DICOM series.

Quick start
-----------

.. code-block:: bash

   python -m MRIsegm --dir='/path/to/input/series/'

where: ``--dir`` is the path of the directory containing the DICOM series (required).

.. image :: ../../extras/imgs/Usage.png

Options
-------

``--model``

Name of the model's weights saved in the weights_ dir.

.. _weights: https://github.com/giuseppefilitto/img-segm/tree/main/data/models/weights

.. code-block:: bash

   python -m MRIsegm --dir='/path/to/input/series/' --model='model_name'

notes:
 *  ``model_name`` set as default: ``efficientnetb0_256_256_BTC=8_alpha3_OPT=Adam_LOSS=DiceBCEloss``
 * Remember to specify the name without ``_weights.h5``
 * you can also use your own model's weight saving the weights in the weights_ dir as ``model_name_weights.h5``. 
 * warning: You need to save also the architecture  as ``model_name.json`` file in the same dir.


``--mask``

When enabled plot the predicted binary [0,1] mask of each slice.

.. code-block:: bash

   python -m MRIsegm --dir='/path/to/input/series/' --mask

.. image :: ../../extras/imgs/usage_mask.png


``--density``

When enabled plot the predicted probability map between 0 and 1 of each slice over the original image.

.. code-block:: bash

   python -m MRIsegm --dir='/path/to/input/series/' --density

.. image :: ../../extras/imgs/usage_density.png


``--3D``

When enabled plot the a 3D mesh of the segmented areas.

.. code-block:: bash

   python -m MRIsegm --dir='/path/to/input/series/' --3D

.. image :: ../../extras/imgs/usage_3D.png