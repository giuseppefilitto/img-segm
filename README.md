| **Author**  | **Project** |  **Build Status** | **License** | **Code Quality** | **Coverage** |
|:------------:|:-----------:|:-----------------:|:-----------:|:----------------:|:------------:|
| [**G. Filitto**](https://github.com/giuseppefilitto) | **MRI colorectal cancer segmentation** | **Linux** : ![linux](https://img.shields.io/travis/giuseppefilitto/img-segm) | ![license](https://img.shields.io/github/license/giuseppefilitto/img-segm?)| **Codacy** : [![Codacy Badge]()]() <br/> **Codebeat** : [![CODEBEAT]()]() | [![codecov](https://codecov.io/gh/giuseppefilitto/img-segm/branch/main/graph/badge.svg?token=2POF72SN06)](https://codecov.io/gh/giuseppefilitto/img-segm) |



[![GitHub stars](https://img.shields.io/github/stars/giuseppefilitto/img-segm?style=social)](https://github.com/giuseppefilitto/img-segm/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/giuseppefilitto/img-segm.svg?label=Watch&style=social)](https://github.com/giuseppegilitto/img-segm/watchers)


# img-segm

This package allows to segment cancer regions on T2-weighted Magnetic Resonance Images (MRI) of patients affected by colorectal cancer.
The segmentation approach is based on Convolutional Neural Networks (CNNs) like U-Net.
This package provides a series of modules to visualize, pre-process the DICOM files and to train a U-Net model.

1. [Overview](#Overview)
2. [Installation](#Installation)
3. [Usage](#Usage)
4. [Authors](#Authors)
5. [Citation](#Citation)


## Overview

Colorectal cancer is a malignant neoplasm of the large intestine resulting from the uncontrolled proliferation of one of the cells making up the colorectal tract. 
In Western countries, colorectal cancer is the second largest malignant tumor after that of the breast in women and the third after that of the lung and prostate in men. 
Risk factors for this kind of cancer include colon polyps, long-standing ulcerative colitis, diabetes II and genetic history (HNPCC or Lynch syndrome). 
In order to get information about diagnosis, therapy evaluation on colorectal cancer, radiomic analysis can be performed on radiological images through the application of dedicated radiomic algorithms.

In this scenario, the correct and fast identification of the cancer regions is a
fundamental task. 
Up to now this task is performed using manual or
semi-automatic techniques, which are time-consuming and
subjected to the operator expertise.

This project provides an automatic pipeline for the segmentation of
cancer areas on T2-weighted Magnetic Resonance Images of patient affected by colorectal cancer.
The segmentation is achieved with a Convolutional Neural Network like U-Net.

## Installation
First, clone the git repository and change directory:

```bash
git clone https://github.com/giuseppefilitto/img-segm.git
cd img-segm
```

Then, pip-install the requirements and run the setup script:
```bash
pip install -r requirements.txt
python setup.py install
```
### Testing

Testing routines use ```PyTest``` and ```Hypothesis``` packages. 
A full set of test is provided in [testing](https://github.com/giuseppefilitto/img-segm/blob/master/testing) directory.
You can run the full list of test with:

```bash
python -m pytest
```
## Usage
Once you have installed it, you can start to segment the images directly from your bash.
The input **dir** is a DICOM series, pass the path to the directory containing
the series files.
Please ensure that the folder contains only one series.
If the directory is a subfolder of more than one directory, the script will find automatically the subfolder containing the DICOM series.

### Quick Start

```bash
   python -m MRIsegm --dir='/path/to/input/series/'  
```
### Options

#### mask

When enabled plot the segmented mask, between 0 and 1, of each slice
```bash
   python -m MRIsegm --dir='/path/to/input/series/'  --mask
```
#### density

When enabled plot the the probability map between 0 and 1 of each slice over the original image
```bash
   python -m MRIsegm --dir='/path/to/input/series/'  --density
```

#### 3D mesh plot

When enabled plot the a 3D mesh plot of the segmented areas
```bash
   python -m MRIsegm --dir='/path/to/input/series/'  --3D
```
### Pipeline

The workflow of this project is available under the [notebooks](https://github.com/giuseppefilitto/img-segm/blob/master/notebooks) folder.




## Author
* <img src="https://avatars.githubusercontent.com/u/61703705?v=4" width="25px;"/> **Giuseppe Filitto** [git](https://github.com/giuseppefilitto)

### Citation

If you have found `img-segm` helpful in your research, please
consider citing this project

```BibTeX
@misc{MRI colorectal cancer segmentation,
  author = {Filitto, Giuseppe},
  title = {MRI colorectal cancer segmentation},
  year = {2021},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/giuseppefilitto/img-segm}},
}
```