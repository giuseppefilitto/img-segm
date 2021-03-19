# img-segm

The aim of this project is to segment MRI colon-rectal cancer images using ML methods.

### utils

folder of utils scripts to handle DCM files

##### dcmsorter

Script that allow to order a DCM dataset. If files such as .zip will be unzipped while images like .tiff will be copied from source path to destination path.


##### dcm2img

Script that allow to convert DCM folder to .png image series.

##### roimanager

Script that allow to see the ROIs of a patience over the original images in sequence.
### notebooks
**dicomexplore**

Jupyter notebook made for visualize and interact with DCM files. 

### extras

Extra materials and examples 

## Getting Started

#### Prerequisites

the main packages needed to run the scripts are the following:
```
- numpy
- pydicom
- PILLOW
- matplotlib
- opencv
- read_roi 
- scikit-learn
- skimage
- ipywidgets
```
#### Installation
```
git clone https://github.com/giuseppefilitto/img-segm
```

### Usage



#### Utils

All the scripts are executable by command line. For a better usage it is recommended to execute all the scripts in the following order:
##### dcmsorter

simply run the following command from the bash or PowerShell:

```
python -m dcmsorter --src path/to/source --dst path/to/dest --patience patienceID
```

_notes_:

* if --patience all or ALL the entire source folder will be sorted

* if no --dst is given then dst will simply be source + "_sorted"

##### dcm2img

simply run the following command from the bash or PowerShell:

```
python -m dcm2img --src path/to/source --dst path/to/dest --patience patienceID
```
_notes_:

* if --patience all or ALL the entire source folder will be converted

* if no --dst is given then dst will simply be source + "_frames"


##### roimanager

simply run the following command from the bash or PowerShell:

```
python -m roimanager --src path/to/source --dst path/to/dest --patience patienceID --weight T2 (or DWI)
```

_note_:

* before running **roimanager**, please ensure to have the ROIs folder as folder/of/images + "ROI"

#### notebooks
Run the notebook with jupyter in order to use the ipywidgets for a better experience.


