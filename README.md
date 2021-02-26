# img-utils

image processing utils.

**dcmsorter**

Script that allow to order a DCM dataset. If files such as .zip will be unzipped while images like .tiff will be copied from source path to destination path.


**dcm2img**

Script that allow to convert DCM folder to .png image series.

**roimanager**

Script that allow to see the ROIs of a patience over the original images in sequence.
## Getting Started

#### Prerequisites

The version of python used to run the scripts is: 3.9.1

the packages needed to run the scripts are the following:
```
- numpy
- pydicom
- PILLOW
- matplotlib
- opencv
```
#### Installation
```
git clone https://github.com/giuseppefilitto/img-utils
cd img-utils
```

### Usage

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