# MRI colon-rectal cancer segmentation


Colon-rectal cancer is a malignant neoplasm of the large intestine resulting from the uncontrolled proliferation of one of the cells making up the colon-rectal tract.  
In Western countries, colorectal cancer is the second largest malignant tumor after that of the breast in women and the third after that of the lung and prostate in men.
Therefore, radiomic analysis applied to radiological imaging is extremely useful in particular in identifying the regions of interest (ROI) through the application of dedicated radiomic algorithms / features. This process is also known as segmentation.  It can take place manually, semi-automatically and automatically.         
 The aim of this project is to perform automatic segmentation on Magnetic Resonance (MR) colon-rectal cancer images using ML methods.  



## Getting Started



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
