
## Utils

Collection of scripts useful for managing dicom file data


### Getting Started

#### Installation
```
git clone https://github.com/giuseppefilitto/img-segm
cd utils
```

#### Usage

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
python -m roimanager --src path/to/source --patience patienceID --weight T2 (or DWI)
```

_note_:

* before running **roimanager**, please ensure to have the ROIs folder as folder/of/images + "ROI"
