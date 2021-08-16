
# Utils

Collection of useful scripts to handle DICOM series and ROI files


## Getting Started

### Usage

All the scripts are executable by command line.

```bash
git clone https://github.com/giuseppefilitto/img-segm.git
cd img-segm/utils
```
#### dcmsorter

Move and sort DICOM files from nested dirs made like like ```patientID/EXAMINATION/DIRECTORY1/DIRECTORY2``` to ```patientID/EXAMINATION``` and save them into ```path/to/dst``` .

Simply run the following command from the bash:

```bash
python -m dcmsorter --src 'path/to/source' --dst 'path/to/dst' --patient 'patientID'
```
_where_:
* ```--src``` is the path of the database or where the nested dirs ```patientID/EXAMINATION/FOLDER1/FOLDER2``` containing the DICOM series are located 

*  ```--dst``` is the path where the sorted DICOM files in ```patientID/EXAMINATION``` will be saved

* ```--patient``` is the ```patientID``` of the database

_notes_:

* if ```--patient``` 'all' or 'ALL' the entire source folder will be sorted

* if no ```--dst``` is given then dst will simply be 'path/to/source_sorted'

#### dcm2img

Convert DICOM series to .png images.

Simply run the following command from the bash:

```bash
python -m dcm2img --src 'path/to/source' --dst 'path/to/dst' --patient 'patientID'
```
_where_:
* ```--src``` is the path of the database or where the nested dirs ```patientID/EXAMINATION/FOLDER1/FOLDER2``` containing the DICOM series are located 

*  ```--dst``` is the path where the sorted DICOM files in ```patientID/EXAMINATION``` will be saved as .png images

* ```--patient``` is the ```patientID``` of the database

_notes_:

* if ```--patient``` 'all' or 'ALL' the entire source folder will be sorted

* if no ```--dst``` is given then dst will simply be 'path/to/source_frames'


#### roimanager

Show the ROI saved as .roi over the images.

Simply run the following command from the bash:

```bash
python -m roimanager --src 'path/to/source' --patient 'patientID' --weight 'T2' (or 'DWI')
```
_where_:
* ```--src``` is the path of the database or where the nested dirs ```patientID/EXAMINATION/FOLDER1/FOLDER2``` containing the DICOM series are located 

* ```--patient``` is the ```patientID``` of the database

* ```--weight``` is the kind of examination performed (```T2``` or  ```DWI```)

_note_:

* before running **roimanager**, please ensure to have the ROI folder in ```path/to/source/patientID/T2ROI``` or ```path/to/source/patientID/DWIROI```
