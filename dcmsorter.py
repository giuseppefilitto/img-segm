import os
import zipfile
import pydicom
import shutil
from tqdm import trange


def clean_text(str):
    '''clean and standardize text descriptions.

    Parameters
    ----------
    str : string
        text to be cleaned and standardized.

    Returns
    -------
        standardized and cleaned string

    '''
    forbidden_symbols = ["*", ".", ",", "\"", "\\", "/", "|", "[", "]", ":", ";", " "]
    for symbol in forbidden_symbols:
        str = str.replace(symbol, "_")  # replace everything with an underscore
    return str.lower()


def sorter(input_path, output_path, patience):
    '''Sorting function for DICOM file by associated information such as the study description and the performed modality.

    Parameters
    ----------
    input_path : str
        path of source folder.
    output_path : str
        path of destination folder.
    patience: str
        ID of the patience

    Returns
    -------
        organized and sorted data by patience, study description, seires description from the input_path to the output_path .

    '''
    if not os.path.exists(input_path):
        raise ValueError("Path not found")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    unsortedList = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if ".dcm" in file:
                unsortedList.append(os.path.join(root, file))
            elif ".zip" in file:
                unsortedList.append(os.path.join(root, file))
            elif ".tiff" in file:
                unsortedList.append(os.path.join(root, file))

    if len(unsortedList) == 0:
        raise ValueError("No files found")

    for file in unsortedList:
        if ".dcm" in file:
            # read the dcm file
            ds = pydicom.read_file(file, force=True)

            # get study, and series information

            studyDescription = clean_text(ds.get("StudyDescription", "NA"))
            seriesDescription = clean_text(ds.get("SeriesDescription", "NA"))

            # generate new, standardized file name

            instanceNumber = str(ds.get("InstanceNumber", "0"))
            fileName = patience + "_" + seriesDescription + "_" + instanceNumber + ".dcm"

            # uncompress files
            try:
                ds.decompress()
            except:
                print('an instance in file   %s - %s" could not be decompressed. exiting.' %
                      (studyDescription, seriesDescription))

            # save files to a 2-tier nested folder structure

            if not os.path.exists(os.path.join(output_path, patience, studyDescription)):
                os.makedirs(os.path.join(output_path, patience, studyDescription))

            if not os.path.exists(os.path.join(output_path, patience, studyDescription, seriesDescription)):
                os.makedirs(os.path.join(output_path, patience,
                                         studyDescription, seriesDescription))

            ds.save_as(os.path.join(output_path, patience,
                                    studyDescription, seriesDescription, fileName))
        if ".zip" in file:
            path_to_roi = os.path.join(output_path, patience,
                                       os.path.splitext(os.path.split(file)[1])[0])

            if not os.path.exists(path_to_roi):
                os.makedirs(path_to_roi)
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(path_to_roi)
            zip_ref.close()
        if ".tiff" in file:
            path_to_tiff = os.path.join(output_path, patience,
                                        os.path.split(os.path.split(file)[0])[1])
            if not os.path.exists(path_to_tiff):
                os.makedirs(path_to_tiff)
            shutil.copy(file, path_to_tiff)


def main(src, dst):

    dirs = os.listdir(src)

    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")

    progress_bar = trange(len(dirs), desc="Sorting in progress")

    for item in dirs:
        sorter(os.path.join(src, item), dst, patience=item)
        progress_bar.update(1)

    progress_bar.close()
    print("[done]")


if __name__ == '__main__':

    src = '/Users/giuseppefilitto/Pazienti_anonym'
    dst = src + '_sorted'

    main(src, dst)
