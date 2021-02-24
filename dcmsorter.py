import os
import zipfile
import pydicom
import shutil
from pathlib import Path
from tqdm import trange


def sorter(input_path, output_path, patience):
    '''

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
        organized and sorted data.

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

            ds = pydicom.read_file(file, force=True)
            instanceNumber = str(ds.get("InstanceNumber", "0"))

            fileName = instanceNumber + ".dcm"

            studyfolder = Path(file).parts[-4]
            dst = os.path.join(output_path, patience, studyfolder)

            if not os.path.exists(dst):
                os.makedirs(dst)

            ds.save_as(os.path.join(dst, fileName))

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
