import os
import pydicom
import numpy as np
from PIL import Image  # PILLOW package
from tqdm import trange


def rescale(img, max, min):
    '''
    Parameters
    ----------
    img :
        pixel_array image.
    max :
        pixel_array image max value.
    min :
        pixel_array image min value.

    Returns
    -------
        rescaled pixel_array image as uint8 type.
    '''
    return ((img.astype(float) - min) * (1. / (max - min)) * 255.).astype('uint8')


def read_dcm(filename):
    '''
    Parameters
    ----------
    filename : str
        path of .dcm file.

    Returns
    -------
        .dcm file as pixel_array image.
    '''

    slide = pydicom.dcmread(filename).pixel_array

    return slide


def converter(src, dst):
    '''
    Parameters
    ----------
    src : path
        path of input dir.
    dst : path
        path of the output dir.

    Returns
    -------
     dir in dst with converted .png images
    '''

    src_path, dir_name = os.path.split(src)

    if not os.path.exists(src):
        raise ValueError("Path not found")

    if not os.path.exists(dst):
        dst = '_'.join((src, 'frames'))
        os.makedirs(dst)

    dcm_list = []
    for root, dirs, files in os.walk(src):
        for file in files:
            if ".dcm" in file:
                dcm_list.append(os.path.join(root, file))

    slices = [read_dcm(f) for f in dcm_list]
    slices = [rescale(x, x.max(), x.min()) for x in slices]

    for i, im in enumerate(slices):

        im = Image.fromarray(im)

        # uncomment for the original name
        # file_path = dcm_list[i]
        # filename = os.path.splitext(os.path.split(file_path)[1])[0]

        output_path = os.path.join(dst, '{0:01d}'.format(i + 1) + '.png')

        im.save(output_path)


def main(src):

    list = []
    for root, dirs, files in os.walk(src):
        for file in files:
            if ".dcm" in file:
                list.append(os.path.join(root, file))

    src_dir = [os.path.split(i)[0] for i in list]
    src_dir = set(src_dir)  # to get unique values

    progress_bar = trange(len(src_dir), desc="Progress")
    for dir in src_dir:

        converter(dir, dir + "frames")
        progress_bar.update(1)

    progress_bar.close()
    print("[done]")


if __name__ == '__main__':

    src = '/Users/giuseppefilitto/Pazienti_anonym_sorted/'

    main(src)
