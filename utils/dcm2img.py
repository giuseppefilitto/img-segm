import os
import pydicom
from PIL import Image  # PILLOW package
from tqdm import trange
import argparse


def parse_args():

    description = 'Converting DCM dirs in image series'

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--src', dest='src', required=True, type=str,
                        action='store', help='source directory')
    parser.add_argument('--dst', dest='dst', required=False, type=str,
                        action='store', help='Output directory')
    parser.add_argument('--patient', dest='patient', required=False, type=str,
                        action='store', help='patient name')

    args = parser.parse_args()

    return args


def rescale(img, max_value, min_value):
    '''
    Parameters
    ----------
    img :
        pixel_array image.
    max_value :
        pixel_array image max value.
    min_value :
        pixel_array image min value.

    Returns
    -------
        rescaled pixel_array image as uint8 type.
    '''
    return ((img.astype(float) - min_value) * (1. / (max_value - min_value)) * 255.).astype('uint8')


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
     converted .png images
    '''

    if not os.path.exists(src):
        raise ValueError("Path not found")

    if not os.path.exists(dst) or src == dst:
        dst = '_'.join((dst, 'frames'))
        os.makedirs(dst)

    dcm_list = []
    for root, _, files in os.walk(src):
        for file in files:
            if ".dcm" in file:
                dcm_list.append(os.path.join(root, file))

    slices = [read_dcm(f) for f in dcm_list]
    slices = [rescale(x, x.max(), x.min()) for x in slices]

    for i, im in enumerate(slices):

        im = Image.fromarray(im)

        file_path = dcm_list[i]
        filename = os.path.splitext(os.path.split(file_path)[1])[0]

        output_path = os.path.join(dst, filename + '.png')

        im.save(output_path)


def main():

    args = parse_args()

    if not os.path.isdir(args.src):
        raise ValueError('Incorrect directory given')

    if not args.dst:

        args.dst = '_'.join((args.src, 'frames'))

        if not os.path.isdir(args.dst):
            os.makedirs(args.dst)

    if args.patient:

        if args.patient == "all" or args.patient == "ALL":

            dirs = os.listdir(args.src)

            if ".DS_Store" in dirs:
                dirs.remove(".DS_Store")  # for mac user

            progress_bar = trange(len(dirs), desc="Convertion in progress")

            for item in dirs:
                input_ = os.path.join(args.src, item)

                list_ = []
                for root, dirs, files in os.walk(input_):
                    for file in files:
                        if ".dcm" in file:
                            list_.append(os.path.join(root, file))

                src_dir = [os.path.split(i)[0] for i in list_]
                src_dir = set(src_dir)  # to get unique values

                for dir_ in src_dir:

                    output = os.path.join(
                        args.dst, item, os.path.split(dir_)[1])

                    converter(dir_, output)

                progress_bar.update(1)

            progress_bar.close()

        else:

            input_ = os.path.join(args.src, args.patient)
            output = os.path.join(args.dst, args.patient)

            list = []
            for root, dirs, files in os.walk(input_):
                for file in files:
                    if ".dcm" in file:
                        list.append(os.path.join(root, file))

            src_dir = [os.path.split(i)[0] for i in list]
            src_dir = set(src_dir)  # to get unique values

            for dir in src_dir:

                output = os.path.join(
                    args.dst, args.patient, os.path.split(dir)[1])
                converter(dir, output)

    else:

        converter(args.src, args.dst)


if __name__ == '__main__':

    main()
