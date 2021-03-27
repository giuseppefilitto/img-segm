import numpy as np
import pydicom
import glob
from read_roi import read_roi_file
import matplotlib.pyplot as plt
import cv2


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def rescale(im, max, min):
    return ((im.astype(float) - min) * (1. / (max - min)) * 255.).astype('uint8')


def read_slices(filename):
    name, ext = filename.split('.')

    if ext != 'dcm':
        raise ValueError('Input filename must be a DICOM file')

    slide = pydicom.dcmread(filename).pixel_array

    return slide


def get_slice(dir_path):

    files = glob.glob(dir_path + '/*.dcm')

    # ordering as istance number
    z = [float(pydicom.read_file(f, force=True).get(
        "InstanceNumber", "0") - 1) for f in files]
    order = np.argsort(z)
    files = np.asarray(files)[order]

    slice = [read_slices(f) for f in files]

    Max = max([x.max() for x in slice])
    Min = min([x.min() for x in slice])

    slice = [rescale(x, Max, Min) for x in slice]

    slice = np.asarray(slice)

    return slice


def get_slice_info(slice):

    depth, height, width = slice.shape
    print(
        f"The image object has the following dimensions: depth:{depth}, height:{height}, width:{width}")


def _dict(dict_list):
    '''

    useful to get true_dict since roi is {name file : true_dict}.

    '''

    true_dict = []

    for i in dict_list:
        _dict = list(i.values())

        for j in _dict:
            keys = j.keys()
            vals = j.values()

            _dict = {key: val for key, val in zip(keys, vals)}
            true_dict.append(_dict)

    return true_dict


def get_rois(roi_path):

    rois_list = glob.glob(roi_path + '/*.roi')

    rois = [read_roi_file(roi) for roi in rois_list]
    rois = _dict(rois)

    # ordering dictionaries by positions and removing rois without x y coords
    rois = sorted(rois, key=lambda d: list(d.values())[-1])
    rois = list(filter(lambda d: d['type'] != 'composite', rois))

    return rois


def make_mask(slice, layer, rois):

    positions = [rois[i].get('position') - 1 for i in range(len(rois))]
    if not layer in positions:
        raise ValueError("no labels found!")

    else:

        background = np.zeros_like(slice[layer, :, :])

        roi = list(filter(lambda d: d['position'] == layer + 1, rois))

        x = [roi[i].get('x') for i in range(len(roi))]
        y = [roi[i].get('y') for i in range(len(roi))]

        points = []
        for i in range(len(x)):
            pts = np.array([(x, y) for(x, y) in zip(
                x[i], y[i])])
            points.append(pts)

        label = cv2.fillPoly(background, points, 255)

        return label


def mask_slice(slice, rois):

    masked_slice = np.zeros_like(slice)

    positions = [rois[i].get('position') - 1 for i in range(len(rois))]

    for layer in range(slice.shape[0]):

        if not layer in positions:
            masked_slice[layer, :, :] = 0
        else:
            masked_slice[layer, :, :] = make_mask(
                slice=slice, layer=layer, rois=rois)

    return masked_slice


def explore_roi(slice, layer, rois):

    # -1 to match slice
    positions = [rois[i].get('position') - 1 for i in range(len(rois))]

    if layer in positions:

        plt.figure(figsize=(12, 7), constrained_layout=True)
        plt.imshow(slice[layer, :, :], cmap='gray')
        plt.title(f'Exploring Layer {layer}', fontsize=20)
        plt.axis('off')

        roi = list(filter(lambda d: d['position'] == layer + 1, rois))

        x = [roi[i].get('x') for i in range(len(roi))]
        y = [roi[i].get('y') for i in range(len(roi))]

        for i in range(len(x)):

            plt.fill(x[i], y[i], edgecolor='r', fill=False)

    else:
        plt.figure(figsize=(12, 7))
        plt.imshow(slice[layer, :, :], cmap='gray')
        plt.title(f'Exploring Layer {layer}', fontsize=20)
        plt.axis('off')


def plot_random_layer(slice):

    maxval = slice.shape[0]
    # Select random layer number
    layer = np.random.randint(0, maxval)

    # figure
    explore_slice(slice=slice, layer=layer)


def explore_slice(slice, layer):
    plt.figure(figsize=(12, 7), constrained_layout=True)
    plt.imshow(slice[layer, :, :], cmap='gray')
    plt.title(f'Exploring Layer {layer}', fontsize=20)
    plt.axis('off')
