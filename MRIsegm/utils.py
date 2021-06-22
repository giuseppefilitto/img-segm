import numpy as np
import pydicom
import glob
from read_roi import read_roi_file
import matplotlib.pyplot as plt
import cv2


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def rescale(im, max, min):
    '''
    Rescale image in range (0,255)

    Parameters
    ----------
    im : array like
        image to be rescaled
    max : value
        max image value
    min : value
        min image value

    Returns
    -------
    rescaled image: array like
        rescaled input image as type uint 8
    '''
    rescaled_image = ((im.astype(float) - min) * (1. / (max - min)) * 255.).astype('uint8')
    return rescaled_image


def read_slices(filename):
    '''

    Read dicom file as pixel array

    Parameters
    ----------
    filename : str
        name of file.dcm

    Returns
    -------
    pix_arr : array
        dcm file as array

    Raises
    ------
    ValueError
        filename must be .dcm format
    '''
    name, ext = filename.split('.')

    if ext != 'dcm':
        raise ValueError('Input filename must be a DICOM file')

    pix_arr = pydicom.dcmread(filename).pixel_array

    return pix_arr


def get_slices(dir_path):
    '''

    Get full stack of slices from single dcm files ordered by "InstanceNumber" as a rescaled 3d array of shape: depth, height, width

    Parameters
    ----------
    dir_path : str
        directory of dcm slices

    Returns
    -------
    slices: array
         array of shape: depth, height, width , ordered by "InstanceNumber" , rescaled in range (0,255)
    '''

    files = glob.glob(dir_path + '/*.dcm')

    # ordering as istance number
    z = [float(pydicom.read_file(f, force=True).get(
        "InstanceNumber", "0") - 1) for f in files]
    order = np.argsort(z)
    files = np.asarray(files)[order]

    slices = [read_slices(f) for f in files]

    Max = max([x.max() for x in slices])
    Min = min([x.min() for x in slices])

    slices = [rescale(x, Max, Min) for x in slices]

    slices = np.asarray(slices)

    return slices


def get_slices_info(slices):
    '''
    Print depth, height, width of the input slices

    Parameters
    ----------
    slices : array-like

    '''
    depth, height, width = slices.shape
    print(f"The image object has the following dimensions: depth:{depth}, height:{height}, width:{width}")


def _dict(dict_list):
    '''

    Function to get true_dict from a dict of dict like {key : true_dict}

    Parameters
    ----------
    dict_list : list
        list of dicts

    Returns
    -------
    true_dict : list
        list of true_dict
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
    '''
    Get ImageJ rois from .roi files stored in roi_path

    Parameters
    ----------
    roi_path : str
        path of dir containing .roi files

    Returns
    -------
    roi: list
        list of roi dicts orderd by position number and without "type":composite
    '''

    rois_list = glob.glob(roi_path + '/*.roi')

    rois = [read_roi_file(roi) for roi in rois_list]
    rois = _dict(rois)

    # ordering dictionaries by positions and removing rois without x y coords
    rois = sorted(rois, key=lambda d: list(d.values())[-1])
    rois = list(filter(lambda d: d['type'] != 'composite', rois))

    return rois


def make_mask(slices, layer, rois):
    '''
    Generate mask of a given slice

    Parameters
    ----------
    slices : array
        array of shape depth, height, width
    layer : int
        value between (0, slices.shape[0])
    rois : list
        roi list

    Returns
    -------
    label : array
        return mask of a given slice. Pixels outside regions of interest are set to 0 (black), pixel inside regions of interest are set to 255 (white)
    Raises
    ------
    ValueError
        if there are no regions of interest: "no labels found!"
    '''

    positions = [rois[i].get('position') - 1 for i in range(len(rois))]
    if layer not in positions:
        raise ValueError("no labels found!")

    else:

        background = np.zeros_like(slices[layer, :, :])

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


def mask_slices(slices, rois):
    '''
    Make an array of shape depth, height, width, containing for each layer the proper mask.
    If no mask if found then masked_slices[layer, :, :] = 0

    Parameters
    ----------
    slices : array
        array of shape depth, height, width

    rois : list
        roi list

    Returns
    -------
    masked_slices : array
        array of shape: depth, height, width containing for each layer the proper mask
    '''

    masked_slices = np.zeros_like(slices)

    positions = [rois[i].get('position') - 1 for i in range(len(rois))]

    for layer in range(slices.shape[0]):

        if layer not in positions:
            masked_slices[layer, :, :] = 0
        else:
            masked_slices[layer, :, :] = make_mask(
                slices=slices, layer=layer, rois=rois)

    return masked_slices


def explore_roi(slices, layer, rois):
    '''
      Show the regions of interest contours from a given slice

      Parameters
      ----------
      slices : array
          array of shape depth, height, width
      layer : int
          value between (0, slices.shape[0])
      rois : list
          roi list

      '''

    # -1 to match slice
    positions = [rois[i].get('position') - 1 for i in range(len(rois))]

    if layer in positions:

        plt.figure(figsize=(12, 7), constrained_layout=True)
        plt.imshow(slices[layer, :, :], cmap='gray')
        plt.title(f'Exploring Slice {layer}', fontsize=20)
        plt.axis('off')

        roi = list(filter(lambda d: d['position'] == layer + 1, rois))

        x = [roi[i].get('x') for i in range(len(roi))]
        y = [roi[i].get('y') for i in range(len(roi))]

        for i in range(len(x)):

            plt.fill(x[i], y[i], edgecolor='r', fill=False)

    else:
        plt.figure(figsize=(12, 7))
        plt.imshow(slice[layer, :, :], cmap='gray')
        plt.title(f'Exploring Slice {layer}', fontsize=20)
        plt.axis('off')


def plot_random_layer(slices):
    '''

    Show figure of the random slice between (0, slices.shape[0])

    Parameters
    ----------
    slices : array
        array of shape depth, height, width
    '''

    maxval = slices.shape[0]
    # Select random layer number
    layer = np.random.randint(0, maxval)

    # figure
    explore_slices(slices=slices, layer=layer)


def explore_slices(slices, layer):
    '''
    Show figure of the given slice

    Parameters
    ----------
    slices : array
        array of shape depth, height, width
    layer : int
        value between (0, slice.shape[0])
    '''
    plt.figure(figsize=(12, 7), constrained_layout=True)
    plt.imshow(slices[layer, :, :], cmap='gray')
    plt.title(f'Exploring Slice {layer}', fontsize=20)
    plt.axis('off')


def display_image(img, figsize=(12, 7), **kwargs):
    '''
    Display greyscale image

    Parameters
    ----------
    img : image, array_like
        image to be displayed
    figsize : tuple, optional
        figsize arg of matplotlib module, by default (12, 7)
    '''

    plt.figure(figsize=(figsize), constrained_layout=True)
    plt.imshow(img, cmap='gray')

    if kwargs:
        title = kwargs.get('title')

        if kwargs.get('fontsize'):
            fontsize = kwargs.get('fontsize')
            plt.title(title, fontsize=fontsize)
        else:
            plt.title(title)


def display_images(display_list, figsize=(12, 8), **kwargs):
    '''
    Display a list of greyscale images

    Parameters
    ----------
    display_list : list
        list of images to be displayed
    figsize : tuple, optional
        figsize arg of matplotlib module, by default (12, 8)
    '''

    plt.figure(figsize=figsize)

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.imshow(display_list[i], cmap='gray')
        if kwargs.get('titles'):
            titles = kwargs.get('titles')
            if kwargs.get('fontsize'):
                fontsize = kwargs.get('fontsize')
                plt.title(titles[i], fontsize=fontsize)
            else:
                plt.title(titles[i])

    plt.show()
