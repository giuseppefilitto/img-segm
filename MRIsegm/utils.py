import numpy as np
import pydicom
import glob
from read_roi import read_roi_file
import matplotlib.pyplot as plt
import cv2


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
        f"The image object has the following dimensions: depth:{depth}, height: {height}, width:{width}")


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

    positions = []
    xs = []
    ys = []
    for i in range(len(rois)):
        position = rois[i]['position']
        x = rois[i]['x']
        y = rois[i]['y']

        x.append(x[0])
        y.append(y[0])

        # -1 to match slice layer
        positions.append(position - 1)
        xs.append(x)
        ys.append(y)

    return positions, xs, ys


def explore_roi(slice, layer, positions, xs, ys):

    if not layer in positions:
        print("No ROI found")
    else:
        plt.figure(figsize=(12, 7))
        plt.imshow(slice[layer, :, :], cmap='gray')
        plt.plot(xs[layer - positions[0]], ys[layer - positions[0]], color="red",
                 linestyle='dashed', linewidth=1)
        plt.title(f'Exploring Layer {layer}', fontsize=20)
        plt.axis('off')


def plot_random_layer(slice):

    maxval = slice.shape[0]
    # Select random layer number
    layer = np.random.randint(0, maxval)

    # figure
    plt.figure(figsize=(12, 7), constrained_layout=True)
    plt.imshow(slice[layer, :, :], cmap='gray')
    plt.title(f"Plotting Layer {layer}", fontsize=20)
    plt.axis('off')


def explore_slice(slice, layer):
    plt.figure(figsize=(12, 7), constrained_layout=True)
    plt.imshow(slice[layer, :, :], cmap='gray')
    plt.title(f'Exploring Layer {layer}', fontsize=20)
    plt.axis('off')


def explore_mask(slice, layer, positions, xs, ys):

    if not layer in positions:
        print("No ROI found")
    else:

        image = slice[layer, :, :].copy()

        pts = np.array([(x, y) for(x, y) in zip(
            xs[layer - positions[0]], ys[layer - positions[0]])])

        cv2.drawContours(image, [pts], -1, (255, 255, 255), -1)
        cv2.polylines(image, [pts], isClosed=True,
                      color=(255, 255, 255), thickness=1)
        mask = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)[1]

        # need opening to remove occasional white points
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # figure
        fig, ax = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)

        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title(" ROI Mask")
        ax[1].axis('off')

        ax[0].imshow(slice[layer, :, :], cmap="gray")
        ax[0].plot(xs[layer - positions[0]], ys[layer - positions[0]], color="red",
                   linestyle='dashed', linewidth=1)
        ax[0].set_title("ROI")
        ax[0].axis('off')

        fig.suptitle(f"Exploring layer: {layer}",  fontsize=20)


def explore_applied_mask(slice, layer, positions, xs, ys):

    if not layer in positions:
        print("No ROI found")
    else:

        image = slice[layer, :, :].copy()

        pts = np.array([(x, y) for(x, y) in zip(
            xs[layer - positions[0]], ys[layer - positions[0]])])

        cv2.drawContours(image, [pts], -1, (255, 255, 255), -1)
        cv2.polylines(image, [pts], isClosed=True,
                      color=(255, 255, 255), thickness=1)
        mask = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)[1]

        # need opening to remove occasional white points
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        masked_img = cv2.bitwise_and(
            slice[layer, :, :].copy(), image, mask=mask)

        # figure
        fig, ax = plt.subplots(1, 3, figsize=(12, 7), constrained_layout=True)

        ax[0].imshow(slice[layer, :, :], cmap="gray")
        ax[0].set_title("Original")
        ax[0].axis('off')

        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("ROI Mask")
        ax[1].axis('off')

        ax[2].imshow(masked_img, cmap="gray")
        ax[2].set_title("Applied ROI mask")
        ax[2].axis('off')
        fig.suptitle(f'Exploring layer {layer}', fontsize=20)
