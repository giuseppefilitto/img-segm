# %% #! importing modules


import numpy as np
import pydicom
import glob
import matplotlib.pyplot as plt
from read_roi import read_roi_file
from ipywidgets import interact, interactive, IntSlider, ToggleButtons

# %% #! Define the image path and load the data


dir_path = '/Users/giuseppefilitto/Pazienti_anonym_sorted/BO11/T2AX'
files = glob.glob(dir_path + '/*.dcm')


# %% # ! read the series as a slice and coverting to np.array


def read_slices(filename):
    name, ext = filename.split('.')

    if ext != 'dcm':
        raise ValueError('Input filename must be a DICOM file')

    slide = pydicom.dcmread(filename).pixel_array

    return slide


# ordering as istance number
z = [float(pydicom.read_file(f, force=True).get(
    "InstanceNumber", "0") - 1) for f in files]
order = np.argsort(z)
files = np.asarray(files)[order]

slice = [read_slices(f) for f in files]
slice = np.asarray(slice)

# %% #! Get the image shape and print it out

depth, height, width = slice.shape
print(
    f"The image object has the following dimensions: height: {height}, width:{width}, depth:{depth}")
# %% #! visualize the data


maxval = depth  # Select random layer number
i = np.random.randint(0, maxval)

print(f"Plotting Layer {i} of Image")
plt.imshow(slice[i, :, :], cmap='gray')
plt.axis('off')

# %% #! data exploration


def explore_slice(layer):
    plt.figure(figsize=(10, 5))
    plt.imshow(slice[layer, :, :], cmap='gray')
    plt.title('Explore Layers', fontsize=20)
    plt.axis('off')

    return layer


# ! Run the ipywidgets interact() function to explore the data
interact(explore_slice, layer=(0, slice.shape[0] - 1))

# %% #! ROIs


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


roi_path = '/Users/giuseppefilitto/Pazienti_anonym_sorted/BO11/T2ROI'
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

    positions.append(position)
    xs.append(x)
    ys.append(y)


def explore_roi(layer):
    plt.figure(figsize=(10, 5))
    plt.imshow(slice[layer, :, :], cmap='gray')
    if layer in positions:
        plt.plot(xs[layer - positions[0]], ys[layer - positions[0]], color="red",
                 linestyle='dashed', linewidth=1)
    plt.title(f'Explore Layer {layer}', fontsize=20)
    plt.axis('off')

    return layer


# %% # ! INTERACTIVE ROI


interact(explore_roi, layer=(0, slice.shape[0] - 1))
