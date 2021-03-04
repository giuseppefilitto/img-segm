# %% importing modules
import numpy as np
import pydicom
import glob
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, IntSlider, ToggleButtons

# %% Define the image path and load the data
dir_path = '/Users/giuseppefilitto/Pazienti_anonym_sorted/BO1/T2AX'
files = glob.glob(dir_path + '/*.dcm')


# %% read the series as a slice and coverting to np.array


def read_slices(filename):
    name, ext = filename.split('.')

    if ext != 'dcm':
        raise ValueError('Input filename must be a DICOM file')

    slide = pydicom.dcmread(filename).pixel_array

    return slide


slice = [read_slices(f) for f in files]

slice = np.asarray(slice)
type(slice)
# %% Get the image shape and print it out

depth, height, width = slice.shape
print(
    f"The image object has the following dimensions: height: {height}, width:{width}, depth:{depth}")
# %% visualize the data

maxval = depth  # Select random layer number
i = np.random.randint(0, maxval)

print(f"Plotting Layer {i} of Image")
plt.imshow(slice[i, :, :], cmap='gray')
plt.axis('off')

# %% data exploration


def explore_slice(layer):
    plt.figure(figsize=(10, 5))
    plt.imshow(slice[layer, :, :], cmap='gray')
    plt.title('Explore Layers', fontsize=20)
    plt.axis('off')

    return layer


# Run the ipywidgets interact() function to explore the data
interact(explore_slice, layer=(0, slice.shape[0] - 1))
