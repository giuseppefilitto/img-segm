import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from MRIsegm.utils import get_slice
from MRIsegm.processing import denoise_slice, predict_slice, resize_slice, contour_slice
from MRIsegm.losses import DiceBCEloss, soft_dice_loss
from MRIsegm.metrics import dice_coef

from skimage.measure import marching_cubes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class IndexTracker:
    def __init__(self, ax, slice):
        self.ax = ax
        ax.set_xlabel('use scroll wheel to navigate layers')
        self.slice = slice
        self.index = 0
        try:
            self.im = ax.imshow(self.slice[self.index, :, :], cmap='gray')
        except:
            self.im = ax.imshow(self.slice[self.index, ...])
        self.update()

    def on_scroll(self, event):

        if event.button == 'up':
            self.index = (self.index + 1) % self.slice.shape[0]
        else:
            self.index = (self.index - 1) % self.slice.shape[0]
        self.update()

    def update(self):
        self.im.set_data(self.slice[self.index, ...])
        self.ax.set_title(f'Layer: {self.index} / {self.slice.shape[0] - 1}')
        self.im.axes.figure.canvas.draw()


class DensityIndexTracker:
    def __init__(self, ax, slice, predictions):
        self.ax = ax
        ax.set_xlabel('use scroll wheel to navigate layers')

        self.slice = slice
        self.predictions = predictions
        self.index = 0

        self.background = ax.imshow(
            self.slice[self.index, ...], cmap='gray',  vmin=0.0, vmax=1.0)
        predictions[predictions <= 0.05] = np.nan

        self.density = ax.imshow(
            predictions[self.index, ...], cmap='magma', vmin=0.0, vmax=1.0, alpha=0.8)

        cax = inset_axes(self.ax, width="5%", height="100%", loc='lower left',
                         bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=self.ax.transAxes,
                         borderpad=0)
        plt.colorbar(self.density, cax=cax)
        self.update()

    def on_scroll(self, event):

        if event.button == 'up':
            self.index = (self.index + 1) % self.slice.shape[0]
        else:
            self.index = (self.index - 1) % self.slice.shape[0]
        self.update()

    def update(self):
        self.background.set_data(self.slice[self.index, ...])
        self.density.set_data(self.predictions[self.index, ...])
        self.ax.set_title(f'Layer: {self.index} / {self.slice.shape[0] -1}')
        self.background.axes.figure.canvas.draw()
        self.density.axes.figure.canvas.draw()


def parse_args():

    description = 'Segmentation of MRI colorectal cancer'
    epilog = 'for more info check https://github.com/giuseppefilitto/img-segm'

    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('--dir', dest='dir', required=True, type=str,
                        action='store', help='DCM directory')
    parser.add_argument('--model', dest='model', required=False, type=str,
                        action='store', help='segmentation model (set in default)', default='UNET-L7-F8_128_128_BATCHNORM_OPT=adam_LOSS=DiceBCEloss')
    parser.add_argument('--mask', dest='mask', action='store_true',
                        help='plot predicted mask', default=False)
    parser.add_argument('--density', dest='density', action='store_true',
                        help='plot predicted map', default=False)
    parser.add_argument('--3D', dest='mesh3D', action='store_true',
                        help='enable 3D mesh plot', default=False)
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # loading dcm data
    dir_path = args.dir

    try:
        slice = get_slice(dir_path)
    except:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".dcm"):
                    dir_path = root
                    break

        print(f"dcm files from: {dir_path}")
        slice = get_slice(dir_path)

    print("[denoising...]")

    slice = denoise_slice(slice, 5)

    # model
    models_dir = 'data/models'
    model_path = os.path.join(models_dir, args.model + '.h5')

    print(f"[loading model --> {args.model}]")

    dependencies = {
        'DiceBCEloss': DiceBCEloss,
        'dice_coef': dice_coef,
        'soft_dice_loss': soft_dice_loss
    }

    model = tf.keras.models.load_model(model_path, custom_objects=dependencies)

    # image specs

    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

    if slice.shape[1:3] != IMG_SIZE:
        print(f"[images rescaled {slice.shape[1:3]} --> {IMG_SIZE}]")

    predicted = predict_slice(
        slice, model, IMAGE_HEIGHT, IMAGE_WIDTH, threshold=0.1)

    resized = resize_slice(slice, IMAGE_HEIGHT, IMAGE_WIDTH)
    countured = contour_slice(resized, predicted)

    fig, ax = plt.subplots(1, 1)

    tracker = IndexTracker(ax=ax, slice=countured)

    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()

    if args.mask:

        fig, ax = plt.subplots(1, 1)

        tracker = IndexTracker(ax=ax, slice=predicted)

        fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
        plt.show()

    if args.density:

        fig, ax = plt.subplots(1, 1)

        tracker = DensityIndexTracker(
            ax=ax, slice=resized, predictions=predicted.copy())

        fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
        plt.show()

    if args.mesh3D:

        predicted_sq = np.squeeze(predicted, axis=-1)
        verts, faces, _, _ = marching_cubes(predicted_sq)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlim(np.min(verts[:, 0]), np.max(verts[:, 0]))
        ax.set_ylim(np.min(verts[:, 1]), np.max(verts[:, 1]))
        ax.set_zlim(np.min(verts[:, 2]), np.max(verts[:, 2]))

        mesh = Poly3DCollection(
            verts[faces], edgecolors='teal', facecolors='orange', alpha=0.9)
        ax.add_collection3d(mesh)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    main()
