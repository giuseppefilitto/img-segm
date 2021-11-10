import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from segmentation_models.losses import DiceLoss, BinaryFocalLoss

from MRIsegm.utils import get_slices
from MRIsegm.processing import pre_processing_data, predict_images, contour_slices
from MRIsegm.metrics import dice_coef

from skimage.measure import marching_cubes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class IndexTracker:
    def __init__(self, ax, slices):
        '''

        Index Tracker class for plotting using scroll wheel

        Parameters
        ----------
        ax : Axes object
            matplotlib axes
        slices : array
            stack of slices
        '''
        self.ax = ax
        ax.set_xlabel('use scroll wheel to navigate slices')
        self.slices = slices
        self.index = 0

        self.im = ax.imshow(self.slices[self.index, ...], cmap='gray', vmin=0.0, vmax=1.0)

        self.update()

    def on_scroll(self, event):

        if event.button == 'up':
            self.index = (self.index + 1) % self.slices.shape[0]
        else:
            self.index = (self.index - 1) % self.slices.shape[0]
        self.update()

    def update(self):
        self.im.set_data(self.slices[self.index, ...])
        self.ax.set_title(f'Slice: {self.index} / {self.slices.shape[0] - 1}')
        self.im.axes.figure.canvas.draw()


class DensityIndexTracker:
    def __init__(self, ax, slices, predictions):
        '''

        Index Tracker class for plotting images and the overlapping predictions using scroll wheel

        Parameters
        ----------
        ax : Axes object
            matplotlib axes
        slices : array
            stack of slices
        predictions : array
            stack of corresponding predicted slices
        '''
        self.ax = ax
        ax.set_xlabel('use scroll wheel to navigate slices')

        self.slices = slices
        self.predictions = predictions
        self.index = 0

        self.background = ax.imshow(
            self.slices[self.index, ...], cmap='gray', vmin=0.0, vmax=1.0)

        predictions[predictions <= 0.005] = np.nan
        self.density = ax.imshow(
            predictions[self.index, ...], cmap='RdYlGn', vmin=0.0, vmax=1.0, alpha=0.7)

        cax = inset_axes(self.ax, width="5%", height="100%", loc='lower left',
                         bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=self.ax.transAxes,
                         borderpad=0)
        plt.colorbar(self.density, cax=cax)
        self.update()

    def on_scroll(self, event):

        if event.button == 'up':
            self.index = (self.index + 1) % self.slices.shape[0]
        else:
            self.index = (self.index - 1) % self.slices.shape[0]
        self.update()

    def update(self):
        self.background.set_data(self.slices[self.index, ...])
        self.density.set_data(self.predictions[self.index, ...])
        self.ax.set_title(f'Slice: {self.index} / {self.slices.shape[0] -1}')
        self.background.axes.figure.canvas.draw()
        self.density.axes.figure.canvas.draw()


def parse_args():

    description = 'Segmentation of MRI colorectal cancer'
    epilog = 'for more info check https://github.com/giuseppefilitto/img-segm'

    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('--dir', dest='dir', required=True, type=str,
                        action='store', help='DCM directory')
    parser.add_argument('--model', dest='model', required=False, type=str,
                        action='store', help='segmentation model (set in default)', default='efficientnetb0_BTC=4_full_150E_OPT=adam_LOSS=dice_loss_plus_1binary_focal_loss')
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
        slices = get_slices(dir_path, uint8=False)
    except:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".dcm"):
                    dir_path = root
                    break

        print(f"dcm files from: {dir_path}")
        slices = get_slices(dir_path, uint8=False)

    print("[pre-processing data...]")

    pre_processed = pre_processing_data(slices)

    # model
    models_dir = 'data/models'

    try:

        model_path = os.path.join(models_dir, args.model + '.h5')

        print(f"[loading model --> {args.model}]")

        dice_loss = DiceLoss()
        focal_loss = BinaryFocalLoss()
        loss = dice_loss + (1 * focal_loss)

        dependencies = {
            'dice_coef': dice_coef,
            'FixedDropout': tf.keras.layers.Dropout(0.2),
            'dice_loss': dice_loss,
            'dice_loss_plus_1binary_focal_loss': loss

        }

        model = tf.keras.models.load_model(model_path, custom_objects=dependencies)

    except:

        weights_dir = 'data/models/weights'
        model_architecture = weights_dir + '/' + args.model + '.json'
        model_weights = weights_dir + '/' + args.model + '_weights.h5'

        import json
        from tensorflow.keras.models import model_from_json

        with open(model_architecture) as json_file:
            data = json.load(json_file)

        model = model_from_json(data)
        model.load_weights(model_weights)

        optimizer = 'adam'

        dice_loss = DiceLoss()
        focal_loss = BinaryFocalLoss()
        loss = dice_loss + (1 * focal_loss)

        metrics = [dice_coef]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    predicted = predict_images(pre_processed, model)

    contoured = contour_slices(pre_processed, predicted)


    if args.mask:
        pred = predicted.copy()
        for i in range(pred.shape[0]):
            pred[i, ...] = np.where(pred[i, ...] >= 0.1, 1, 0)

        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax=ax, slices=pred)

        fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
        plt.show()
        exit(1)

    if args.density:

        fig, ax = plt.subplots(1, 1)

        tracker = DensityIndexTracker(
            ax=ax, slices=pre_processed, predictions=predicted.copy())

        fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
        plt.show()
        exit(1)

    if args.mesh3D:
        from matplotlib.colors import LightSource

        predicted_sq = np.squeeze(predicted, axis=-1)
        verts, faces, normals, _ = marching_cubes(predicted_sq)
        ls = LightSource(azdeg=225.0, altdeg=45.0)

        normalsarray = np.array([np.array((np.sum(normals[face[:], 0] / 3), np.sum(normals[face[:], 1] / 3), np.sum(normals[face[:], 2] / 3)) / np.sqrt(np.sum(normals[face[:], 0] / 3)**2 + np.sum(normals[face[:], 1] / 3)**2 + np.sum(normals[face[:], 2] / 3)**2)) for face in faces])

        min = np.min(ls.shade_normals(normalsarray, fraction=1.0))  # min shade value
        max = np.max(ls.shade_normals(normalsarray, fraction=1.0))  # max shade value
        diff = max - min
        newMin = 0.3
        newMax = 0.95
        newdiff = newMax - newMin

        # Using a constant color, put in desired RGB values here.
        colourRGB = np.array((255.0 / 255.0, 54.0 / 255.0, 57 / 255.0, 1.0))

        # The correct shading for shadows are now applied. Use the face normals and light orientation to generate a shading value and apply to the RGB colors for each face.
        rgbNew = np.array([colourRGB * (newMin + newdiff * ((shade - min) / diff)) for shade in ls.shade_normals(normalsarray, fraction=1.0)])



        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlim(np.min(verts[:, 0]), np.max(verts[:, 0]))
        ax.set_ylim(np.min(verts[:, 1]), np.max(verts[:, 1]))
        ax.set_zlim(np.min(verts[:, 2]), np.max(verts[:, 2]))

        mesh = Poly3DCollection(verts[faces], alpha=1)

        # Apply color to face
        mesh.set_facecolor(rgbNew)

        ax.add_collection3d(mesh)
        plt.tight_layout()
        plt.show()
        exit(1)

    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax=ax, slices=contoured)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()

    pred_f = np.squeeze(predicted, axis=-1)
    t = 0.01
    for i in range(pred_f.shape[0]):
        pred_f[i, ...][pred_f[i, ...] > t] = 255
        pred_f[i, ...][pred_f[i, ...] <= t] = 0


if __name__ == '__main__':

    main()
