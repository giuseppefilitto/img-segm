import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def create_segmentation_generator_train(img_path, mask_path, BATCH_SIZE, IMG_SIZE, SEED):

    data_gen_args_img = dict(rescale=1./255,
                             #                      featurewise_center=True,
                             #                      featurewise_std_normalization=True,
                             rotation_range=5,
                             #                      width_shift_range=0.2,
                             #                      height_shift_range=0.2,
                             #                    zoom_range=0.5,
                             horizontal_flip=True

                             )

    data_gen_args_mask = dict(rescale=1./255,
                              #                      featurewise_center=True,
                              #                      featurewise_std_normalization=True,
                              rotation_range=5,
                              #                      width_shift_range=0.2,
                              #                      height_shift_range=0.2,
                              #                    zoom_range=0.5,
                              horizontal_flip=True
                              )

    img_data_gen = ImageDataGenerator(**data_gen_args_img)
    mask_data_gen = ImageDataGenerator(**data_gen_args_mask)

    img_generator = img_data_gen.flow_from_directory(
        img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    mask_generator = mask_data_gen.flow_from_directory(
        mask_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)

    return zip(img_generator, mask_generator)


def create_segmentation_generator_validation(img_path, mask_path, BATCH_SIZE, IMG_SIZE, SEED):

    data_gen_args = dict(rescale=1./255)

    img_data_gen = ImageDataGenerator(**data_gen_args)
    mask_data_gen = ImageDataGenerator(**data_gen_args)

    img_generator = img_data_gen.flow_from_directory(
        img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    mask_generator = mask_data_gen.flow_from_directory(
        mask_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)

    return zip(img_generator, mask_generator)


def create_segmentation_generator_test(img_path, mask_path, BATCH_SIZE, IMG_SIZE, SEED):

    data_gen_args = dict(rescale=1./255)

    img_data_gen = ImageDataGenerator(**data_gen_args)
    mask_data_gen = ImageDataGenerator(**data_gen_args)

    img_generator = img_data_gen.flow_from_directory(
        img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    mask_generator = mask_data_gen.flow_from_directory(
        mask_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)

    return zip(img_generator, mask_generator)


def display(display_list, colormap=False):
    plt.figure(figsize=(12, 8))
    title = ['Input image', 'True Mask', 'Predicted mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if i == 2 and colormap == True:
            ax = plt.gca()

            im = ax.imshow((display_list[i]),
                           cmap='gist_heat', vmin=0.0, vmax=1.0)

            axins = inset_axes(ax, width="5%", height="100%", loc='lower left',
                               bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax.transAxes,
                               borderpad=0)

            plt.colorbar(im, cax=axins)

        else:
            # optional: tf.keras.preprocessing.image.array_to_img, just to rescale from 0.-1. to 0-255
            plt.imshow(tf.keras.preprocessing.image.array_to_img(
                display_list[i]), cmap='gray')

    plt.show()


def show_dataset(datagen, num=1):
    for i in range(0, num):
        image, mask = next(datagen)
        display([image[0], mask[0]])


def plot_history(model_name, metrics, loss, save=True, custom_metrics=True, custom_loss=False, **kwargs):

    plt.style.use('seaborn')
    plt.figure(figsize=(18, 8))
    plt.suptitle('Model: ' + model_name, fontsize=17)
    if kwargs.get('labelsize'):
        labelsize = kwargs.get('labelsize')
        plt.rc('xtick', labelsize=labelsize)
        plt.rc('ytick', labelsize=labelsize)

    for i in range(len(metrics)):
        plt.subplot(1, len(metrics) + 1, i+1)
        if custom_metrics:
            plt.plot(history.history['{}'.format(metrics[i].__name__)])
            plt.plot(history.history['val_' +
                                     '{}'.format(metrics[i].__name__)])
            plt.title('model  {}'.format(metrics[i].__name__), fontsize=15)
            plt.xlabel('epoch', fontsize=15)
            plt.ylabel('{}'.format(metrics[i].__name__), fontsize=15)

        else:
            plt.plot(history.history[metrics[i]])
            plt.plot(history.history['val_' + metrics[i]])
            plt.title('model ' + metrics[i], fontsize=15)
            plt.xlabel('epoch', fontsize=15)
            plt.ylabel(metrics[i], fontsize=15)

    if custom_loss:
        plt.subplot(1, len(metrics) + 1, len(metrics) + 1)
        plt.plot(history.history['{}'.format(loss.__name__)])
        plt.plot(history.history['val_' + '{}'.format(loss.__name__)])
        plt.title('model {}'.format(loss.__name__), fontsize=15)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('{}'.format(loss.__name__), fontsize=15)

    else:
        plt.subplot(1, len(metrics) + 1, len(metrics) + 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss', fontsize=15)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel(loss, fontsize=15)

    if save:
        path = kwargs.get('path')
        plt.savefig(path)
    plt.show()


def show_prediction(datagen, num=1, colorbar=True):
    for i in range(0, num):
        image, mask = next(datagen)
        pred_mask = model.predict(image)
        display([image[0], mask[0], pred_mask[0]], colorbar)
