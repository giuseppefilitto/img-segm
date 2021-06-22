import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def otsu_thresholding(img, **kwargs):
    '''
    Perform Otsu thresholding on input image.

    Parameters
    ----------
    img : image, array_like
        input image to be thresholded.

    Returns
    -------
    th  : thresholded image
        thresholded image using cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU.
    '''
    if kwargs.get("gaussian"):
        ksize = kwargs.get("ksize")
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return th


def add_images(img1, img2):
    '''
    Add inputs images one over the other using cv2.add() function.

    Parameters
    ----------
    img1 : image, array_like
        first input image .
    img2 : image, array_like
        second input image.

    Returns
    -------
    combo: image, array_like
        return img1 + img2.
    '''
    combo = cv2.add(img1, img2)

    return combo


def show_image_histogram(img, show_original=True):
    '''

    Plot the histogram of the input image.

    Parameters
    ----------
    img : image, array_like
        image to calculate the histogram from.
    show_original : bool, optional
        if True show both the image and the histogram otherwise only the histogram, by default True.
    '''
    plt.figure(figsize=(12, 5), constrained_layout=True)
    if show_original:
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Image', fontsize=15)
        plt.subplot(1, 2, 2)
        plt.hist(img.ravel(), 256, [0, 256], color="black")
        plt.title('Histogram', fontsize=15)
    else:
        plt.hist(img.ravel(), 256, [0, 256], color="black")
        plt.title('Histogram', fontsize=15)
    plt.show()


def show_slice_histogram(slices, layer, show_original=True):
    '''

    Show the histogram of the given slices of the slices stack.

    Parameters
    ----------
    slices : array
        array of shape depth, height, width.
    layer : int
        value between (0, slices.shape[0]).
    show_original : bool, optional
        if True show both the image and the histogram otherwise only the histogram, by default True.
    '''

    args = dict(img=slices[layer, :, :], show_original=show_original)
    show_image_histogram(**args)
    plt.suptitle(f'Layer : {layer}')


def denoise_nlm(img, alpha, show=False, **kwargs):
    '''

    Denoise image using non-local mean filter.

    Parameters
    ----------
    img : image, array_like
        image to be denoised.
    alpha : int, float
        smoothing parameters. A higher value of alpha results in a smoother image, at the expense of blurring features.
    show : bool, optional
        if True show both the original image and the denoised one otherwise returns 'denoised_img', by default False.

    Returns
    -------
    denoised_img: image, array_like
        return denoised input image.
    '''

    sigma = np.mean(estimate_sigma(img, multichannel=False))

    denoised_img = denoise_nl_means(
        img, h=alpha * sigma, multichannel=False, preserve_range=True)

    if show:
        figsize = kwargs.get('figsize')
        fig, ax = plt.subplots(1, 2, figsize=(
            figsize), constrained_layout=True)

        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("original")

        ax[1].imshow(denoised_img, cmap="gray")
        ax[1].set_title("after NLM denoise")

    else:
        return denoised_img.astype('uint8')


def denoise_slices(slices, alpha=1.15):
    '''
    Denoise each layer of a the entire slices stack.

    Parameters
    ----------
    slices : array
        array of shape depth, height, width
    alpha : int, float
        smoothing parameters. A higher value of alpha results in a smoother image, at the expense of blurring features, by default 1.15 .

    Returns
    -------
    denoised_slices: array
        array of shape depth, height, width.
    '''

    denoised_slices = np.zeros_like(slices)

    for layer in range(slices.shape[0]):

        img = slices[layer, :, :].copy()

        denoised_img = denoise_nlm(img, alpha)

        denoised_slices[layer, :, :] = denoised_img

    return denoised_slices


def compare_denoised_histo(img, alpha, figsize=(15, 15)):
    '''
    Compare the histogram of the original and denoised image showing both the image and the relative histogram.

    Parameters
    ----------
    img : image, array_like
        image to be denoised.
    alpha : int, float
        smoothing parameters. A higher value of alpha results in a smoother image, at the expense of blurring features.
    figsize : tuple, optional
        figsize arg of matplotlib module, by default (15, 15).
    '''

    plt.figure(figsize=figsize)

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image', fontsize=15)
    plt.subplot(2, 2, 2)
    plt.hist(img.ravel(), 256, [0, 256], color="black")
    plt.title('Histogram', fontsize=15)

    plt.subplot(2, 2, 3)
    denoised_img = denoise_nlm(img, alpha)
    plt.imshow(denoised_img, cmap='gray')
    plt.title('Image denoised', fontsize=15)
    plt.subplot(2, 2, 4)
    plt.hist(denoised_img.ravel(), 256, [0, 256], color="black")
    plt.title('Histogram', fontsize=15)


def apply_mask(img, mask):
    '''
    Apply a mask to the image using bitwise_and from cv2.

    Parameters
    ----------
    img : image, array_like
        image to be masked.
    mask : image, array_like
        black and white image.

    Returns
    -------
    applied_mask: image, array_like
        image where each pixel is the same of the 'img' only if the corrisponding pixel of the mask is white.
    '''

    img = img.copy()
    applied_mask = cv2.bitwise_and(img, mask, mask=mask)

    return applied_mask


def resize_slices(slices, IMAGE_HEIGHT, IMAGE_WIDTH):
    '''

    Resize slices to shape = (layers, height, width) to shape = (layers, height, width, 1).

    Parameters
    ----------
    slices : numpy array
        Numpy array of shape = (layers, height, width).
    IMAGE_HEIGHT : int
        image height.
    IMAGE_WIDTH : int
        image width.

    Returns
    -------
    resized_slice: numpy array
        resized array of shape (layers, height, width, 1) where 1 denotes the number of channels.
    '''

    resized_slices = np.zeros(shape=(slices.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)

    for layer in range(slices.shape[0]):
        IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
        norm = slices[layer, :, :] * 1. / 255
        resized = cv2.resize(norm.copy(), IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        resized = resized[np.newaxis, :, :, np.newaxis]
        resized_slices[layer, :, :, :] = resized

    return resized_slices


def predict_slices(slices, model, IMAGE_HEIGHT, IMAGE_WIDTH, threshold):
    '''
    Create stack of predicted slice mask using the given model.

    Parameters
    ----------
    slices : numpy array
         Numpy array of shape = (layers, height, width, 1).
    model :  Keras model instance
        loaded .h5 keras model by tf.keras.model.load_model() .
    IMAGE_HEIGHT : int
        image height.
    IMAGE_WIDTH : int
        image width.
    threshold : float
        min threshold value.

    Returns
    -------
    numpy array
        array of shape = (layers, height, width, 1).
    '''

    predicted_slices = np.zeros(
        shape=(slices.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)

    for layer in range(slices.shape[0]):

        IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
        norm = slices[layer, :, :] * 1. / 255
        resized = cv2.resize(norm, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        resized = resized[np.newaxis, :, :, np.newaxis]
        predicted_slices[layer, :, :, :] = model.predict(resized) > threshold

    return predicted_slices


def write_contour(slices, predicted_slices, layer):
    '''
    Draw the contours of the predicted mask over the original image.

    Parameters
    ----------
    slices : numpy array
        Numpy array of shape = (layers, height, width, 1).
    predicted_slices : numpy array
        array of shape = (layers, height, width, 1).
    layer : int
        value between (0, slices.shape[0]).

    Returns
    -------
    array
        return an image with shape=(*slices.shape[1:3], 3).
    '''

    pred_mask = predicted_slices[layer, :, :, :] * 255
    pred_mask = np.squeeze(pred_mask, axis=-1)
    pred_mask = pred_mask.astype('uint8')

    contours = cv2.findContours(
        pred_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    original = slices[layer, :, :, :] * 255
    original = np.squeeze(original, axis=-1)
    original = original.astype('uint8')

    # ! RGB image so 3 channels

    ori = np.expand_dims(original, axis=2).repeat(3, axis=2)

    for k, _ in enumerate(contours):
        cv2.drawContours(ori, contours, k, (255, 0, 0), 1)

    return ori


def contour_slices(slices, predicted_slices):
    '''
    Draw the contours of the predicted mask over the original image for each slice of the stack.

    Parameters
    ----------
    slices : numpy array
        Numpy array of shape = (layers, height, width, 1).
    predicted_slices : numpy array
        array of shape = (layers, height, width, 1).

    Returns
    -------
    array
        return an array of shape = (*slices.shape, 3)
    '''

    contoured_slices = np.zeros(
        shape=(*predicted_slices.shape[0:3], 3), dtype=np.uint8)

    for layer in range(predicted_slices.shape[0]):
        cont = write_contour(slices, predicted_slices, layer)
        contoured_slices[layer, ...] = cont

    return contoured_slices
