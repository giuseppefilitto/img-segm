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


def show_image_histogram(img, show_original=True):  # pragma: no cover
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


def show_slice_histogram(slices, layer, show_original=True):  # pragma: no cover
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


def denoise_nlm(img, alpha, show=False, **kwargs):  # pragma: no cover
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
        fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        ax[0].axis('off')
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("original")
        ax[1].axis('off')
        ax[1].imshow(denoised_img, cmap="gray")
        ax[1].set_title("filtered")

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


def compare_denoised_histo(img, alpha, figsize=(15, 15)):  # pragma: no cover
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
    plt.title('original', fontsize=15)
    plt.subplot(2, 2, 2)
    plt.hist(img.ravel(), 256, [0, 256], color="black")
    plt.title('Histogram', fontsize=15)

    plt.subplot(2, 2, 3)
    denoised_img = denoise_nlm(img, alpha)
    plt.imshow(denoised_img, cmap='gray')
    plt.title('filtered', fontsize=15)
    plt.subplot(2, 2, 4)
    plt.hist(denoised_img.ravel(), 256, [0, 256], color="black")
    plt.title('Histogram', fontsize=15)



def resize_slices(slices, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=np.float32):
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
    dtype : numpy dtype
        dtype of the resized slices, by default np.float32.
    Returns
    -------
    resized_slice: numpy array
        resized array of shape (layers, height, width, 1) where 1 denotes the number of channels.
    '''

    resized_slices = np.zeros(shape=(slices.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=dtype)

    for layer in range(slices.shape[0]):
        IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
        norm = slices[layer, :, :] * 1. / 255
        resized = cv2.resize(norm.copy(), IMG_SIZE)
        resized = resized[np.newaxis, :, :, np.newaxis]
        resized_slices[layer, :, :, :] = resized

    return resized_slices


def predict_slices(slices, model, IMAGE_HEIGHT, IMAGE_WIDTH):
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

    Returns
    -------
    numpy array
        array of shape = (layers, height, width, 1).
    '''

    predicted_slices = np.zeros(
        shape=(slices.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    for layer in range(slices.shape[0]):

        IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
        norm = slices[layer, :, :] * 1. / 255
        resized = cv2.resize(norm, IMG_SIZE)
        resized = resized[np.newaxis, :, :, np.newaxis]
        predicted_slices[layer, :, :, :] = model.predict(resized)[...]

    return predicted_slices


def write_contour(slices, predicted_slices, layer):  # pragma: no cover
    '''
    Draw the contours of the predicted mask over the original image.

    Parameters
    ----------
    slices : numpy array
        Numpy array of shape (layers, height, width, 1).
    predicted_slices : numpy array
        array of shape = (layers, height, width, 1).
    layer : int
        value between (0, slices.shape[0]).

    Returns
    -------
    array
        return a RGB image array.
    '''

    pred_mask = predicted_slices[layer, ...] * 255
    pred_mask = np.squeeze(pred_mask, axis=-1)
    pred_mask = pred_mask.astype('uint8')

    contours = cv2.findContours(
        pred_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    original = slices[layer, ...] * 255
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
        return an array of shape = (*slices.shape, 3).
    '''

    contoured_slices = np.zeros(
        shape=(*predicted_slices.shape[0:3], 3), dtype=np.uint8)

    for layer in range(predicted_slices.shape[0]):
        cont = write_contour(slices, predicted_slices, layer)
        contoured_slices[layer, ...] = cont

    return contoured_slices


def crop_image(img):
    '''
    Crop full image to center from 512x512 to 256 x 256

    Parameters
    ----------
    img : array like
        image to be cropped

    Returns
    -------
    array like image
        cropped image
    '''

    height, width = img.shape[0], img.shape[1]

    if height != 512:
        img = cv2.resize(img, (512, 512))

    assert img.shape[0] == 512

    y, x = 256, 256
    dy, dx = y // 2, x // 2

    return (img[(y - dy):(y + dy), (x - dx):(x + dx)])


def rescale(img):
    '''
    Normalize and rescale image to binary floating 32-bit

    Parameters
    ----------
    img : image, array like
        image to be normalized and rescaled

    Returns
    -------
    image, array like
        [normalaized and rescaled image
    '''
    rescaled = cv2.normalize(img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return rescaled


def denoise(img, alpha=10):
    '''
    Denoise the image using non-local means algorithm

    Parameters
    ----------
    img : image, array like
        image to be denoised
    alpha : float, optional
        smoothing parameter, by default 10

    Returns
    -------
    image, array like
        smoothed denoised image
    '''

    patch_kw = dict(patch_size=5, patch_distance=6,)
    sigma_est = np.mean(estimate_sigma(img))
    denoised = denoise_nl_means(img, h=alpha * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
    return denoised


def gamma_correction(img, gamma=1.0):
    '''
    Perform gamma correction. The true value of gamma used in the formula is 1/gamma.

    Parameters
    ----------
    img : image, array like
        image to be filtered
    gamma : float, optional
        gamma value, by default 1.0

    Returns
    -------
    image, array like
        gamma corrected image
    '''
    igamma = 1.0 / gamma
    imin, imax = img.min(), img.max()

    img_c = img.copy()
    img_c = ((img_c - imin) / (imax - imin)) ** igamma
    img_c = img_c * (imax - imin) + imin
    return img_c



def pre_processing_data(slices, alpha=10):
    '''
    Single-shot preprocessind data function. Performing rescaling, denoising, gamma correction.

    Parameters
    ----------
    slices : array of shape: (depth, hight, width)
        stack of images to be preprocessed
    alpha : float, optional
        smoothing parameter, by default 10

    Returns
    -------
    array of shape: (depth, hight, width)
        pre-processed slices
    '''

    imgs = []
    for layer in range(slices.shape[0]):
        img = slices[layer, :, :]
        if slices.shape[1:3] != 512:
            resized = cv2.resize(img, (512, 512))
        else:
            resized = img
        rescaled = rescale(resized)
        denoised = denoise(rescaled, alpha)
        gamma = gamma_correction(denoised)
        imgs.append(gamma)

    images = [np.expand_dims(im, axis=-1) for im in imgs]
    images = np.array(images)

    return images


def predict_images(slices, model, pre_processing=False, t=0.1):
    '''
    Predict full stack of slices, slice by slice

    Parameters
    ----------
    slices : array of shape: (depth, hight, width)
        stack of images
    model : Tensorflow-Keras model instance
        loaded .h5 keras model by tf.keras.model.load_model()
    pre_processing : bool, optional
        if true pre-process the images, by default False
    t : float, optional
        threshold parameter. Values below this threshold value are set to 0, by default 0.1

    Returns
    -------
    array of shape: (depth, hight, width, 1)
        stack of predicted images, corresponding to the prediction for the original slices.
    '''

    if pre_processing:
        prep_slices = pre_processing_data(slices)
    else:
        prep_slices = slices

    predicted_slices = np.zeros_like(prep_slices)

    for layer in range(slices.shape[0]):
        img = prep_slices[layer, ...]
        predicted_slices[layer, ...] = model.predict(img[np.newaxis, ...])[...]
        predicted_slices[layer, ...] = np.where(predicted_slices[layer, ...] <= 0.1, 0, predicted_slices[layer, ...])


    return predicted_slices


def crop_masks(slices):
    '''
    Crop slices of masks

    Parameters
    ----------
    slices : array of shape: (depth, hight, width)
        stack of masks to be cropped

    Returns
    -------
    array of shape: (depth, hight, width)
        stack of cropped masks
    '''
    imgs = []
    for layer in range(slices.shape[0]):
        img = slices[layer, :, :]
        cropped = crop_image(img)

        imgs.append(cropped)

    images = np.array(imgs)
    return images
