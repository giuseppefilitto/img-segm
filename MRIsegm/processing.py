import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def otsu_thresholding(img, **kwargs):
    '''
    Perform Otsu thresholding on input image

    Parameters
    ----------
    img : image, array_like
        input image to be thresholded

    Returns
    -------
    th  : thresholded image
        thresholded image using cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU 
    '''
    if kwargs.get("gaussian"):
        ksize = kwargs.get("ksize")
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    return th


def add_images(img1, img2):
    '''
    Add inputs images one over the other using cv2.add() function

    Parameters
    ----------
    img1 : image, array_like
        first input image 
    img2 : image, array_like
        second input image 

    Returns
    -------
    combo: image, array_like
        return img1 + img2 
    '''
    combo = cv2.add(img1, img2)

    return combo


def show_image_histogram(img, show_original=True):
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


def show_slice_histogram(slice, layer, show_original=True):

    args = dict(img=slice[layer, :, :], show_original=show_original)
    show_image_histogram(**args)
    plt.suptitle(f'Layer : {layer}')


def denoise_nlm(img, alpha, show=False, **kwargs):

    sigma = np.mean(estimate_sigma(img, multichannel=False))

    denoised_img = denoise_nl_means(
        img, h=alpha*sigma, multichannel=False, preserve_range=True)

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


def denoise_slice(slice, alpha=1.15):

    denoised_slice = np.zeros_like(slice)

    for layer in range(slice.shape[0]):

        img = slice[layer, :, :].copy()

        denoised_img = denoise_nlm(img, alpha)

        denoised_slice[layer, :, :] = denoised_img

    return denoised_slice


def compare_denoised_histo(img, alpha, figsize=(15, 15)):
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
