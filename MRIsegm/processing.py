import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2

from MRIsegm.utils import make_label

__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def explore_histogram(slice, layer):

    fig, ax = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)

    ax[0].imshow(slice[layer, :, :], cmap="gray")
    ax[0].set_title("Original image")
    ax[0].axis("off")

    ax[1].hist(slice[layer, :, :].ravel(), 256, [0, 256], color="black")
    ax[1].set_title("Histogram")

    fig.suptitle(f"Exploring layer: {layer}",  fontsize=20)
    plt.show()


def manual_tresh(slice, layer, threshold):

    image = slice[layer, :, :].copy()

    t_min = threshold[0]
    t_max = threshold[1]

    mask = cv2.threshold(image, t_min, t_max, cv2.THRESH_BINARY_INV)[1]

    fig, ax = plt.subplots(1, 3, figsize=(12, 8), constrained_layout=True)

    ax[0].imshow(slice[layer, :, :], cmap="gray")
    ax[0].set_title("Original image")
    ax[0].axis("off")

    ax[1].hist(slice[layer, :, :].ravel(), 256, [0, 256], color="black")
    ax[1].set_title("Histogram")
    ax[1].vlines(t_min, 0, 10000, color="red", linestyle="dashed")
    ax[1].vlines(t_max, 0, 10000, color="red", linestyle="dashed")
    ax[1].text(t_min, 10000, "$T_{min}$")
    ax[1].text(t_max, 10000, "$T_{max}$")
    ax[1].set_yticks([])

    ax[2].imshow(mask, cmap="gray")
    ax[2].set_title("Thresholded Image")
    ax[2].axis("off")

    fig.suptitle(f"Exploring layer: {layer}",  fontsize=20)

    plt.show()


def adaptive_threshold(slice, layer):

    img = image = slice[layer, :, :].copy()
    blur = cv2.medianBlur(img, ksize=5)

    th = cv2.adaptiveThreshold(
        blur, 150, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    th2 = cv2.adaptiveThreshold(
        blur, 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    fig, ax = plt.subplots(1, 3, figsize=(12, 8), constrained_layout=True)

    ax[0].imshow(img, "gray")
    ax[0].axis("off")
    ax[0].set_title("Original image")

    ax[1].imshow(th, "gray")
    ax[1].axis("off")
    ax[1].set_title("Adaptive Threshold Mean")

    ax[2].imshow(th2, "gray")
    ax[2].axis("off")
    ax[2].set_title("Adaptive Threshold Gaussian")

    fig.suptitle(f'Exploring layer: {layer}', fontsize=20)
    plt.show()


def otsu_threshold(slice, layer):

    img = slice[layer, :, :].copy()
    gauss = cv2.GaussianBlur(img, (5, 5), 0)

    t_min = 0
    t_max = 255

    thresh_otsu = cv2.threshold(
        gauss, t_min, t_max, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)

    ax[0].imshow(slice[layer, :, :], cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Original")

    ax[1].imshow(thresh_otsu, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title(f"Otsu")

    fig.suptitle(f'Exploring layer: {layer}', fontsize=20)
    plt.show()


def box_fov(slice, layer, threshold):

    img = slice[layer, :, :].copy()
    box = np.zeros(img.shape[:2], np.uint8)
    box[150:350, 150:350] = 255

    boxed_original = cv2.bitwise_and(img, img, mask=box)

    t_min = threshold[0]
    t_max = threshold[1]

    th = cv2.threshold(img, t_min, t_max, cv2.THRESH_BINARY_INV)[1]
    masked_th = cv2.bitwise_and(th, th, mask=box)

    gauss = cv2.GaussianBlur(img, (5, 5), 0)
    thresh_otsu = cv2.threshold(
        gauss, 0, 256, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    masked_otsu = cv2.bitwise_and(thresh_otsu, thresh_otsu, mask=box)

    fig, ax = plt.subplots(1, 3, figsize=(12, 8), constrained_layout=True)

    ax[0].imshow(boxed_original, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Original")

    ax[1].imshow(masked_th, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title(f"Global Thresh {t_min, t_max}")

    ax[2].imshow(masked_otsu, cmap="gray")
    ax[2].axis("off")
    ax[2].set_title(f"Otsu")

    fig.suptitle(f'Exploring layer: {layer}', fontsize=20)
    plt.show()


def gabor_filter(ksize, theta, sigma, lamb, gamma, psi):

    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lamb, gamma, psi, ktype=cv2.CV_32F)

    return kernel


def show_gabor_filter(slice, layer, ksize, sigma, theta, lamb, gamma, psi):

    kernel = gabor_filter(ksize=ksize, theta=theta,
                          sigma=sigma, lamb=lamb, gamma=gamma, psi=psi)

    img = slice[layer, :, :].copy()

    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)

    ax[0].imshow(kernel)
    ax[0].set_title("Gabor Kernel")

    ax[1].imshow(filtered_img, cmap="gray")
    ax[1].set_title("Filtered image")

    fig.suptitle(f'Exploring layer: {layer}', fontsize=20)
    plt.show()


def denoise_nlm(slice, layer, alpha, show=False):

    img = slice[layer, :, :].copy()

    sigma = np.mean(estimate_sigma(img, multichannel=False))

    denoise_img = denoise_nl_means(
        img, h=alpha*sigma, patch_size=5, patch_distance=3, multichannel=False, preserve_range=True)

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)

        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("original")

        ax[1].imshow(denoise_img, cmap="gray")
        ax[1].set_title("after NLM denoise")

        fig.suptitle(f"Exploring layer {layer}", fontsize=20)

    else:
        return denoise_img.astype('uint8')


def compare_denoised_histo(slice, layer, alpha):

    img = slice[layer, :, :].copy()

    denoised_img = denoise_nlm(slice, layer, alpha=alpha, show=False)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax[0][0].imshow(denoised_img, cmap="gray")
    ax[0][0].set_title("denoised")
    ax[0][0].axis("off")

    ax[0][1].hist(denoised_img.ravel(), 256, [0, 256], color="black")
    ax[0][1].set_title("Histogram")

    ax[1][0].imshow(img, cmap="gray")
    ax[1][0].set_title("original")
    ax[1][0].axis("off")

    ax[1][1].hist(img.ravel(), 256, [0, 256], color="black")
    ax[1][1].set_title("Histogram")

    fig.suptitle(f"Exploring layer: {layer}",  fontsize=20)
    plt.show()


def denoise_slice(slice, alpha=1.15, patch_size=5, patch_distance=3):

    denoised_slice = np.zeros_like(slice)

    for layer in range(slice.shape[0]):

        img = slice[layer, :, :].copy()

        sigma = np.mean(estimate_sigma(img, multichannel=False))

        denoised_img = denoise_nl_means(img, h=alpha*sigma, patch_size=patch_size,
                                        patch_distance=patch_distance, multichannel=False, preserve_range=True)

        denoised_slice[layer, :, :] = denoised_img

    return denoised_slice.astype('uint8')


def zoom_image(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def compare_zoomed(slice, layer, zoom_factor, show_mask, **kwargs):

    img = slice[layer, :, :].copy()
    zoomed_img = zoom_image(img=img, zoom_factor=zoom_factor)

    if show_mask:

        positions = kwargs.get('positions', None)
        xs = kwargs.get('xs', None)
        ys = kwargs.get('ys', None)

        mask_label = make_label(slice=slice, layer=layer,
                                positions=positions, xs=xs, ys=ys)
        zoomed_mask = zoom_image(img=mask_label, zoom_factor=zoom_factor)

        fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        ax[0][0].imshow(img, cmap="gray")
        ax[0][0].set_title("original")

        ax[0][1].imshow(zoomed_img, cmap="gray")
        ax[0][1].set_title(f"zoom x {zoom_factor}")

        ax[1][0].imshow(mask_label, cmap="gray")
        ax[1][0].set_title("original mask")

        ax[1][1].imshow(zoomed_mask, cmap="gray")
        ax[1][1].set_title(f"zoom x {zoom_factor}")

        fig.suptitle(f"Exploring layer: {layer}", fontsize=20)
    else:

        fig, ax = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("original")

        ax[1].imshow(zoomed_img, cmap="gray")
        ax[1].set_title(f"zoom x {zoom_factor}")
        fig.suptitle(f"Exploring layer: {layer}", fontsize=20)


def zoom_slice(slice, zoom_factor):

    zoomed_slice = np.zeros_like(slice)

    for layer in range(slice.shape[0]):

        img = slice[layer, :, :].copy()
        zoomed_img = zoom_image(img, zoom_factor)

        zoomed_slice[layer, :, :] = zoomed_img

    return zoomed_slice


def box_slice_manual(slice, h=150, w=350):

    boxed_slice = np.zeros_like(slice)

    for layer in range(slice.shape[0]):

        img = slice[layer, :, :].copy()
        box = np.zeros(slice.shape[1:3], np.uint8)
        box[h:w, h:w] = 255

        boxed_img = cv2.bitwise_and(img, img, mask=box)

        boxed_slice[layer, :, :] = boxed_img

    return boxed_slice_manual


def boxer(slice, layer, positions, xs, ys):

    mask = make_label(slice=slice, layer=layer,
                      positions=positions, xs=xs, ys=ys)

    x, y, w, h = cv2.boundingRect(mask)

    img = slice[layer, :, :].copy()
    box = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)

    boxed = cv2.bitwise_and(img, img, mask=box)

    return boxed


def box_slice(slice, positions, xs, ys):

    boxed_slice = np.zeros_like(slice)

    for layer in range(slice.shape[0]):

        if not layer in positions:
            boxed_img = 0
        else:
            boxed_img = boxer(slice=slice, layer=layer,
                              positions=positions, xs=xs, ys=ys)

        boxed_slice[layer, :, :] = boxed_img

    return boxed_slice


def opening(image, ksize):

    kernel = np.ones((ksize, ksize), np.uint8)
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return result
