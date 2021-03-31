import numpy as np
import matplotlib.pyplot as plt
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
