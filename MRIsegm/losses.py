import tensorflow as tf
import tensorflow.keras.backend as K
from MRIsegm.metrics import dice_coef, iou, tversky_index


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def iou_loss(y_true, y_pred):
    '''

    Also know as Jaccard distance, measures dissimilarity between sample sets, is complementary to the Jaccard coefficient and is obtained by subtracting the Jaccard coefficient from 1, or, equivalently, by dividing the difference of the sizes of the union and the intersection of two sets by the size of the union.

    Parameters
    ----------
    y_true : Tensor
        input tensor with shape: [batch_size, height, width, channels].
    y_pred : Tensor
        predicted tensor with shape: [batch_size, height, width, channels].

    References
    -----------
    - Wiki https://en.wikipedia.org/wiki/Jaccard_index


    '''

    iou = iou(y_true, y_pred)
    return 1 - iou


def tversky_loss(y_true, y_pred):
    '''

     Measure of dissimilarity between sample sets, is complementary to the Tversky index and is obtained by subtracting the Tversky index from 1.

    Parameters
    ----------
    y_true : Tensor
        input tensor with shape: [batch_size, height, width, channels].
    y_pred : Tensor
        predicted tensor with shape: [batch_size, height, width, channels].

    References
    -----------
    - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8759329

    '''

    T = tversky_index(y_true, y_pred)

    return 1 - T


def focal_tversky_loss(y_true, y_pred, gamma=1.33):
    '''

   Generalized focal loss function based on the Tversky index to address the issue of data imbalance in medical image segmentation.

    Parameters
    ----------
    y_true : Tensor
        input tensor with shape: [batch_size, height, width, channels].
    y_pred : Tensor
        predicted tensor with shape: [batch_size, height, width, channels].
    gamma: float
        control parameter, by default 1.33. Proposed in range [1,3].

    References
    -----------
    - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8759329

    '''

    T = tversky(y_true, y_pred)

    return K.pow((1-T), 1/gamma)
