import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def dice_loss(y_true, y_pred):
    '''

    Also know as dice distance, measures dissimilarity between sample sets, it is complementary to the dice coefficient and it is obtained by subtracting the dice coefficient from 1.

    Parameters
    ----------
    y_true : Tensor
        input tensor with shape [batch_size, d0, .. dN].
    y_pred : Tensor
        predicted tensor with same shape of y_true.

    Returns
    -------
    Tensor
       Loss float tensor with shape [batch_size, d0, .. dN -1].

    References
    -----------
    - Wiki https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient

    '''

    smooth = 1  # to avoid 0 division
    intersection = y_true * y_pred
    total = y_true + y_pred
    dice = tf.reduce_mean((2. * intersection + smooth) /
                          (total + smooth), axis=-1)
    dice_loss = 1. - dice

    return dice_loss


def DiceBCEloss(y_true, y_pred):
    '''

    This loss combines Dice loss with the standard binary cross-entropy (BCE) loss that is generally the default for segmentation models. Combining the two methods allows for some diversity in the loss, while benefitting from the stability of BCE.

    Parameters
    ----------
    y_true : Tensor
        input tensor with shape [batch_size, d0, .. dN].
    y_pred : Tensor
        predicted tensor with same shape of y_true.

    Returns
    -------
    Tensor
       Loss float tensor with shape [batch_size, d0, .. dN -1].

    References
    -----------
    - Kaggle https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

    '''

    smooth = 1  # to avoid 0 division

    bce = binary_crossentropy(y_true, y_pred)

    dice_l = dice_loss(y_true, y_pred)
    dice_bce = bce + dice_l

    return dice_bce


def iou_loss(y_true, y_pred):
    '''

    Also know as Jaccard distance, measures dissimilarity between sample sets, it is complementary to the Jaccard coefficient and is obtained by subtracting the Jaccard coefficient from 1, or, equivalently, by dividing the difference of the sizes of the union and the intersection of two sets by the size of the union.

    Parameters
    ----------
    y_true : Tensor
        input tensor with shape [batch_size, d0, .. dN].
    y_pred : Tensor
        predicted tensor with same shape of y_true.

    Returns
    -------
    Tensor
       Loss float tensor with shape [batch_size, d0, .. dN -1].

    References
    -----------
    - Wiki https://en.wikipedia.org/wiki/Jaccard_index


    '''
    smooth = 1

    intersection = y_true * y_pred
    total = y_true + y_pred
    union = total - intersection
    iou = tf.reduce_mean((intersection + smooth) /
                         (union + smooth), axis=-1)
    iou_loss = 1. - iou

    return iou_loss


def tversky_loss(y_true, y_pred):
    '''

     Measure of dissimilarity between sample sets, is complementary to the Tversky index and is obtained by subtracting the Tversky index from 1.

    Parameters
    ----------
    y_true : Tensor
        input tensor with shape [batch_size, d0, .. dN].
    y_pred : Tensor
        predicted tensor with same shape of y_true.

    Returns
    -------
    Tensor
       Loss float tensor with shape [batch_size, d0, .. dN -1].

    References
    -----------
    - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8759329

    '''
    smooth = 1
    beta = 1 - alpha  # since alpha + beta = 1 cases are of more interest

    true_pos = y_true * y_pred
    false_neg = y_true * (1. - y_pred)
    false_pos = (1. - y_true) * y_pred
    tversky = tf.reduce_mean((true_pos + smooth)/(true_pos + alpha *
                                                  false_neg + beta * false_pos + smooth), axis=-1)
    tversky_loss = 1. - tversky

    return tversky_loss


def focal_tversky_loss(y_true, y_pred):
    '''

   Generalized focal loss function based on the Tversky index to address the issue of data imbalance in medical image segmentation.

    Parameters
    ----------
    y_true : Tensor
        input tensor with shape [batch_size, d0, .. dN].
    y_pred : Tensor
        predicted tensor with same shape of y_true.

    Returns
    -------
    Tensor
       Loss float tensor with shape [batch_size, d0, .. dN -1].

    References
    -----------
    - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8759329

    '''

    smooth = 1
    gamma = 1.33  # by default 1.33. Proposed in range [1,3]

    T = tversky_loss(y_true, y_pred)

    return tf.pow(T, 1/gamma)
