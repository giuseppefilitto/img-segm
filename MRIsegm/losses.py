import tensorflow as tf
from MRIsegm.metrics import dice_coef


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
    Weighted loss float Tensor
       scalar dice loss tensor with shape=().

    References
    -----------
    - Wiki https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient

    '''

    dice = dice_coef(y_true, y_pred)
    dice_loss = 1 - dice

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
    Weighted loss float Tensor
        Dice binary cross-entropy loss. If reduction is NONE, this has shape [batch_size, d0, .. dN-1]; otherwise, it is scalar.

    References
    -----------
    - Kaggle https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

    '''

    bce = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.AUTO)
    dice_l = dice_loss(y_true, y_pred)

    dice_bce = bce(y_true, y_pred) + dice_l

    return dice_bce


def soft_dice_loss(y_true, y_pred):
    '''

    Soft dice distance, measures dissimilarity between sample sets, it is complementary to the dice coefficient and it is obtained by subtracting the dice coefficient from 1. It is similar to dice loss but |X| + |Y| is obtained as y_true**2 + y_pred**2

    Parameters
    ----------
    y_true : Tensor
        input tensor with shape [batch_size, d0, .. dN].
    y_pred : Tensor
        predicted tensor with same shape of y_true.

    Returns
    -------
    Weighted loss float Tensor
       scalar dice loss tensor with shape=().

    References
    -----------
    - Wiki https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient

    '''

    smooth = 1
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    total = tf.reduce_sum(
        y_true**2, axis=[1, 2, 3]) + tf.reduce_sum(y_pred**2, axis=[1, 2, 3])
    dice = tf.reduce_mean((2. * intersection + smooth) / (total + smooth))

    return 1 - dice
