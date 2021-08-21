import tensorflow as tf


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def dice_coef(y_true, y_pred, smooth=1):
    '''
    Dice coefficient, also know as Sørensen-Dice index, is used to gauge the similarity of two samples. Given 2 sets it is defined as  twice the number of elements common to both sets divided by the sum of the number of elements in each set.

    Parameters
    ----------
    y_true : Tensor
        input tensor with shape: [batch_size, height, width, channels].
    y_pred : Tensor
        predicted tensor with shape: [batch_size, height, width, channels].
    smooth : int, float, optional
        value that will be added to the numerator and denominator to avoid 0 division, by default 1.

    Returns
    -------
    dice: float
        dice coefficient. The index is a number between 0 and 1 , if 1 sets totally match.

    References
    -----------
    - Wiki https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient


    '''

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    total = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tf.reduce_mean((2. * intersection + smooth) / (total + smooth))
    return dice
