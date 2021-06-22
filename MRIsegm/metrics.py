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


def iou(y_true, y_pred, smooth=1):
    '''

    Intersection over union (IoU) also know as Jaccard coefficient, measures similarity between finite sample sets. Defined as the size of the intersection divided by the size of the union of the sample sets.

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
    iou: float
        Intersection over union. The index is a number between 0 and 1 , if 1 sets totally match.

    References
    -----------
    - Wiki https://en.wikipedia.org/wiki/Jaccard_index


    '''

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = tf.reduce_mean((intersection + smooth) / (union + smooth))
    return iou


def tversky_index(y_true, y_pred, smooth=1, alpha=0.7):
    '''

     The Tversky index is an asymmetric similarity measure on sets that compares a variant to a prototype. The Tversky index can be seen as a generalization of the Sørensen–Dice coefficient and the Jaccard index. Setting alpha = beta = 1 produces the jaccard coefficient, setting alpha = beta = 0.5 produces the Sørensen–Dice coefficient


    Parameters
    ----------
    y_true : Tensor
        input tensor with shape: [batch_size, height, width, channels].
    y_pred : Tensor
        predicted tensor with shape: [batch_size, height, width, channels].
    smooth : int, float, optional
        value that will be added to the numerator and denominator to avoid 0 division, by default 1.
    alpha: float, optional
        parameter  of the Tversky index, by default 0.7.

    Returns
    -------
    tversky: float
        Tversky index. The index is a number between 0 and 1 , if 1 sets totally match.

    References
    -----------
    - Wiki https://en.wikipedia.org/wiki/Tversky_index


    '''

    beta = 1 - alpha  # since alpha + beta = 1 cases are of more interest

    true_pos = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    false_neg = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    false_pos = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    tversky = tf.reduce_mean((true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth))

    return tversky
