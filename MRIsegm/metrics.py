import tensorflow as tf


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def dice_coe(output, target, axis=(1, 2, 3), smooth=1e-5):
    """

    Sørensen coefficient for comparing the similarity of two batch of data,
    usually be used for binary image segmentation
    i.e. labels are binary.
    The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    axis : tuple of int
         default (1,2,3).
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    References
    -----------
    Wiki <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>

    """
    inse = tf.reduce_sum(output * target, axis=axis)

    l = tf.reduce_sum(output * output, axis=axis)
    r = tf.reduce_sum(target * target, axis=axis)

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice


def jacc_index(output, target, axis=(1, 2, 3), smooth=1e-5):
    """

    jaccard index for comparing the similarity of two batch of data,
    usually be used for binary image segmentation
    i.e. labels are binary.
    The coefficient between 0 to 1, 1 means totally match.

    jaccard index 'J' also know as 'intersection over union' is obtained from dice coefficeint 'D' by the following formula:

                            J = D / (2 - D)


    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    axis : tuple of int
         default (1,2,3).
    smooth : float
        This small value will be added to the numerator and denominator of the dice_coe function which the jaccard index is calculated from..

    References
    -----------
    Wiki <https://en.wikipedia.org/wiki/Jaccard_index>

    """

    dice = dice_coe(output, target, axis=(1, 2, 3), smooth=1e-5)
    jacc = dice / (2 - dice)
    return jacc


def jacc_loss(output, target, axis=(1, 2, 3), smooth=1e-5):
    """

    The Jaccard distance, which measures dissimilarity between sample sets, is complementary to the Jaccard coefficient and is obtained by subtracting the Jaccard coefficient from 1


    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    axis : tuple of int
         default (1,2,3).
    smooth : float
        This small value will be added to the numerator and denominator of the dice_coe function which the jaccard index is calculated from.

    References
    -----------
    Wiki <https://en.wikipedia.org/wiki/Jaccard_index>

    """

    jacc = jacc_index(output, target, axis=(1, 2, 3), smooth=1e-5)

    return 1 - jacc
