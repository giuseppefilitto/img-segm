import hypothesis.strategies as st
from hypothesis import given, settings

from MRIsegm.metrics import dice_coef
from MRIsegm.losses import DiceBCEloss, soft_dice_loss

import numpy as np
from random import choice
import tensorflow as tf


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']

################################################################################
###                                                                          ###
###                              STRATEGIES                                  ###
###                                                                          ###
################################################################################


@st.composite
def generate_true_tensor_strategy(draw):
    '''
    Generates a test tensor for y_true
    '''
    stack = np.zeros((8, 512, 512), dtype=None)

    k = np.random.randint(2, 8)
    a = (lambda x: 2 ** x)(k)
    block_shape = draw(st.tuples(*[st.just(a)] * 2))
    block = np.ones(block_shape, dtype=None)

    middle_point = 256
    half_width = block.shape[0] // 2

    x_sup = y_sup = middle_point - half_width
    x_inf = y_inf = middle_point + half_width

    stack[:, x_sup:x_inf, y_sup:y_inf] = block

    stack = np.expand_dims(stack, axis=-1)
    tensor = tf.convert_to_tensor(stack)

    return tensor



@st.composite
def generate_pred_tensor_strategy(draw):
    '''
    Generates a test tensor for y_pred
    '''
    stack = np.zeros((8, 512, 512), dtype=None)

    k = np.random.randint(2, 8)
    a = (lambda x: 2 ** x)(k)
    block_shape = draw(st.tuples(*[st.just(a)] * 2))
    block = np.ones(block_shape, dtype=None)

    middle_point = 256
    half_width = block.shape[0] // 2
    c = np.random.randint(a // 2, 128)

    middle_point_2 = choice([middle_point + c, middle_point - c])

    x_sup = y_sup = middle_point_2 - half_width
    x_inf = y_inf = middle_point_2 + half_width

    stack[:, x_sup:x_inf, y_sup:y_inf] = block

    stack = np.expand_dims(stack, axis=-1)
    tensor = tf.convert_to_tensor(stack)

    return tensor


################################################################################
###                                                                          ###
###                              TESTING                                     ###
###                                                                          ###
################################################################################


@given(generate_true_tensor_strategy(), generate_pred_tensor_strategy())
@settings(deadline=None)
def test_dice_coeff(y_true, y_pred):
    '''
    Given :
        - test tensor images
    So :
        - return the dice similarity coefficient [0, 1]

    And :
        - assert that dice_coeff() is 1.0 when y_true == y_pred
        - assert that dice_coeff() is close to 0 when y_pred == 0
        - assert that dice_coeff() is less than 1 when y_true and y_pred are not completely overlapped
        - assert that dice_coeff() is close yo 0.5 when y_pred==y_true is half-overlaping y_true
    '''

    dice_same = dice_coef(y_true, y_true)

    y_null = tf.math.multiply(y_true, 0)
    dice_null = dice_coef(y_true, y_null)

    dice_not_overlap = dice_coef(y_true, y_pred)

    block_l = np.where(y_true.numpy()[0, 256, :] == 1)
    size = block_l[0].size
    shift = size // 2
    y_half = np.roll(y_true.numpy(), shift, 1)

    dice_half_overlap = dice_coef(y_true, y_half)

    assert dice_same.numpy() == 1.0
    assert np.isclose(dice_null.numpy(), 0, atol=0.01)
    assert dice_not_overlap < 1.0
    assert np.isclose(dice_half_overlap.numpy(), 0.5, atol=0.001)



@given(generate_true_tensor_strategy(), generate_pred_tensor_strategy())
@settings(deadline=None)
def test_DiceBCEloss(y_true, y_pred):
    '''
    Given :
        - test tensor images
    So :
        - return the Dice Binary Cross-Entropy Loss

    And :
        - assert that DiceBCEloss is 0 when y_true == y_pred
        - assert that DiceBCEloss is greater than 0 when y_true and y_pred are not completely overlapped
        - assert that DiceBCEloss is greater than 1 when y_pred == 0 and y_true is not == 0
    '''


    loss_t = DiceBCEloss(y_true, y_true)
    loss_p = DiceBCEloss(y_pred, y_pred)
    loss_not_overlap = DiceBCEloss(y_true, y_pred)

    y_null = tf.math.multiply(y_true, 0)
    loss_null = DiceBCEloss(y_true, y_null)

    assert loss_t.numpy() == 0 and loss_p.numpy() == 0
    assert loss_not_overlap.numpy() > 0
    assert loss_null.numpy() > 1


@given(generate_true_tensor_strategy(), generate_pred_tensor_strategy())
@settings(deadline=None)
def test_soft_dice_loss(y_true, y_pred):
    '''
    Given :
        - test tensor images
    So :
        - return the soft_dice_loss [0, 1]

    And :
        - assert that the soft_dice_loss is 0 when y_true == y_pred
        - assert that soft_dice_loss is close to 1 when y_pred == 0
        - assert that soft_dice_loss is greater than 0 when y_true and y_pred are not completely overlapped
        - assert that soft_dice_loss is close yo 0.5 when y_pred==y_true is half-overlapping y_true
    '''


    dice_loss_same = soft_dice_loss(y_true, y_true)

    y_null = tf.math.multiply(y_true, 0)
    dice_loss_null = soft_dice_loss(y_true, y_null)

    dice_loss_not_overlap = dice_coef(y_true, y_pred)

    block_l = np.where(y_true.numpy()[0, 256, :] == 1)
    size = block_l[0].size
    shift = size // 2
    y_half = np.roll(y_true.numpy(), shift, 1)

    dice_loss_half = soft_dice_loss(y_true, y_half)

    assert dice_loss_same.numpy() == 0
    assert np.isclose(dice_loss_null.numpy(), 1, atol=0.01)
    assert dice_loss_not_overlap > 0
    assert np.isclose(dice_loss_half.numpy(), 0.5, atol=0.001)
