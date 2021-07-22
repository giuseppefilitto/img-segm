import hypothesis.strategies as st
from hypothesis import given, settings

from MRIsegm.metrics import dice_coef
from MRIsegm.losses import DiceBCEloss

import numpy as np
import tensorflow as tf


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']

################################################################################
###                                                                          ###
###                              STRATEGIES                                  ###
###                                                                          ###
################################################################################


@st.composite
def generate_test_tensor_strategy(draw):
    '''
    Generate test tensor with shape: [N, height, width, 1] of 0 and 1
    '''
    N = draw(st.integers(10, 50))
    stack = np.zeros((N, 512, 512), dtype=None)

    k = np.random.randint(2, 8)
    a = (lambda x: 2 ** x)(k)
    block_shape = draw(st.tuples(*[st.just(a)] * 2))
    block = np.ones(block_shape, dtype=None)

    x_inf = 256 - block.shape[0] // 2
    y_inf = 256 - block.shape[0] // 2

    x_sup = 256 + block.shape[0] // 2
    y_sup = 256 + block.shape[0] // 2

    stack[:, x_inf:x_sup, y_inf:y_sup] = block

    stack = np.expand_dims(stack, axis=-1)
    tensor = tf.convert_to_tensor(stack)

    return tensor
################################################################################
###                                                                          ###
###                              TESTING                                     ###
###                                                                          ###
################################################################################


@given(generate_test_tensor_strategy())
def test_dice_coeff(tensor):
    '''
    Given :
        - test tensor images of 0 and 1
    So :
        - return the dice similarity coefficient [0, 1]

    And :
        - assert that dice_coeff() if 1.0 when y_true == y_pred
        - assert that dice_coeff() is close to 0 when y_pred == 0
    '''

    y_true = tensor
    y_pred = y_true * 0

    dice_same = dice_coef(y_true, y_true)
    dice_null = dice_coef(y_true, y_pred)

    assert dice_same.numpy() == 1.0
    assert np.isclose(dice_null.numpy(), 0, atol=0.001)



@given(generate_test_tensor_strategy())
@settings(deadline=None)
def test_DiceBCEloss(tensor):
    '''
    Given :
        - test tensor images of 0 and 1
    So :
        - return the Dice Binary Cross-Entropy Loss

    And :
        - assert that DiceBCEloss is 0 when y_true == y_pred
    '''

    y_true = tensor
    y_pred = y_true * 0


    loss = DiceBCEloss(y_true, y_true)
    loss_null = DiceBCEloss(y_pred, y_pred)

    assert loss.numpy() == 0
    assert loss_null.numpy() == 0
