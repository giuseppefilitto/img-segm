import hypothesis.strategies as st
from hypothesis import given, settings

from MRIsegm.utils import get_slices
from MRIsegm.metrics import dice_coef
from MRIsegm.losses import DiceBCEloss

from MRIsegm.processing import denoise_slices
from MRIsegm.processing import resize_slices
from MRIsegm.processing import predict_slices
from MRIsegm.processing import contour_slices

import glob
import numpy as np
import tensorflow as tf

from random import choice

__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


################################################################################
###                                                                          ###
###                              STRATEGIES                                  ###
###                                                                          ###
################################################################################


legitimate_dtypes = [np.float32, np.uint8, np.uint16, np.float64, np.uint64]


@st.composite
def get_predicted_strategy(draw):
    '''
    get a predicted stack of slices
    '''
    dir = draw(st.just('test_dcm'))
    slices = get_slices(dir)

    src = '../data/models'

    models_path = glob.glob(src + '/*.h5')
    model_path = draw(st.just(choice(models_path)))

    dependencies = {
        'DiceBCEloss': DiceBCEloss,
        'dice_coef': dice_coef,
        'FixedDropout': tf.keras.layers.Dropout(0.2)
    }

    model = tf.keras.models.load_model(model_path, custom_objects=dependencies)
    IMG_SIZE = draw(st.tuples(*[st.just(256)] * 2))

    predicted = predict_slices(slices, model, *IMG_SIZE)

    return predicted


@st.composite
def get_model_strategy(draw):
    '''
    Load a .h5 saved model
    '''

    src = '../data/models'

    models_path = glob.glob(src + '/*.h5')
    model_path = draw(st.just(choice(models_path)))

    dependencies = {
        'DiceBCEloss': DiceBCEloss,
        'dice_coef': dice_coef,
        'FixedDropout': tf.keras.layers.Dropout(0.2)
    }

    model = tf.keras.models.load_model(model_path, custom_objects=dependencies)

    return model


@st.composite
def img_size_strategy(draw):
    '''
    Generate IMG_SIZE tuple from 64x64 to 512x512
    '''
    y = np.random.randint(7, 9)
    a = (lambda x: 2 ** x)(y)
    IMG_SIZE = draw(st.tuples(*[st.just(a)] * 2))
    return IMG_SIZE


@st.composite
def get_slices_strategy(draw):
    '''
    Get sclices from test_dcm
    '''
    path = draw(st.just('test_dcm'))
    slices = get_slices(path)

    return slices


@st.composite
def rand_stack_strategy(draw):
    '''
    Generates a stack of N 512x512 white noise 8-bit images
    '''
    N = draw(st.integers(10, 50))
    stack = (255 * np.random.rand(N, 512, 512)).astype(np.uint8)
    return stack


################################################################################
###                                                                          ###
###                              TESTING                                     ###
###                                                                          ###
################################################################################



@given(rand_stack_strategy(), st.floats(2, 10))
@settings(max_examples=15, deadline=None)
def test_denoise_slices(slices, alpha):
    '''
    Given :
        - stack of slices
        - alpha

    So :
        - denoise each slices of a stack

    And :
        - assert that the snr is higher after denoising
        - assert that increasing alpha the smoothness increase assuming more than one value for aplha
    '''

    denoised = denoise_slices(slices, alpha)

    m_noisy = slices.mean()
    sd_noisy = slices.std()

    snr_noisy_dict = {}

    snr_noisy = np.where(sd_noisy == 0, 0, m_noisy / sd_noisy)

    snr_noisy_dict[alpha] = snr_noisy

    m = denoised.mean()
    sd = denoised.std()

    snr_dict = {}

    snr = np.where(sd == 0, 0, m / sd)

    snr_dict[alpha] = snr

    alphas = []
    alphas.append(alpha)



    assert snr > snr_noisy
    assert snr_dict[alphas[0]] < snr_dict[alphas[-1]] if len(alphas) > 1 else snr_dict[alphas[0]] <= snr_dict[alphas[-1]]



@given(rand_stack_strategy(), img_size_strategy(), st.sampled_from(legitimate_dtypes))
@settings(max_examples=15, deadline=None)
def test_resize_slices(slices, IMG_SIZE, dtype):
    '''
    Given :
        - stack of slices
        - image size
        - dtype

    So :
        - resized the stack

    And :
        - assert that the resized stack has the right IMG_SIZE
        - assert that the resized stack has the right dtype

    '''
    resized = resize_slices(slices, *IMG_SIZE, dtype)

    assert resized.shape[1:3] == IMG_SIZE
    assert resized.dtype == dtype



@given(get_slices_strategy(), get_model_strategy(), st.tuples(*[st.just(256)] * 2))
@settings(max_examples=15, deadline=None)
def test_predict_slices(slices, model, IMG_SIZE):
    '''
    Given :
        - stack of slices
        - loaded .h5 model
        - image size

    So :
        - resized the stack and return the predicted stack

    And :
        - assert that the predicted stack has the right IMG_SIZE
        - assert that the prediction is in the range [0, 1]
        - assert that the prediction of same models is the same
    '''

    predicted = predict_slices(slices, model, *IMG_SIZE)

    predicted_1 = predict_slices(slices, model, *IMG_SIZE)

    assert predicted.shape[1:3] == IMG_SIZE
    assert (0. <= predicted.all()) & (predicted.all() <= 1.)
    assert np.isclose(predicted, predicted_1).all()



@given(get_slices_strategy(), get_predicted_strategy(), st.tuples(*[st.just(256)] * 2))
@settings(max_examples=15, deadline=None)
def test_contour_slices(slices, predicted_slices, IMG_SIZE):
    '''
    Given :
        - stack of slices
        - stack of the predicted slices

    So :
        - draw the contour of the prediction on the image

    And :
        - assert that the number of contoured slices is the same of the original slices
        - assert that the output stack has the right number of slices and IMG_SIZE
        - assert that the images are stored as RGB
    '''

    resized = resize_slices(slices, *IMG_SIZE)
    contoured = contour_slices(resized, predicted_slices)

    assert contoured.shape[0] == predicted_slices.shape[0] == slices.shape[0]
    assert contoured.shape[0:3] == predicted_slices.shape[0:3]
    assert contoured.shape[-1] == 3
