import hypothesis.strategies as st
from hypothesis import assume, given, settings

from hypothesis import HealthCheck as HC

from MRIsegm.utils import get_slices
from MRIsegm.metrics import dice_coef
from MRIsegm.losses import DiceBCEloss

from MRIsegm.processing import denoise_slices
from MRIsegm.processing import resize_slices
from MRIsegm.processing import predict_slices
from MRIsegm.processing import contour_slices
from MRIsegm.processing import pre_processing_data
from MRIsegm.processing import predict_images

import glob
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import model_from_json
from segmentation_models.losses import DiceLoss, BinaryFocalLoss

import random

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
    dir = draw(st.just('testing/test_dcm'))
    slices = get_slices(dir)

    weights_dir = 'data/models/weights'

    arch_list = glob.glob(weights_dir + '/*256*.json')
    chosen_arch = draw(st.just(random.choice(arch_list)))

    model_weights = chosen_arch.replace('.json', '_weights.h5')


    with open(chosen_arch) as json_file:
        data = json.load(json_file)

    model = model_from_json(data)
    model.load_weights(model_weights)

    optimizer = 'Adam'
    loss = DiceBCEloss
    metrics = [dice_coef]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    IMG_SIZE = draw(st.tuples(*[st.just(256)] * 2))

    predicted = predict_slices(slices, model, *IMG_SIZE)

    return predicted


@st.composite
def get_model_strategy(draw):
    '''
    Load a model from saved weights
    '''

    weights_dir = 'data/models/weights'

    arch_list = glob.glob(weights_dir + '/*256*.json')
    chosen_arch = draw(st.just(random.choice(arch_list)))

    model_weights = chosen_arch.replace('.json', '_weights.h5')


    with open(chosen_arch) as json_file:
        data = json.load(json_file)

    model = model_from_json(data)
    model.load_weights(model_weights)

    optimizer = 'Adam'
    loss = DiceBCEloss
    metrics = [dice_coef]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


@st.composite
def get_model2_strategy(draw):
    '''
    Load a model from saved weights
    '''

    weights_dir = 'data/models/weights'

    arch_list = glob.glob(weights_dir + '/*1binary*.json')
    chosen_arch = draw(st.just(random.choice(arch_list)))

    model_weights = chosen_arch.replace('.json', '_weights.h5')

    with open(chosen_arch) as json_file:
        data = json.load(json_file)

    model = model_from_json(data)
    model.load_weights(model_weights)

    optimizer = 'adam'

    dice_loss = DiceLoss()
    focal_loss = BinaryFocalLoss()
    loss = dice_loss + (1 * focal_loss)

    metrics = [dice_coef]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

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
    path = draw(st.just('testing/test_dcm'))
    slices = get_slices(path)

    return slices


@st.composite
def rand_stack_strategy(draw):
    '''
    Generates a stack of N 512x512 white noise 8-bit images
    '''
    N = draw(st.integers(5, 30))
    stack = (255 * np.random.rand(N, 512, 512)).astype(np.uint8)
    return stack


################################################################################
###                                                                          ###
###                              TESTING                                     ###
###                                                                          ###
################################################################################


@given(rand_stack_strategy(), st.tuples(*[st.floats(2, 10)] * 2))
@settings(max_examples=4, deadline=None, suppress_health_check=(HC.too_slow,))
def test_denoise_slices(slices, alphas):
    '''
    Given :
        - stack of slices
        - denoising paramters alphas

    So :
        - denoise each slices of a stack

    And :
        - assert that the snr is higher after denoising
        - assert that increasing alpha the smoothness increase assuming alphas[0] < alphas[1]
    '''

    assume(alphas[0] < alphas[1])

    denoised_0 = denoise_slices(slices, alphas[0])
    denoised_1 = denoise_slices(slices, alphas[1])

    m_noisy = slices.mean()
    sd_noisy = slices.std()

    snr_noisy = np.where(sd_noisy == 0, 0, m_noisy / sd_noisy)

    m_0 = denoised_0.mean()
    sd_0 = denoised_0.std()

    snr_0 = np.where(sd_0 == 0, 0, m_0 / sd_0)

    m_1 = denoised_1.mean()
    sd_1 = denoised_1.std()

    snr_1 = np.where(sd_1 == 0, 0, m_1 / sd_1)

    assert snr_0 > snr_noisy and snr_1 > snr_noisy
    assert snr_0 < snr_1



@given(rand_stack_strategy(), img_size_strategy(), st.sampled_from(legitimate_dtypes))
@settings(max_examples=10, deadline=None)
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



@given(st.just('testing/test_dcm'), get_model_strategy(), st.tuples(*[st.just(256)] * 2))
@settings(max_examples=10, deadline=None)
def test_predict_slices(path, model, IMG_SIZE):
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
    slices = get_slices(path)
    predicted = predict_slices(slices, model, *IMG_SIZE)

    predicted_1 = predict_slices(slices, model, *IMG_SIZE)

    assert predicted.shape[1:3] == IMG_SIZE
    assert (predicted.all() >= 0.) & (predicted.all() <= 1.)
    assert np.isclose(predicted, predicted_1).all()



@given(get_slices_strategy(), get_predicted_strategy(), st.tuples(*[st.just(256)] * 2))
@settings(max_examples=10, deadline=None)
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
        - assert that the contour is red
    '''

    resized = resize_slices(slices, *IMG_SIZE)
    contoured = contour_slices(resized, predicted_slices)

    blu_ch = contoured[..., 0]
    green_ch = contoured[..., 1]

    assert contoured.shape[0] == predicted_slices.shape[0] == slices.shape[0]
    assert contoured.shape[0:3] == predicted_slices.shape[0:3]
    assert contoured.shape[-1] == 3
    assert blu_ch.all() == green_ch.all() == 0


@given(get_slices_strategy(), st.tuples(*[st.floats(2, 10)] * 2))
@settings(max_examples=10, deadline=None, suppress_health_check=(HC.too_slow,))
def test_pre_processing_data(slices, alphas):
    '''
    Given :
        - stack of slices
        - denoising paramters alphas

    So :
        - pre-process each slice of a stack

    And :
        - assert that the snr is higher after denoising
        - assert that increasing alpha the smoothness increase assuming alphas[0] < alphas[1]
        - assert that the image size is (512, 512)
        - assert that the image dtype is float32
        - assert that the stack is 4-dimensional
    '''

    assume(alphas[0] < alphas[1])

    preprocessed_0 = pre_processing_data(slices, alphas[0])
    preprocessed_1 = pre_processing_data(slices, alphas[1])

    m_noisy = slices.mean()
    sd_noisy = slices.std()

    snr_noisy = np.where(sd_noisy == 0, 0, m_noisy / sd_noisy)

    m_0 = preprocessed_0.mean()
    sd_0 = preprocessed_0.std()

    snr_0 = np.where(sd_0 == 0, 0, m_0 / sd_0)

    m_1 = preprocessed_1.mean()
    sd_1 = preprocessed_1.std()

    snr_1 = np.where(sd_1 == 0, 0, m_1 / sd_1)

    assert snr_0 > snr_noisy and snr_1 > snr_noisy
    assert snr_0 < snr_1
    assert preprocessed_0.shape[1:3] == preprocessed_1.shape[1:3] == (512, 512)
    assert preprocessed_0.dtype == preprocessed_0.dtype == np.float32
    assert len(preprocessed_0.shape) == len(preprocessed_0.shape) == 4


@given(get_slices_strategy(), get_model2_strategy())
@settings(max_examples=10, deadline=None)
def test_predict_images(slices, model):
    '''
    Given :
        - stack of slices
        - loaded .h5 model

    So :
        - pre-process the stack and return the predicted stack

    And :
        - assert that the predicted stack has the right image size
        - assert that the prediction is in the range [0, 1]
        - assert that the prediction of same models is the same
    '''

    predicted = predict_images(slices, model, pre_processing=True)
    predicted_1 = predict_images(slices, model, pre_processing=True)


    assert predicted.shape[1:3] == predicted_1.shape[1:3] == (512, 512)
    assert (predicted.all() >= 0.) & (predicted.all() <= 1.)
    assert predicted.shape == predicted_1.shape
    assert np.isclose(predicted, predicted_1).all()
