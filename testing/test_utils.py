import hypothesis.strategies as st
from hypothesis import given

from MRIsegm.utils import get_slices
from MRIsegm.utils import get_rois
from MRIsegm.utils import mask_slices

import numpy as np
import glob
import pydicom


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


################################################################################
###                                                                          ###
###                              TESTING                                     ###
###                                                                          ###
################################################################################


@given(st.just('test_dcm'))
def test_get_slices(dir_path):
    '''
    Given :
        - .dcm dir_path

    So :
        - read, sort and rescale the slices

    And :
        - assert that the number of slices is equal to the number of dcm files
        - assert that the slices are ordered by instace number
        - assert that the slices are rescaled as uint8

    '''

    n_files = len(glob.glob(dir_path + '/*.dcm'))
    slices = get_slices(dir_path)

    maxval = slices.shape[0]
    N = np.random.randint(0, maxval)

    assert slices.shape[0] == n_files
    assert N == (pydicom.read_file(dir_path + '/{}'.format(N + 1) + '.dcm', force=True).get("InstanceNumber", 0) - 1)
    assert slices.dtype == np.uint8


@given(st.just('test_ROIs'))
def test_get_rois(rois_path):
    '''
    Given :
        - .roi dir_path

    So :
        - read .roi files and sort by "position" excluding "type":"composite"

    And :
        - assert that the dicts have no "type":composite
        - assert that the dicts are sorted in ascending order by "position"

    '''
    rois = get_rois(rois_path)

    maxval = len(rois)
    N = np.random.randint(0, maxval)

    positions = [rois[y].get("position") for y in range(len(rois))]

    assert rois[N].get("type") != "composite"
    assert positions == sorted(positions)


@given(st.just('test_dcm'), st.just('test_ROIs'))
def test_mask_slices(dir_path, rois_path):
    '''
    Given :
        - .dcm dir_path
        - .roi dir_path
    So :
        - make an array containing for each slice the proper mask: 0 (background) and 255 (label)

    And :
        - assert that the number of masked slices is equal to the number of found rois
        - assert that the mask is made only by 0 (background) and 255 (label)
        - assert that if no rois the mask is all black

    '''

    slices = get_slices(dir_path)
    rois = get_rois(rois_path)

    positions = [rois[z].get("position") for z in range(len(rois))]
    positions = [x - 1 for x in positions]

    masked = mask_slices(slices, rois)
    no_mask = list(set(np.arange(0, masked.shape[0], 1)) - set(positions))

    assert masked[positions].shape[0] == len(positions)
    assert set(np.unique(masked[positions])) == {0, 255}
    assert set(np.unique(masked[no_mask])) == {0}
