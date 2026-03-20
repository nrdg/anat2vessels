import numpy as np
import pytest

from anat2vessels import features as avf


@pytest.fixture
def skeleton():
    vol = np.zeros((5, 5, 5))
    vol[2, 2, 2] = 1
    vol[2, 2, 1] = 1
    return vol


def test_num_neighbors(skeleton):
    neighbor_count = avf.num_neighbors(skeleton)
    assert neighbor_count[2, 2, 2] == 1


def test_bifurcation_endpoint_arrays(skeleton):
    bifurcations, endpoints = avf.bifurcation_endpoint_arrays(skeleton)
    assert bifurcations[2, 2, 2] == 1
