import numpy as np
from anat2vessels import features as avf


def test_num_neighbors():
    skeleton = np.zeros((5, 5, 5), dtype=np.uint8)
    skeleton[2, 2, 1] = 1
    skeleton[2, 2, 2] = 1
    skeleton[2, 2, 3] = 1
    skeleton[2, 1, 2] = 1
    skeleton[2, 3, 2] = 1

    neighbor_count = avf.num_neighbors(skeleton)

    assert neighbor_count[2, 2, 2] == 4
    assert neighbor_count[2, 2, 1] == 1
    assert neighbor_count[2, 2, 3] == 1
    assert neighbor_count[2, 1, 2] == 1
    assert neighbor_count[2, 3, 2] == 1


def test_bifurcation_endpoint_arrays():
    skeleton = np.zeros((5, 5, 5), dtype=np.uint8)
    skeleton[2, 2, 1] = 1
    skeleton[2, 2, 2] = 1
    skeleton[2, 2, 3] = 1
    skeleton[2, 1, 2] = 1
    skeleton[2, 3, 2] = 1

    bifurcations, endpoints = avf.bifurcation_endpoint_arrays(skeleton)

    assert bifurcations[2, 2, 2] == 1
    assert endpoints[2, 2, 1] == 1
    assert endpoints[2, 2, 3] == 1
    assert endpoints[2, 1, 2] == 1
    assert endpoints[2, 3, 2] == 1
