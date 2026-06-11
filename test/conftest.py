import os

import ants
import numpy as np
import pytest

from anat2vessels.data.fetch import fetch_ref_img, fetch_test_data


@pytest.fixture(scope="session")
def ref_path():
    return fetch_ref_img()


@pytest.fixture(scope="session")
def t1w_path():
    return fetch_test_data()["t1w"]


@pytest.fixture(scope="session")
def t2w_path():
    return fetch_test_data()["t2w"]


@pytest.fixture(scope="session")
def ants_ref(ref_path):
    return ants.image_read(ref_path)


@pytest.fixture(scope="session")
def ants_t1w(t1w_path):
    return ants.image_read(t1w_path)


@pytest.fixture(scope="session")
def ants_t2w(t2w_path):
    return ants.image_read(t2w_path)


def _make_small_image(image_path, cache_dir):
    """Create a 4x-downsampled copy of an image and cache it."""
    base = os.path.basename(image_path).replace(".nii.gz", "_small.nii.gz")
    small_path = os.path.join(cache_dir, base)
    if not os.path.exists(small_path):
        img = ants.image_read(image_path)
        target_voxels = tuple(max(d // 4, 10) for d in img.shape)
        small = ants.resample_image(img, target_voxels, use_voxels=True, interp_type=0)
        ants.image_write(small, small_path)
    return small_path


@pytest.fixture(scope="session")
def small_t1w_path(t1w_path, tmp_path_factory):
    cache_dir = str(tmp_path_factory.mktemp("small"))
    return _make_small_image(t1w_path, cache_dir)


@pytest.fixture(scope="session")
def small_t2w_path(t2w_path, tmp_path_factory):
    cache_dir = str(tmp_path_factory.mktemp("small"))
    return _make_small_image(t2w_path, cache_dir)


@pytest.fixture(scope="session")
def small_ref_path(ref_path, tmp_path_factory):
    cache_dir = str(tmp_path_factory.mktemp("small"))
    return _make_small_image(ref_path, cache_dir)


# ---------------------------------------------------------------------------
# Features test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def straight_line_skeleton():
    """A 5-voxel straight line along z at (3,3,1:6) in a 7x7x7 array."""
    arr = np.zeros((7, 7, 7), dtype=np.uint8)
    arr[3, 3, 1:6] = 1
    return arr


@pytest.fixture(scope="session")
def t_shape_skeleton():
    """A T-junction: vertical 3 voxels + horizontal 5 voxels at intersection."""
    arr = np.zeros((9, 9, 9), dtype=np.uint8)
    arr[4, 4, 2:5] = 1
    arr[4, 2:7, 4] = 1
    return arr
