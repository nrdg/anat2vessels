import pytest
import ants
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


@pytest.fixture
def ants_ref(ref_path):
    return ants.image_read(ref_path)


@pytest.fixture
def ants_t1w(t1w_path):
    return ants.image_read(t1w_path)


@pytest.fixture
def ants_t2w(t2w_path):
    return ants.image_read(t2w_path)
