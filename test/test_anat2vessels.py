from anat2vessels import features as avf
from anat2vessels import preprocess as avp
from anat2vessels import vessels2csv as csv
from anat2vessels.data import fetch_ref_img, fetch_test_data


def test_smoke():
    avf
    avp
    csv
    fetch_ref_img
    fetch_test_data
    pass
