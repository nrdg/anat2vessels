import os

import ants
import nibabel as nib

from anat2vessels import preprocess as avp
from anat2vessels.data.fetch import fetch_test_data


class TestFetchTestData:
    def test_download_returns_dict(self):
        data = fetch_test_data()
        assert isinstance(data, dict)

    def test_download_has_t1w_key(self):
        data = fetch_test_data()
        assert "t1w" in data

    def test_download_has_t2w_key(self):
        data = fetch_test_data()
        assert "t2w" in data

    def test_download_t1w_file_exists(self):
        data = fetch_test_data()
        assert os.path.exists(data["t1w"])

    def test_download_t2w_file_exists(self):
        data = fetch_test_data()
        assert os.path.exists(data["t2w"])

    def test_download_t1w_is_nifti(self):
        data = fetch_test_data()
        assert data["t1w"].endswith(".nii.gz")

    def test_download_t2w_is_nifti(self):
        data = fetch_test_data()
        assert data["t2w"].endswith(".nii.gz")


class TestRefImgPath:
    def test_ref_img_path_exists(self):
        assert os.path.exists(avp.REF_IMG_PATH)

    def test_ref_img_path_is_nifti(self):
        assert avp.REF_IMG_PATH.endswith(".nii.gz")

    def test_ref_img_is_valid_nifti(self):
        img = nib.load(avp.REF_IMG_PATH)
        assert len(img.shape) == 3

    def test_ref_img_can_be_read_by_ants(self):
        img = ants.image_read(avp.REF_IMG_PATH)
        assert img.dimension == 3
