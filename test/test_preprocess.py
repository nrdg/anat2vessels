import os

import ants
import nibabel as nib

from anat2vessels import preprocess as avp
from anat2vessels.data.fetch import fetch_test_data, REGISTRY


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

    def test_caching_returns_same_path(self):
        data1 = fetch_test_data()
        data2 = fetch_test_data()
        assert data1["t1w"] == data2["t1w"]
        assert data1["t2w"] == data2["t2w"]

    def test_cache_dir_exists(self):
        cache_dir = REGISTRY.abspath
        assert os.path.isdir(cache_dir)

    def test_cached_files_exist(self):
        t1w_path = fetch_test_data()["t1w"]
        assert os.path.isfile(t1w_path)

    def test_t1w_valid_nifti(self):
        data = fetch_test_data()
        img = nib.load(data["t1w"])
        assert len(img.shape) == 3
        assert img.shape == (274, 384, 384)

    def test_t2w_valid_nifti(self):
        data = fetch_test_data()
        img = nib.load(data["t2w"])
        assert len(img.shape) == 3
        assert img.shape == (274, 384, 384)

    def test_t1w_can_be_read_by_ants(self):
        data = fetch_test_data()
        img = ants.image_read(data["t1w"])
        assert img.dimension == 3

    def test_t2w_can_be_read_by_ants(self):
        data = fetch_test_data()
        img = ants.image_read(data["t2w"])
        assert img.dimension == 3

    def test_t1w_data_range(self):
        data = fetch_test_data()
        img = nib.load(data["t1w"])
        fdata = img.get_fdata()
        assert fdata.min() >= 0
        assert fdata.max() >= 1

    def test_t2w_data_range(self):
        data = fetch_test_data()
        img = nib.load(data["t2w"])
        fdata = img.get_fdata()
        assert fdata.min() >= 0
        assert fdata.max() >= 1


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
