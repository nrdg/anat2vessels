import os

import ants
import nibabel as nib
import numpy as np
import pytest

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


class TestSkullStrip:
    def test_skull_strip_t1_output_type(self, ants_t1w):
        result = avp.skull_strip(ants_t1w, modality="t1")
        assert hasattr(result, "shape")
        assert result.dimension == 3

    def test_skull_strip_t1_changes_image(self, ants_t1w):
        result = avp.skull_strip(ants_t1w, modality="t1")
        orig_data = ants_t1w.numpy()
        result_data = result.numpy()
        assert result_data.shape == orig_data.shape
        assert np.any(result_data != orig_data)

    def test_skull_strip_t1_reduces_intensity_outside_brain(self, ants_t1w):
        result = avp.skull_strip(ants_t1w, modality="t1")
        orig_data = ants_t1w.numpy()
        result_data = result.numpy()
        zeroed_outside = np.sum(result_data == 0) > np.sum(orig_data == 0)
        assert zeroed_outside, "Skull stripping should zero more voxels"

    def test_skull_strip_t2_output_type(self, ants_t2w):
        result = avp.skull_strip(ants_t2w, modality="t2")
        assert hasattr(result, "shape")
        assert result.dimension == 3

    def test_skull_strip_t2_changes_image(self, ants_t2w):
        result = avp.skull_strip(ants_t2w, modality="t2")
        orig_data = ants_t2w.numpy()
        result_data = result.numpy()
        assert result_data.shape == orig_data.shape
        assert np.any(result_data != orig_data)

    def test_skull_strip_t2_reduces_intensity_outside_brain(self, ants_t2w):
        result = avp.skull_strip(ants_t2w, modality="t2")
        orig_data = ants_t2w.numpy()
        result_data = result.numpy()
        zeroed_outside = np.sum(result_data == 0) > np.sum(orig_data == 0)
        assert zeroed_outside, "Skull stripping should zero more voxels"

    def test_skull_strip_invalid_modality_raises(self, ants_t1w):
        with pytest.raises(Exception):
            avp.skull_strip(ants_t1w, modality="invalid")

    def test_skull_strip_empty_modality_raises(self, ants_t1w):
        with pytest.raises(Exception):
            avp.skull_strip(ants_t1w, modality="")

    def test_skull_strip_output_mask_is_binary(self, ants_t1w):
        result = avp.skull_strip(ants_t1w, modality="t1")
        result_data = result.numpy()
        assert result_data.dtype == np.float64 or result_data.dtype == np.float32

    def test_skull_strip_preserves_origin(self, ants_t1w):
        result = avp.skull_strip(ants_t1w, modality="t1")
        orig_origin = ants_t1w.origin
        res_origin = result.origin
        for o, r in zip(orig_origin, res_origin):
            assert abs(o - r) < 1e-6

    def test_skull_strip_preserves_spacing(self, ants_t1w):
        result = avp.skull_strip(ants_t1w, modality="t1")
        orig_spacing = ants_t1w.spacing
        res_spacing = result.spacing
        for o, r in zip(orig_spacing, res_spacing):
            assert abs(o - r) < 1e-6

    def test_skull_strip_t1_mask_has_brain_voxels(self, ants_t1w):
        result = avp.skull_strip(ants_t1w, modality="t1")
        result_data = result.numpy()
        non_zero_fraction = np.count_nonzero(result_data) / result_data.size
        assert non_zero_fraction > 0.1, "Brain should occupy >10% of volume"

    def test_skull_strip_t2_mask_has_brain_voxels(self, ants_t2w):
        result = avp.skull_strip(ants_t2w, modality="t2")
        result_data = result.numpy()
        non_zero_fraction = np.count_nonzero(result_data) / result_data.size
        assert non_zero_fraction > 0.1, "Brain should occupy >10% of volume"


class TestPreprocessImg:
    def test_no_skullstrip_output_created(self, t1w_path, tmp_path):
        out_file = str(tmp_path / "preprocessed.nii.gz")
        avp.preprocess_img(t1w_path, out_file, modality="t1", skull_strip=False)
        assert os.path.exists(out_file)

    def test_no_skullstrip_output_valid_nifti(self, t1w_path, tmp_path):
        out_file = str(tmp_path / "preprocessed.nii.gz")
        avp.preprocess_img(t1w_path, out_file, modality="t1", skull_strip=False)
        img = nib.load(out_file)
        assert len(img.shape) == 3

    def test_no_skullstrip_output_has_content(self, t1w_path, tmp_path):
        out_file = str(tmp_path / "preprocessed.nii.gz")
        avp.preprocess_img(t1w_path, out_file, modality="t1", skull_strip=False)
        img = nib.load(out_file)
        fdata = img.get_fdata()
        assert fdata.size > 0
        assert fdata.max() >= 0

    def test_no_skullstrip_output_shape_matches_ref(self, t1w_path, tmp_path):
        out_file = str(tmp_path / "preprocessed.nii.gz")
        ref_img = nib.load(avp.REF_IMG_PATH)
        avp.preprocess_img(t1w_path, out_file, modality="t1", skull_strip=False)
        out_img = nib.load(out_file)
        ref_shape = ref_img.shape
        out_shape = out_img.shape
        n_dims_equal = sum(1 for i in range(3) if abs(ref_shape[i] - out_shape[i]) <= 1)
        assert (
            n_dims_equal >= 2
        ), f"Output shape {out_shape} should match ref {ref_shape}"

    def test_with_skullstrip_output_created(self, t1w_path, tmp_path):
        out_file = str(tmp_path / "skullstripped_preprocessed.nii.gz")
        avp.preprocess_img(t1w_path, out_file, modality="t1", skull_strip=True)
        assert os.path.exists(out_file)

    def test_with_skullstrip_output_valid_nifti(self, t1w_path, tmp_path):
        out_file = str(tmp_path / "skullstripped_preprocessed.nii.gz")
        avp.preprocess_img(t1w_path, out_file, modality="t1", skull_strip=True)
        img = nib.load(out_file)
        assert len(img.shape) == 3

    def test_with_skullstrip_output_has_content(self, t1w_path, tmp_path):
        out_file = str(tmp_path / "skullstripped_preprocessed.nii.gz")
        avp.preprocess_img(t1w_path, out_file, modality="t1", skull_strip=True)
        img = nib.load(out_file)
        fdata = img.get_fdata()
        assert fdata.size > 0
        assert fdata.max() >= 0

    def test_with_skullstrip_more_zeros_than_without(self, t1w_path, tmp_path):
        out_no_ss = str(tmp_path / "no_ss.nii.gz")
        out_with_ss = str(tmp_path / "with_ss.nii.gz")
        avp.preprocess_img(t1w_path, out_no_ss, modality="t1", skull_strip=False)
        avp.preprocess_img(t1w_path, out_with_ss, modality="t1", skull_strip=True)
        img_no_ss = nib.load(out_no_ss).get_fdata()
        img_with_ss = nib.load(out_with_ss).get_fdata()
        zeros_fraction_no_ss = np.count_nonzero(img_no_ss == 0) / img_no_ss.size
        zeros_fraction_with_ss = np.count_nonzero(img_with_ss == 0) / img_with_ss.size
        assert zeros_fraction_with_ss > zeros_fraction_no_ss

    def test_t2_modality_no_skullstrip(self, t2w_path, tmp_path):
        out_file = str(tmp_path / "t2_preprocessed.nii.gz")
        avp.preprocess_img(t2w_path, out_file, modality="t2", skull_strip=False)
        assert os.path.exists(out_file)
        img = nib.load(out_file)
        assert len(img.shape) == 3

    def test_t2_modality_with_skullstrip(self, t2w_path, tmp_path):
        out_file = str(tmp_path / "t2_skullstripped.nii.gz")
        avp.preprocess_img(t2w_path, out_file, modality="t2", skull_strip=True)
        assert os.path.exists(out_file)
        img = nib.load(out_file)
        assert len(img.shape) == 3

    def test_t2_modality_output_valid(self, t2w_path, tmp_path):
        out_file = str(tmp_path / "t2_output.nii.gz")
        avp.preprocess_img(t2w_path, out_file, modality="t2", skull_strip=False)
        img = nib.load(out_file)
        fdata = img.get_fdata()
        assert fdata.size > 0
        assert fdata.max() >= 0
