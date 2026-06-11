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
    def test_no_skullstrip_output_created(
        self, small_t1w_path, small_ref_path, tmp_path
    ):
        out_file = str(tmp_path / "preprocessed.nii.gz")
        avp.preprocess_img(
            small_t1w_path,
            out_file,
            modality="t1",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        assert os.path.exists(out_file)

    def test_no_skullstrip_output_valid_nifti(
        self, small_t1w_path, small_ref_path, tmp_path
    ):
        out_file = str(tmp_path / "preprocessed.nii.gz")
        avp.preprocess_img(
            small_t1w_path,
            out_file,
            modality="t1",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        img = nib.load(out_file)
        assert len(img.shape) == 3

    def test_no_skullstrip_output_has_content(
        self, small_t1w_path, small_ref_path, tmp_path
    ):
        out_file = str(tmp_path / "preprocessed.nii.gz")
        avp.preprocess_img(
            small_t1w_path,
            out_file,
            modality="t1",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        img = nib.load(out_file)
        fdata = img.get_fdata()
        assert fdata.size > 0
        assert fdata.max() >= 0

    def test_no_skullstrip_output_spacing_matches_ref(
        self, small_t1w_path, small_ref_path, tmp_path
    ):
        out_file = str(tmp_path / "preprocessed.nii.gz")
        avp.preprocess_img(
            small_t1w_path,
            out_file,
            modality="t1",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        ref_img = ants.image_read(small_ref_path)
        out_img = ants.image_read(out_file)
        for i in range(3):
            assert (
                abs(ref_img.spacing[i] - out_img.spacing[i]) < 1e-4
            ), f"Spacing mismatch at dim {i}"

    def test_with_skullstrip_output_created(self, t1w_path, tmp_path):
        out_file = str(tmp_path / "skullstripped_preprocessed.nii.gz")
        avp.preprocess_img(t1w_path, out_file, modality="t1", do_skull_strip=True)
        assert os.path.exists(out_file)

    def test_with_skullstrip_output_valid_nifti(self, t1w_path, tmp_path):
        out_file = str(tmp_path / "skullstripped_preprocessed.nii.gz")
        avp.preprocess_img(t1w_path, out_file, modality="t1", do_skull_strip=True)
        img = nib.load(out_file)
        assert len(img.shape) == 3

    def test_with_skullstrip_output_has_content(self, t1w_path, tmp_path):
        out_file = str(tmp_path / "skullstripped_preprocessed.nii.gz")
        avp.preprocess_img(t1w_path, out_file, modality="t1", do_skull_strip=True)
        img = nib.load(out_file)
        fdata = img.get_fdata()
        assert fdata.size > 0
        assert fdata.max() >= 0

    def test_t2_modality_no_skullstrip(self, small_t2w_path, small_ref_path, tmp_path):
        out_file = str(tmp_path / "t2_preprocessed.nii.gz")
        avp.preprocess_img(
            small_t2w_path,
            out_file,
            modality="t2",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        assert os.path.exists(out_file)
        img = nib.load(out_file)
        assert len(img.shape) == 3

    def test_t2_modality_with_skullstrip(self, t2w_path, tmp_path):
        out_file = str(tmp_path / "t2_skullstripped.nii.gz")
        avp.preprocess_img(t2w_path, out_file, modality="t2", do_skull_strip=True)
        assert os.path.exists(out_file)
        img = nib.load(out_file)
        assert len(img.shape) == 3

    def test_t2_modality_output_valid(self, small_t2w_path, small_ref_path, tmp_path):
        out_file = str(tmp_path / "t2_output.nii.gz")
        avp.preprocess_img(
            small_t2w_path,
            out_file,
            modality="t2",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        img = nib.load(out_file)
        fdata = img.get_fdata()
        assert fdata.size > 0
        assert fdata.max() >= 0

    def test_output_can_be_read_by_ants(self, small_t1w_path, small_ref_path, tmp_path):
        out_file = str(tmp_path / "ants_readable.nii.gz")
        avp.preprocess_img(
            small_t1w_path,
            out_file,
            modality="t1",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        img = ants.image_read(out_file)
        assert img.dimension == 3

    def test_output_spacing_matches_ref(self, small_t1w_path, small_ref_path, tmp_path):
        out_file = str(tmp_path / "resampled.nii.gz")
        avp.preprocess_img(
            small_t1w_path,
            out_file,
            modality="t1",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        ref_img = ants.image_read(small_ref_path)
        out_img = ants.image_read(out_file)
        for i in range(3):
            assert abs(ref_img.spacing[i] - out_img.spacing[i]) < 1e-4, (
                f"Spacing mismatch at dim {i}: "
                f"ref={ref_img.spacing[i]}, out={out_img.spacing[i]}"
            )

    def test_output_origin_matches_ref(self, small_t1w_path, small_ref_path, tmp_path):
        out_file = str(tmp_path / "coregistered.nii.gz")
        avp.preprocess_img(
            small_t1w_path,
            out_file,
            modality="t1",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        ref_origin = ants.image_read(small_ref_path).origin
        out_origin = ants.image_read(out_file).origin
        for i in range(3):
            assert abs(ref_origin[i] - out_origin[i]) < 1e-4, (
                f"Origin mismatch at dim {i}: "
                f"ref={ref_origin[i]}, out={out_origin[i]}"
            )

    def test_identity_registration(self, small_ref_path, tmp_path):
        out_file = str(tmp_path / "identity.nii.gz")
        avp.preprocess_img(
            small_ref_path,
            out_file,
            modality="t1",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        assert os.path.exists(out_file)
        ref_data = ants.image_read(small_ref_path).numpy()
        out_data = ants.image_read(out_file).numpy()
        assert out_data.shape == ref_data.shape
        corr = np.corrcoef(ref_data.ravel(), out_data.ravel())[0, 1]
        assert corr > 0.99, f"Self-registration should preserve content, corr={corr}"

    def test_missing_input_raises(self, tmp_path):
        out_file = str(tmp_path / "nonexistent.nii.gz")
        with pytest.raises(Exception):
            avp.preprocess_img(
                "/nonexistent/path/input.nii.gz",
                out_file,
                modality="t1",
                do_skull_strip=False,
            )

    def test_invalid_modality_raises(self, small_t1w_path, tmp_path):
        out_file = str(tmp_path / "bad_modality.nii.gz")
        with pytest.raises(Exception):
            avp.preprocess_img(
                small_t1w_path,
                out_file,
                modality="invalid",
                do_skull_strip=False,
            )

    def test_empty_modality_raises(self, small_t1w_path, tmp_path):
        out_file = str(tmp_path / "empty_modality.nii.gz")
        with pytest.raises(Exception):
            avp.preprocess_img(
                small_t1w_path,
                out_file,
                modality="",
                do_skull_strip=False,
            )

    def test_nonexistent_outdir_raises(self, small_t1w_path):
        bad_out = "/nonexistent_directory/output.nii.gz"
        with pytest.raises(Exception):
            avp.preprocess_img(
                small_t1w_path,
                bad_out,
                modality="t1",
                do_skull_strip=False,
            )
