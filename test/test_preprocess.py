import json
import os

import ants
import nibabel as nib
import numpy as np
import pytest

from anat2vessels import preprocess as avp
from anat2vessels.data.fetch import fetch_test_data


class TestFetchTestData:
    def test_download_structure(self, t1w_path, t2w_path):
        data = fetch_test_data()
        assert isinstance(data, dict)
        assert data["t1w"] == t1w_path
        assert data["t2w"] == t2w_path

    def test_t1w_is_nifti(self, t1w_path):
        assert t1w_path.endswith(".nii.gz")

    def test_t2w_is_nifti(self, t2w_path):
        assert t2w_path.endswith(".nii.gz")

    def test_t1w_valid_nifti(self, t1w_path):
        img = nib.load(t1w_path)
        assert len(img.shape) == 3
        assert img.shape == (274, 384, 384)

    def test_t2w_valid_nifti(self, t2w_path):
        img = nib.load(t2w_path)
        assert len(img.shape) == 3
        assert img.shape == (274, 384, 384)

    def test_t1w_readable_by_ants(self, t1w_path):
        img = ants.image_read(t1w_path)
        assert img.dimension == 3

    def test_t2w_readable_by_ants(self, t2w_path):
        img = ants.image_read(t2w_path)
        assert img.dimension == 3

    def test_t1w_data_range(self, t1w_path):
        fdata = nib.load(t1w_path).get_fdata()
        assert fdata.min() >= 0
        assert fdata.max() >= 1

    def test_t2w_data_range(self, t2w_path):
        fdata = nib.load(t2w_path).get_fdata()
        assert fdata.min() >= 0
        assert fdata.max() >= 1

    def test_caching(self, t1w_path, t2w_path):
        data = fetch_test_data()
        assert data["t1w"] == t1w_path
        assert data["t2w"] == t2w_path


class TestRefImgPath:
    def test_ref_img_path_exists(self):
        assert os.path.exists(avp.REF_IMG_PATH)

    def test_ref_img_path_is_nifti(self):
        assert avp.REF_IMG_PATH.endswith(".nii.gz")

    def test_ref_img_is_valid_nifti(self):
        img = nib.load(avp.REF_IMG_PATH)
        assert len(img.shape) == 3

    def test_ref_img_readable_by_ants(self):
        img = ants.image_read(avp.REF_IMG_PATH)
        assert img.dimension == 3


class TestSkullStrip:
    @pytest.fixture(scope="class", autouse=True)
    def _run_skull_strip(self, request, ants_t1w, ants_t2w):
        request.cls._t1w_result = avp.skull_strip(ants_t1w, modality="t1")
        request.cls._t2w_result = avp.skull_strip(ants_t2w, modality="t2")

    def test_t1_output_type(self):
        assert hasattr(self._t1w_result, "shape")
        assert self._t1w_result.dimension == 3

    def test_t1_changes_image(self, ants_t1w):
        orig_data = ants_t1w.numpy()
        result_data = self._t1w_result.numpy()
        assert result_data.shape == orig_data.shape
        assert np.any(result_data != orig_data)

    def test_t1_reduces_intensity_outside_brain(self, ants_t1w):
        orig_data = ants_t1w.numpy()
        result_data = self._t1w_result.numpy()
        assert np.sum(result_data == 0) > np.sum(orig_data == 0)

    def test_t1_mask_has_brain_voxels(self):
        non_zero = np.count_nonzero(self._t1w_result.numpy())
        assert non_zero / self._t1w_result.numpy().size > 0.1

    def test_t1_preserves_origin(self, ants_t1w):
        for o, r in zip(ants_t1w.origin, self._t1w_result.origin):
            assert abs(o - r) < 1e-6

    def test_t1_preserves_spacing(self, ants_t1w):
        for o, r in zip(ants_t1w.spacing, self._t1w_result.spacing):
            assert abs(o - r) < 1e-6

    def test_t1_output_dtype(self):
        assert self._t1w_result.numpy().dtype in (np.float64, np.float32)

    def test_t2_output_type(self):
        assert hasattr(self._t2w_result, "shape")
        assert self._t2w_result.dimension == 3

    def test_t2_changes_image(self, ants_t2w):
        orig_data = ants_t2w.numpy()
        result_data = self._t2w_result.numpy()
        assert result_data.shape == orig_data.shape
        assert np.any(result_data != orig_data)

    def test_t2_reduces_intensity_outside_brain(self, ants_t2w):
        orig_data = ants_t2w.numpy()
        result_data = self._t2w_result.numpy()
        assert np.sum(result_data == 0) > np.sum(orig_data == 0)

    def test_t2_mask_has_brain_voxels(self):
        non_zero = np.count_nonzero(self._t2w_result.numpy())
        assert non_zero / self._t2w_result.numpy().size > 0.1

    def test_invalid_modality_raises(self, ants_t1w):
        with pytest.raises(Exception):
            avp.skull_strip(ants_t1w, modality="invalid")

    def test_empty_modality_raises(self, ants_t1w):
        with pytest.raises(Exception):
            avp.skull_strip(ants_t1w, modality="")


class TestPreprocessImg:
    @pytest.fixture(scope="class", autouse=True)
    def _run_pipeline(
        self, request, small_t1w_path, small_t2w_path, small_ref_path, tmp_path_factory
    ):
        tmp = str(tmp_path_factory.mktemp("pipe"))

        out_no_ss = os.path.join(tmp, "no_ss.nii.gz")
        avp.preprocess_img(
            small_t1w_path,
            out_no_ss,
            modality="t1",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        request.cls._no_ss_path = out_no_ss

        out_with_ss = os.path.join(tmp, "with_ss.nii.gz")
        avp.preprocess_img(
            small_t1w_path,
            out_with_ss,
            modality="t1",
            do_skull_strip=True,
            ref_path=small_ref_path,
        )
        request.cls._with_ss_path = out_with_ss

        out_t2 = os.path.join(tmp, "t2_no_ss.nii.gz")
        avp.preprocess_img(
            small_t2w_path,
            out_t2,
            modality="t2",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        request.cls._t2_no_ss_path = out_t2

    def test_no_skullstrip_output_created(self):
        assert os.path.exists(self._no_ss_path)

    def test_no_skullstrip_output_valid_nifti(self):
        img = nib.load(self._no_ss_path)
        assert len(img.shape) == 3

    def test_no_skullstrip_output_has_content(self):
        fdata = nib.load(self._no_ss_path).get_fdata()
        assert fdata.size > 0
        assert fdata.max() >= 0

    def test_no_skullstrip_spacing_matches_ref(self, small_ref_path):
        ref_spacing = ants.image_read(small_ref_path).spacing
        out_spacing = ants.image_read(self._no_ss_path).spacing
        for i in range(3):
            assert abs(ref_spacing[i] - out_spacing[i]) < 1e-4

    def test_no_skullstrip_readable_by_ants(self):
        img = ants.image_read(self._no_ss_path)
        assert img.dimension == 3

    def test_no_skullstrip_origin_finite(self):
        origin = ants.image_read(self._no_ss_path).origin
        assert all(np.isfinite(origin))

    def test_with_skullstrip_output_created(self):
        assert os.path.exists(self._with_ss_path)

    def test_with_skullstrip_output_valid_nifti(self):
        img = nib.load(self._with_ss_path)
        assert len(img.shape) == 3

    def test_with_skullstrip_output_has_content(self):
        fdata = nib.load(self._with_ss_path).get_fdata()
        assert fdata.size > 0
        assert fdata.max() >= 0

    def test_with_skullstrip_more_zeros(self):
        no_ss_data = nib.load(self._no_ss_path).get_fdata()
        with_ss_data = nib.load(self._with_ss_path).get_fdata()
        frac_no_ss = np.count_nonzero(no_ss_data == 0) / no_ss_data.size
        frac_with_ss = np.count_nonzero(with_ss_data == 0) / with_ss_data.size
        assert frac_with_ss > frac_no_ss

    def test_t2_modality_output_created(self):
        assert os.path.exists(self._t2_no_ss_path)

    def test_t2_modality_output_valid_nifti(self):
        img = nib.load(self._t2_no_ss_path)
        assert len(img.shape) == 3

    def test_t2_modality_output_has_content(self):
        fdata = nib.load(self._t2_no_ss_path).get_fdata()
        assert fdata.size > 0
        assert fdata.max() >= 0

    def test_self_registration_completes(self, small_ref_path, tmp_path):
        out_file = str(tmp_path / "self_reg.nii.gz")
        avp.preprocess_img(
            small_ref_path,
            out_file,
            modality="t1",
            do_skull_strip=False,
            ref_path=small_ref_path,
        )
        assert os.path.exists(out_file)
        img = ants.image_read(out_file)
        assert img.dimension == 3
        assert np.all(np.isfinite(img.numpy()))

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(Exception):
            avp.preprocess_img(
                "/nonexistent/path/input.nii.gz",
                str(tmp_path / "out.nii.gz"),
                modality="t1",
                do_skull_strip=False,
            )

    def test_nonexistent_outdir_raises(self):
        with pytest.raises(Exception):
            avp.preprocess_img(
                "/tmp/fake.nii.gz",
                "/nonexistent_directory/output.nii.gz",
                modality="t1",
                do_skull_strip=False,
            )


class TestBidsPreprocessing:
    @pytest.fixture(scope="class", autouse=True)
    def _setup_bids(
        self, request, bids_dataset, small_ref_path_for_bids, tmp_path_factory
    ):
        import shutil

        _tmp = str(tmp_path_factory.mktemp("bids"))

        # Copy reference to a stable location
        ref_dir = str(tmp_path_factory.mktemp("ref"))
        ref_path = os.path.join(ref_dir, "ref.nii.gz")
        shutil.copy2(small_ref_path_for_bids, ref_path)
        request.cls._ref_path = ref_path

        request.cls._bids_dir = bids_dataset

        # Run preprocess_bids with the small reference
        output_dir = os.path.join(_tmp, "output")
        original_ref = avp._get_ref_path
        avp._get_ref_path = lambda: ref_path
        try:
            avp.preprocess_bids(
                bids_dataset,
                output_dir,
                model="t1t2",
                skull_strip=False,
                use_ray=False,
            )
        finally:
            avp._get_ref_path = original_ref
        request.cls._output_dir = output_dir

    def test_output_dir_created(self):
        assert os.path.isdir(self._output_dir)

    def test_all_subjects_processed(self):
        for sub_id in ("01", "02"):
            assert os.path.exists(
                os.path.join(self._output_dir, f"{sub_id}_0000.nii.gz")
            )

    def test_t1t2_produces_two_modalities(self):
        assert os.path.exists(os.path.join(self._output_dir, "02_0000.nii.gz"))
        assert os.path.exists(os.path.join(self._output_dir, "02_0001.nii.gz"))

    def test_outputs_are_valid_nifti(self):
        files = os.listdir(self._output_dir)
        assert len(files) > 0
        for fname in files:
            img = nib.load(os.path.join(self._output_dir, fname))
            assert len(img.shape) == 3

    def test_outputs_have_content(self):
        files = os.listdir(self._output_dir)
        assert len(files) > 0
        for fname in files:
            fdata = nib.load(os.path.join(self._output_dir, fname)).get_fdata()
            assert fdata.size > 0

    def test_process_modality_finds_file(self, tmp_path):
        layout = avp.BIDSLayout(self._bids_dir, validate=False)
        out_dir = str(tmp_path / "modality_test")
        os.makedirs(out_dir)
        original_ref = avp._get_ref_path
        avp._get_ref_path = lambda: self._ref_path
        try:
            avp._process_modality(
                layout,
                "01",
                "T1w",
                [".nii.gz", ".nii"],
                out_dir,
                "t1",
                "0000",
                skull_strip=False,
            )
        finally:
            avp._get_ref_path = original_ref
        assert os.path.exists(os.path.join(out_dir, "01_0000.nii.gz"))

    def test_process_modality_no_match(self, tmp_path):
        layout = avp.BIDSLayout(self._bids_dir, validate=False)
        out_dir = str(tmp_path / "no_match")
        os.makedirs(out_dir)
        avp._process_modality(
            layout,
            "01",
            "FLAIR",
            [".nii.gz", ".nii"],
            out_dir,
            "t1",
            "0000",
            skull_strip=False,
        )
        assert not os.path.exists(os.path.join(out_dir, "01_0000.nii.gz"))

    def test_preprocess_subject_t1(self, tmp_path):
        layout = avp.BIDSLayout(self._bids_dir, validate=False)
        out_dir = str(tmp_path / "t1_only")
        os.makedirs(out_dir)
        original_ref = avp._get_ref_path
        avp._get_ref_path = lambda: self._ref_path
        try:
            avp._preprocess_subject(
                layout, "01", "t1", skull_strip=False, output_dir=out_dir
            )
        finally:
            avp._get_ref_path = original_ref
        assert os.path.exists(os.path.join(out_dir, "01_0000.nii.gz"))

    def test_preprocess_subject_t2(self, tmp_path):
        layout = avp.BIDSLayout(self._bids_dir, validate=False)
        out_dir = str(tmp_path / "t2_only")
        os.makedirs(out_dir)
        original_ref = avp._get_ref_path
        avp._get_ref_path = lambda: self._ref_path
        try:
            avp._preprocess_subject(
                layout, "02", "t2", skull_strip=False, output_dir=out_dir
            )
        finally:
            avp._get_ref_path = original_ref
        assert os.path.exists(os.path.join(out_dir, "02_0000.nii.gz"))

    def test_preprocess_subject_t1t2(self, tmp_path):
        layout = avp.BIDSLayout(self._bids_dir, validate=False)
        out_dir = str(tmp_path / "t1t2_full")
        os.makedirs(out_dir)
        original_ref = avp._get_ref_path
        avp._get_ref_path = lambda: self._ref_path
        try:
            avp._preprocess_subject(
                layout, "02", "t1t2", skull_strip=False, output_dir=out_dir
            )
        finally:
            avp._get_ref_path = original_ref
        assert os.path.exists(os.path.join(out_dir, "02_0000.nii.gz"))
        assert os.path.exists(os.path.join(out_dir, "02_0001.nii.gz"))

    def test_preprocess_bids_empty_dir(self, tmp_path):
        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        with open(os.path.join(empty, "dataset_description.json"), "w") as f:
            json.dump({"Name": "Empty", "BIDSVersion": "1.8.0"}, f)
        with pytest.raises(ValueError, match="No subjects found"):
            avp.preprocess_bids(empty, str(tmp_path / "out"), use_ray=False)
