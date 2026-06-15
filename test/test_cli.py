import os

import nibabel as nib
import numpy as np
import pandas as pd


def _make_nifti(data, spacing, path):
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms(spacing)
    nib.save(img, str(path))


def _vessel_seg(shape=(10, 10, 10)):
    data = np.zeros(shape, dtype=np.uint8)
    data[5, 5, 2:8] = 1
    return data


SPACING = (1.0, 1.0, 1.0)


class TestA2vFeaturesCli:
    def test_help(self, script_runner):
        ret = script_runner.run(["a2v-features", "--help"])
        assert ret.success
        assert "--input_dir" in ret.stdout
        assert "--output_path" in ret.stdout
        assert "--no_ray" in ret.stdout

    def test_run_with_real_files(self, script_runner, tmp_path):
        _make_nifti(_vessel_seg(), SPACING, tmp_path / "sub-01_seg.nii.gz")
        _make_nifti(_vessel_seg(), SPACING, tmp_path / "sub-02_seg.nii.gz")
        output_path = tmp_path / "output.csv"
        ret = script_runner.run(
            [
                "a2v-features",
                "--input_dir",
                str(tmp_path),
                "--output_path",
                str(output_path),
                "--no_ray",
            ]
        )
        assert ret.success
        assert os.path.exists(output_path)
        df = pd.read_csv(str(output_path))
        assert len(df) == 2
        assert set(df["sub_id"]) == {"sub-01", "sub-02"}

    def test_bad_dir_fails(self, script_runner, tmp_path):
        output_path = tmp_path / "output.csv"
        ret = script_runner.run(
            [
                "a2v-features",
                "--input_dir",
                str(tmp_path / "nonexistent"),
                "--output_path",
                str(output_path),
            ]
        )
        assert not ret.success


class TestA2vPreprocessCli:
    def test_help(self, script_runner):
        ret = script_runner.run(["a2v-preprocess", "--help"])
        assert ret.success
        assert "--bids_dir" in ret.stdout
        assert "--output_dir" in ret.stdout
        assert "--model" in ret.stdout
        assert "--skull_strip" in ret.stdout
        assert "--no_ray" in ret.stdout

    def test_bad_dir_fails(self, script_runner, tmp_path):
        ret = script_runner.run(
            [
                "a2v-preprocess",
                "--bids_dir",
                str(tmp_path / "nonexistent"),
                "--output_dir",
                str(tmp_path / "output"),
            ]
        )
        assert not ret.success
