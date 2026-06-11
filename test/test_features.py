import numpy as np

import nibabel as nib
import pytest

from anat2vessels.features import (
    _calc_full_path_from_points,
    _calc_shortest_path_from_points,
    _calc_tortuosities_also_lengths,
    _extract_radius,
    _extract_skeleton,
    _get_bifurcation_endpoint_arrays,
    _get_branch_array_by_label,
    _get_labeled_branches,
    _get_num_neighbors,
    _get_points_in_order,
    _get_skel_seg_spacing,
    compute_features,
    extract_features,
)


class TestGetNumNeighbors:
    def test_all_zeros(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        result = _get_num_neighbors(arr)
        assert result.shape == (5, 5, 5)
        assert result.max() == 0
        assert result.min() == 0

    def test_single_voxel(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        result = _get_num_neighbors(arr)
        assert result[2, 2, 2] == 0

    def test_adjacent_pair(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        arr[2, 2, 3] = 1
        result = _get_num_neighbors(arr)
        assert result[2, 2, 2] == 1
        assert result[2, 2, 3] == 1

    def test_line_of_three(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 1:4] = 1
        result = _get_num_neighbors(arr)
        assert result[2, 2, 1] == 1
        assert result[2, 2, 2] == 2
        assert result[2, 2, 3] == 1

    def test_3x3x3_block(self):
        arr = np.zeros((7, 7, 7), dtype=np.uint8)
        arr[2:5, 2:5, 2:5] = 1
        result = _get_num_neighbors(arr)
        assert result[2, 2, 2] == 7
        assert result[3, 3, 3] == 26
        assert result[2, 3, 3] == 17

    def test_preserves_input(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        original = arr.copy()
        _get_num_neighbors(arr)
        assert np.array_equal(arr, original)


class TestGetBifurcationEndpointArrays:
    def test_straight_line(self, straight_line_skeleton):
        bif, end = _get_bifurcation_endpoint_arrays(straight_line_skeleton)
        assert bif.sum() == 0
        assert end.sum() == 2

    def test_cross_three_axes(self):
        arr = np.zeros((7, 7, 7), dtype=np.uint8)
        arr[4, 4, 2:7] = 1
        arr[4, 2:7, 4] = 1
        arr[2:7, 4, 4] = 1
        bif, end = _get_bifurcation_endpoint_arrays(arr)
        assert bif.sum() == 7
        assert end.sum() == 6

    def test_t_shape(self, t_shape_skeleton):
        bif, end = _get_bifurcation_endpoint_arrays(t_shape_skeleton)
        assert bif.sum() == 4
        assert end.sum() == 3

    def test_single_voxel(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        bif, end = _get_bifurcation_endpoint_arrays(arr)
        assert bif.sum() == 0
        assert end.sum() == 0

    def test_closed_loop_four_voxels(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        arr[2, 3, 2] = 1
        arr[3, 3, 2] = 1
        arr[3, 2, 2] = 1
        bif, end = _get_bifurcation_endpoint_arrays(arr)
        assert bif.sum() == 4
        assert end.sum() == 0

    def test_all_zeros(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        bif, end = _get_bifurcation_endpoint_arrays(arr)
        assert bif.sum() == 0
        assert end.sum() == 0

    def test_outputs_are_binary(self, straight_line_skeleton):
        bif, end = _get_bifurcation_endpoint_arrays(straight_line_skeleton)
        assert bif.dtype == np.uint8 or bif.dtype == bool
        assert end.dtype == np.uint8 or end.dtype == bool
        assert set(np.unique(bif)).issubset({0, 1})
        assert set(np.unique(end)).issubset({0, 1})


class TestGetLabeledBranches:
    def test_single_line(self, straight_line_skeleton):
        labeled, names = _get_labeled_branches(straight_line_skeleton)
        assert set(np.unique(labeled)) == {0, 1}
        assert names == [1]

    def test_two_disconnected_lines(self):
        arr = np.zeros((7, 7, 7), dtype=np.uint8)
        arr[2, 2, 1:4] = 1
        arr[5, 5, 1:4] = 1
        labeled, names = _get_labeled_branches(arr)
        assert len(names) == 2
        assert np.count_nonzero(labeled == names[0]) >= 2
        assert np.count_nonzero(labeled == names[1]) >= 2

    def test_single_voxel_removed(self):
        arr = np.zeros((5, 5, 5), dtype=np.uint8)
        arr[2, 2, 2] = 1
        labeled, names = _get_labeled_branches(arr)
        assert len(names) == 0
        assert labeled.sum() == 0

    def test_output_has_same_shape(self, straight_line_skeleton):
        labeled, _ = _get_labeled_branches(straight_line_skeleton)
        assert labeled.shape == straight_line_skeleton.shape

    def test_bifurcation_voxels_excluded(self, t_shape_skeleton):
        labeled, _ = _get_labeled_branches(t_shape_skeleton)
        bif, _ = _get_bifurcation_endpoint_arrays(t_shape_skeleton)
        for z in range(t_shape_skeleton.shape[0]):
            for y in range(t_shape_skeleton.shape[1]):
                for x in range(t_shape_skeleton.shape[2]):
                    if bif[z, y, x]:
                        assert labeled[z, y, x] == 0


class TestExtractSkeleton:
    def test_output_is_uint8(self, straight_line_skeleton):
        seg = straight_line_skeleton.astype(bool)
        skel = _extract_skeleton(seg)
        assert skel.dtype == np.uint8

    def test_thickens_line(self):
        seg = np.zeros((7, 7, 7), dtype=bool)
        seg[3, 3, 2:5] = True
        skel = _extract_skeleton(seg)
        assert skel.sum() <= seg.sum()

    def test_empty_segmentation(self):
        seg = np.zeros((5, 5, 5), dtype=bool)
        skel = _extract_skeleton(seg)
        assert skel.sum() == 0

    def test_single_voxel(self):
        seg = np.zeros((5, 5, 5), dtype=bool)
        seg[2, 2, 2] = True
        skel = _extract_skeleton(seg)
        assert skel[2, 2, 2] == 1
        assert skel.sum() == 1


class TestExtractRadius:
    def test_radius_positive(self):
        seg = np.zeros((7, 7, 7), dtype=bool)
        seg[3:5, 3:5, 2:5] = 1
        skel = np.zeros((7, 7, 7), dtype=np.uint8)
        skel[3, 3, 2:5] = 1
        radii = _extract_radius(seg, skel, (1.0, 1.0, 1.0))
        assert len(radii) > 0
        assert (radii > 0).all()

    def test_uniform_line_radius(self):
        seg = np.zeros((7, 7, 7), dtype=bool)
        seg[3:5, 3:5, 2:5] = 1
        skel = np.zeros((7, 7, 7), dtype=np.uint8)
        skel[3, 3, 2:5] = 1
        radii = _extract_radius(seg, skel, (1.0, 1.0, 1.0))
        expected = np.full(3, 1.0)
        assert np.allclose(radii, expected)

    def test_non_uniform_spacing_changes_radius(self):
        seg = np.zeros((7, 7, 7), dtype=bool)
        seg[3:5, 3:7, 2:4] = 1
        skel = np.zeros((7, 7, 7), dtype=np.uint8)
        skel[3, 3, 2:4] = 1
        r_uniform = _extract_radius(seg, skel, (1.0, 1.0, 1.0))
        r_aniso = _extract_radius(seg, skel, (1.0, 0.5, 1.0))
        assert not np.allclose(r_uniform, r_aniso)

    def test_empty_skeleton(self):
        seg = np.zeros((7, 7, 7), dtype=bool)
        seg[3:5, 3:5, 2:5] = 1
        skel = np.zeros((7, 7, 7), dtype=np.uint8)
        radii = _extract_radius(seg, skel, (1.0, 1.0, 1.0))
        assert len(radii) == 0


class TestGetPointsInOrder:
    def test_three_voxel_line(self):
        branch = np.zeros((5, 5, 5), dtype=np.int8)
        branch[2, 2, 1:4] = 1
        points = _get_points_in_order(branch, (1.0, 1.0, 1.0))
        assert len(points) == 3

    def test_two_voxel_line(self):
        branch = np.zeros((5, 5, 5), dtype=np.int8)
        branch[2, 2, 1:3] = 1
        points = _get_points_in_order(branch, (1.0, 1.0, 1.0))
        assert len(points) == 2

    def test_first_point_is_endpoint(self):
        branch = np.zeros((5, 5, 5), dtype=np.int8)
        branch[2, 2, 1:4] = 1
        points = _get_points_in_order(branch, (1.0, 1.0, 1.0))
        first = tuple(points[0].astype(int))
        _, endpoints = _get_bifurcation_endpoint_arrays(branch.astype(np.uint8))
        assert endpoints[first] == 1

    def test_last_point_is_endpoint(self):
        branch = np.zeros((5, 5, 5), dtype=np.int8)
        branch[2, 2, 1:4] = 1
        points = _get_points_in_order(branch, (1.0, 1.0, 1.0))
        last = tuple(points[-1].astype(int))
        _, endpoints = _get_bifurcation_endpoint_arrays(branch.astype(np.uint8))
        assert endpoints[last] == 1

    def test_single_voxel_returns_empty(self):
        branch = np.zeros((5, 5, 5), dtype=np.int8)
        branch[2, 2, 2] = 1
        points = _get_points_in_order(branch, (1.0, 1.0, 1.0))
        assert len(points) == 0

    def test_no_endpoints_returns_empty(self):
        branch = np.zeros((5, 5, 5), dtype=np.int8)
        branch[2, 2, 2] = 1
        branch[2, 3, 2] = 1
        branch[3, 3, 2] = 1
        branch[3, 2, 2] = 1
        points = _get_points_in_order(branch, (1.0, 1.0, 1.0))
        assert len(points) == 0

    def test_disconnected_raises(self):
        branch = np.zeros((7, 7, 7), dtype=np.int8)
        branch[2, 2, 1:4] = 1
        branch[5, 5, 1:4] = 1
        with pytest.raises(ValueError, match="Invalid path"):
            _get_points_in_order(branch, (1.0, 1.0, 1.0))

    def test_non_uniform_spacing(self):
        branch = np.zeros((5, 5, 5), dtype=np.int8)
        branch[2, 2, 1:4] = 1
        points = _get_points_in_order(branch, (2.0, 3.0, 1.0))
        expected_first = np.array([4.0, 6.0, 1.0])
        assert np.allclose(points[0], expected_first)


class TestCalcFullPathFromPoints:
    def test_single_point(self):
        points = np.array([[1.0, 2.0, 3.0]])
        assert _calc_full_path_from_points(points) == 1.0

    def test_two_points(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        assert _calc_full_path_from_points(points) == 5.0

    def test_three_collinear(self):
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 5.0]])
        assert _calc_full_path_from_points(points) == 5.0

    def test_three_non_collinear(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        expected = 3.0 + 4.0
        assert _calc_full_path_from_points(points) == expected


class TestCalcShortestPathFromPoints:
    def test_single_point(self):
        points = np.array([[1.0, 2.0, 3.0]])
        assert _calc_shortest_path_from_points(points) == 1.0

    def test_two_points(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        assert _calc_shortest_path_from_points(points) == 5.0

    def test_straight_line_matches_full(self):
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 5.0]])
        full = _calc_full_path_from_points(points)
        shortest = _calc_shortest_path_from_points(points)
        assert full == shortest == 5.0

    def test_shortest_is_less_than_full(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        full = _calc_full_path_from_points(points)
        shortest = _calc_shortest_path_from_points(points)
        assert shortest < full
        assert shortest == 5.0


class TestGetBranchArrayByLabel:
    def test_existing_label(self, straight_line_skeleton):
        labeled, names = _get_labeled_branches(straight_line_skeleton)
        branch = _get_branch_array_by_label(labeled, names[0])
        assert branch.dtype == np.int8
        assert branch.sum() > 0
        assert branch.shape == straight_line_skeleton.shape

    def test_nonexistent_label(self):
        labeled = np.zeros((5, 5, 5), dtype=np.int32)
        branch = _get_branch_array_by_label(labeled, 99)
        assert branch.sum() == 0

    def test_all_zeros(self):
        labeled = np.zeros((5, 5, 5), dtype=np.int32)
        branch = _get_branch_array_by_label(labeled, 1)
        assert branch.sum() == 0


class TestCalcTortuositiesAlsoLengths:
    def test_straight_branch(self):
        skel = np.zeros((7, 7, 7), dtype=np.uint8)
        skel[3, 3, 1:6] = 1
        labeled, names = _get_labeled_branches(skel)
        branches = _calc_tortuosities_also_lengths(labeled, names, (1.0, 1.0, 1.0))
        assert len(branches) == 1
        assert branches[0]["tortuosity"] == pytest.approx(1.0)

    def test_tortuous_branch(self):
        skel = np.zeros((9, 9, 9), dtype=np.uint8)
        skel[3, 3, 1:5] = 1
        skel[3, 5, 5] = 1
        labeled, names = _get_labeled_branches(skel)
        branches = _calc_tortuosities_also_lengths(labeled, names, (1.0, 1.0, 1.0))
        assert len(branches) >= 1
        for b in branches:
            assert b["tortuosity"] >= 1.0

    def test_multiple_branches(self):
        skel = np.zeros((9, 9, 9), dtype=np.uint8)
        skel[2, 2, 1:4] = 1
        skel[6, 6, 1:4] = 1
        labeled, names = _get_labeled_branches(skel)
        branches = _calc_tortuosities_also_lengths(labeled, names, (1.0, 1.0, 1.0))
        assert len(branches) == 2

    def test_empty_label_list(self):
        labeled = np.zeros((5, 5, 5), dtype=np.int32)
        branches = _calc_tortuosities_also_lengths(labeled, [], (1.0, 1.0, 1.0))
        assert len(branches) == 0

    def test_single_voxel_skipped(self):
        labeled = np.zeros((5, 5, 5), dtype=np.int32)
        labeled[2, 2, 2] = 1
        branches = _calc_tortuosities_also_lengths(labeled, [1], (1.0, 1.0, 1.0))
        assert len(branches) == 0


class TestGetSkelSegSpacing:
    def _make_nifti(self, data, spacing, tmp_path):
        path = str(tmp_path / "test.nii.gz")
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms(spacing)
        nib.save(img, path)
        return path

    def test_returns_correct_spacing(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[5, 5, 2:8] = 1
        nii_path = self._make_nifti(data, (1.5, 1.0, 2.0), tmp_path)
        skeleton, seg, spacing = _get_skel_seg_spacing(nii_path)
        assert spacing == (1.5, 1.0, 2.0)

    def test_segmentation_is_bool(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[5, 5, 2:8] = 1
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        _, seg, _ = _get_skel_seg_spacing(nii_path)
        assert seg.dtype == bool

    def test_skeleton_is_binary(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[5, 5, 2:8] = 1
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        skeleton, _, _ = _get_skel_seg_spacing(nii_path)
        assert skeleton.dtype == np.uint8
        assert set(np.unique(skeleton)).issubset({0, 1})

    def test_skeleton_is_thinner_or_equal(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[5, 5, 2:8] = 1
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        skeleton, seg, _ = _get_skel_seg_spacing(nii_path)
        assert skeleton.sum() <= seg.sum()


class TestExtractFeatures:
    def _make_nifti(self, data, spacing, tmp_path):
        path = str(tmp_path / "test.nii.gz")
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms(spacing)
        nib.save(img, path)
        return path

    def test_returns_dict_with_expected_keys(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[5, 5, 2:8] = 1
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        result = extract_features(nii_path)
        expected_keys = {
            "branch_list",
            "bifurcations",
            "endpoints",
            "radius_list",
            "total_volume",
            "num_branches",
        }
        assert set(result.keys()) == expected_keys

    def test_num_branches(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[5, 5, 2:8] = 1
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        result = extract_features(nii_path)
        assert result["num_branches"] >= 0
        assert isinstance(result["num_branches"], int)

    def test_total_volume_positive(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[5, 5, 2:8] = 1
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        result = extract_features(nii_path)
        assert result["total_volume"] > 0.0

    def test_bifurcations_is_array(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[5, 5, 2:8] = 1
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        result = extract_features(nii_path)
        assert isinstance(result["bifurcations"], np.ndarray)
        assert isinstance(result["endpoints"], np.ndarray)

    def test_radius_list_contents(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[5, 5, 2:8] = 1
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        result = extract_features(nii_path)
        assert isinstance(result["radius_list"], list)
        assert all(r > 0 for r in result["radius_list"])

    def test_branch_list_is_list(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[5, 5, 2:8] = 1
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        result = extract_features(nii_path)
        assert isinstance(result["branch_list"], list)


class TestExtractFeaturesEmpty:
    def _make_nifti(self, data, spacing, tmp_path):
        path = str(tmp_path / "test.nii.gz")
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms(spacing)
        nib.save(img, path)
        return path

    def test_all_zero_volume(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        result = extract_features(nii_path)
        assert result["total_volume"] == 0.0

    def test_empty_radius_list(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        result = extract_features(nii_path)
        assert result["radius_list"] == []

    def test_zero_branches(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        result = extract_features(nii_path)
        assert result["num_branches"] == 0
        assert result["branch_list"] == []

    def test_bifurcations_all_zero(self, tmp_path):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        nii_path = self._make_nifti(data, (1.0, 1.0, 1.0), tmp_path)
        result = extract_features(nii_path)
        assert result["bifurcations"].sum() == 0


class TestComputeFeatures:
    def test_with_numpy_arrays(self):
        seg = np.zeros((10, 10, 10), dtype=bool)
        seg[5, 5, 2:8] = True
        result = compute_features(seg, (1.0, 1.0, 1.0))
        expected_keys = {
            "branch_list",
            "bifurcations",
            "endpoints",
            "radius_list",
            "total_volume",
            "num_branches",
        }
        assert set(result.keys()) == expected_keys

    def test_total_matches_extract_features(self, tmp_path):
        seg = np.zeros((10, 10, 10), dtype=bool)
        seg[5, 5, 2:8] = True
        array_result = compute_features(seg, (1.0, 1.0, 1.0))

        path = str(tmp_path / "test.nii.gz")
        img = nib.Nifti1Image(seg.astype(np.uint8), np.eye(4))
        img.header.set_zooms((1.0, 1.0, 1.0))
        nib.save(img, path)
        nifti_result = extract_features(path)

        assert array_result["total_volume"] == nifti_result["total_volume"]
        assert array_result["num_branches"] == nifti_result["num_branches"]

    def test_empty_segmentation(self):
        seg = np.zeros((10, 10, 10), dtype=bool)
        result = compute_features(seg, (1.0, 1.0, 1.0))
        assert result["total_volume"] == 0.0
        assert result["num_branches"] == 0
        assert result["radius_list"] == []
        assert result["branch_list"] == []

    def test_non_uniform_spacing(self):
        seg = np.zeros((10, 10, 10), dtype=bool)
        seg[5, 5, 2:8] = True
        result = compute_features(seg, (2.0, 1.0, 3.0))
        assert result["total_volume"] > 0
        assert result["num_branches"] >= 0
