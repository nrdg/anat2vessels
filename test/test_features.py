import numpy as np

from anat2vessels.features import (
    _get_bifurcation_endpoint_arrays,
    _get_labeled_branches,
    _get_num_neighbors,
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
