import numpy as np

from anat2vessels.features import _get_num_neighbors


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
