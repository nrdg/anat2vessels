import numpy as np
import pandas as pd
import pytest

from anat2vessels.vessels2csv import out_list_to_df


class TestOutListToDf:
    def test_returns_dataframe(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        assert isinstance(result, pd.DataFrame)

    def test_single_subject_has_one_row(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        assert len(result) == 1

    def test_expected_columns_present(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        expected = {
            "sub_id",
            "num_branches",
            "total_volume",
            "bifurcations",
            "endpoints",
            "radius_list",
            "mean_radius",
            "max_radius",
            "min_radius",
            "mean_tortuosity",
            "max_tortuosity",
            "min_tortuosity",
            "tortuosity_list",
            "branch_list",
            "total_branch_length",
            "mean_branch_length",
            "max_branch_length",
        }
        assert set(result.columns) == expected

    def test_multiple_subjects_have_correct_rows(self, multi_subject_features):
        result = out_list_to_df(multi_subject_features)
        assert len(result) == 2
        assert list(result["sub_id"]) == ["sub-01", "sub-02"]

    def test_sub_id_is_string(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        assert isinstance(result["sub_id"].iloc[0], str)

    def test_num_branches_is_int(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        assert result["num_branches"].iloc[0] == 2
        assert isinstance(result["num_branches"].iloc[0], (int, np.integer))

    def test_total_volume_is_float(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        assert result["total_volume"].iloc[0] == 100.0
        assert isinstance(result["total_volume"].iloc[0], float)

    def test_radius_list_is_stored(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        assert result["radius_list"].iloc[0] == [1.5, 2.0, 2.5]

    def test_branch_list_is_stored(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        branch_list = result["branch_list"].iloc[0]
        assert isinstance(branch_list, list)
        assert len(branch_list) == 2
        for entry in branch_list:
            assert isinstance(entry, dict)
            assert "full_path" in entry
            assert "straight_path" in entry
            assert "tortuosity" in entry

    def test_tortuosity_list_is_stored(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        tort_list = result["tortuosity_list"].iloc[0]
        assert isinstance(tort_list, list)
        assert tort_list == [1.25, 1.25]


class TestOutListToDfEdgeCases:
    def test_empty_input_list_returns_empty_dataframe(self):
        result = out_list_to_df([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_empty_radius_list_defaults_zero(self, subject_empty_radius):
        result = out_list_to_df([subject_empty_radius])
        assert result["mean_radius"].iloc[0] == 0.0
        assert result["max_radius"].iloc[0] == 0.0
        assert result["min_radius"].iloc[0] == 0.0

    def test_empty_branch_list_skipped(self, subject_empty_branches):
        result = out_list_to_df([subject_empty_branches])
        assert len(result) == 0

    def test_malformed_subject_skipped(self, subject_malformed_features):
        result = out_list_to_df([subject_malformed_features])
        assert len(result) == 0

    def test_mixed_valid_and_malformed(
        self, single_subject_features, subject_malformed_features
    ):
        result = out_list_to_df(
            [
                single_subject_features,
                subject_malformed_features,
                single_subject_features,
            ]
        )
        assert len(result) == 2
        assert list(result["sub_id"]) == ["sub-01", "sub-01"]

    def test_empty_branch_list_alongside_valid(
        self, single_subject_features, subject_empty_branches
    ):
        result = out_list_to_df(
            [
                single_subject_features,
                subject_empty_branches,
            ]
        )
        assert len(result) == 1
        assert result["sub_id"].iloc[0] == "sub-01"


class TestOutListToDfStats:
    def test_mean_radius(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        expected = (1.5 + 2.0 + 2.5) / 3
        assert result["mean_radius"].iloc[0] == pytest.approx(expected)

    def test_max_radius(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        assert result["max_radius"].iloc[0] == 2.5

    def test_min_radius(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        assert result["min_radius"].iloc[0] == 1.5

    def test_mean_tortuosity(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        assert result["mean_tortuosity"].iloc[0] == 1.25

    def test_tortuosity_range(self):
        features = {
            "sub_id": "test",
            "num_branches": 2,
            "total_volume": 10.0,
            "bifurcations": np.array([0, 0]),
            "endpoints": np.array([1, 1]),
            "radius_list": [1.0],
            "branch_list": [
                {"full_path": 1.0, "straight_path": 1.0, "tortuosity": 1.0},
                {"full_path": 2.0, "straight_path": 1.0, "tortuosity": 2.0},
            ],
        }
        result = out_list_to_df([features])
        assert result["max_tortuosity"].iloc[0] == 2.0
        assert result["min_tortuosity"].iloc[0] == 1.0
        assert result["mean_tortuosity"].iloc[0] == 1.5

    def test_total_branch_length(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        expected = 10.0 + 15.0
        assert result["total_branch_length"].iloc[0] == pytest.approx(expected)

    def test_mean_branch_length(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        expected = (10.0 + 15.0) / 2
        assert result["mean_branch_length"].iloc[0] == pytest.approx(expected)

    def test_max_branch_length(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        assert result["max_branch_length"].iloc[0] == 15.0

    def test_bifurcations_sum(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        expected = float(single_subject_features["bifurcations"].sum())
        assert result["bifurcations"].iloc[0] == expected

    def test_endpoints_sum(self, single_subject_features):
        result = out_list_to_df([single_subject_features])
        expected = float(single_subject_features["endpoints"].sum())
        assert result["endpoints"].iloc[0] == expected

    def test_multiple_subjects_stats(self, multi_subject_features):
        result = out_list_to_df(multi_subject_features)
        assert result["mean_radius"].iloc[0] == (1.0 + 2.0) / 2
        assert result["max_radius"].iloc[0] == 2.0
        assert result["min_radius"].iloc[0] == 1.0
        assert result["mean_radius"].iloc[1] == 3.0
