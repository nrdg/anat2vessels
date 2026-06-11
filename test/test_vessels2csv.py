import numpy as np
import pandas as pd

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
