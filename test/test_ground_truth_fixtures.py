"""
Check anat2vessels.features.extract_features against the known-answer NIfTI
fixtures in test/niftis, going through anat2vessels.vessels2csv.out_list_to_df
the same way the production pipeline does. That function still applies real
logic on top of the raw extract_features() arrays (connected-component
counting of bifurcations/endpoints, radius stats, branch-length stats), so
routing through it -- rather than reading extract_features()'s output
directly -- is what's actually exercised in production.
"""
import os

import pytest

from anat2vessels.features import extract_features
from anat2vessels.vessels2csv import out_list_to_df

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "niftis")


def _extract_and_count(path):
    raw = dict(extract_features(path))
    raw["sub_id"] = "test"
    df = out_list_to_df([raw])
    assert not df.empty, f"vessels2csv produced no rows for {path}"
    return df.iloc[0]


# (fixture filename, ground-truth value, how to read the feature, tolerance)
CASES = [
    ("bifurcation_overcount_1.nii.gz", 1, lambda o: int(o["bifurcations"]), 0),
    ("bifurcation_overcount_2.nii.gz", 1, lambda o: int(o["bifurcations"]), 0),
    ("branches_undercount.nii.gz", 4, lambda o: o["num_branches"], 0),
    ("radius_test.nii.gz", 2.0, lambda o: round(o["max_radius"], 2), 0.25),
]
CASE_IDS = [fname for fname, *_ in CASES]


class TestGroundTruthFixtures:
    @pytest.mark.parametrize("fname, truth, reader, tol", CASES, ids=CASE_IDS)
    def test_matches_ground_truth(self, fname, truth, reader, tol):
        path = os.path.join(FIXTURE_DIR, fname)
        assert os.path.exists(path), f"missing fixture: {fname}"
        out = _extract_and_count(path)
        value = reader(out)
        assert abs(value - truth) <= tol
