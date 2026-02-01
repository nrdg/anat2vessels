from anat2vessels import features as avf


def test_extract_features():
    nifti_path = "tests/data/anat2vessels/test_skel_seg.nii.gz"
    features = avf.extract_features(nifti_path)

    assert "branch_list" in features
    assert "bifurcations" in features
    assert "endpoints" in features

    assert len(features["branch_list"]) == 3
    assert features["bifurcations"].sum() == 2
    assert features["endpoints"].sum() == 4
