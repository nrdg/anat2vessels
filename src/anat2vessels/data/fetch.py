import pooch

_BASE_URL = "https://huggingface.co/huggingbrain/AnatomicalVesselSeg/resolve/main/"

REGISTRY = pooch.create(
    path=pooch.os_cache("anat2vessels"),
    base_url=_BASE_URL,
    registry={
        "ref.nii.gz": "sha256:a73a27eb80db1bdd36e33adb843da21e5df145402c773695193c56ade1fa30b0",
        "sub-01_ses-forrestgump_T1w.nii.gz": "sha256:7c702aee386767418cc1a914fb8957c75ff2844e1d72de2c2d4b85cbac88f0cf",
        "sub-01_ses-forrestgump_T2w.nii.gz": "sha256:652248cb57ccbca27778660960b005e9206a9281811ea8480cae6fb3eb2652d2",
    },
    retry_if_failed=5,
)


def fetch_ref_img():
    return REGISTRY.fetch("ref.nii.gz")


def fetch_test_data():
    return {
        "t1w": REGISTRY.fetch("sub-01_ses-forrestgump_T1w.nii.gz"),
        "t2w": REGISTRY.fetch("sub-01_ses-forrestgump_T2w.nii.gz"),
    }
