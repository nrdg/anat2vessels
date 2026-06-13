import os
import subprocess

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

_MODEL_HASHES = {
    "t1": "sha256:f83364b0ca95d4955347c0d4856560fb05f8f5eca45f8704d2455973974c6bb0",
    "t2": "sha256:4b79dc0fd427ee1ba80799b7f0c3495872abd7fd49d4da919e47eecead780d51",
    "t1t2": "sha256:403f08d971aca8d389eaa64221709d3632acfa0ccbfa24a5ecb6dc99fdaa1e36",
}

_MODEL_DATASETS = {"t1": "076", "t2": "086", "t1t2": "096"}

_MODEL_POOCH = pooch.create(
    path=pooch.os_cache("anat2vessels") / "models",
    base_url=_BASE_URL,
    registry={
        "t1_model.zip": _MODEL_HASHES["t1"],
        "t2_model.zip": _MODEL_HASHES["t2"],
        "t1t2_model.zip": _MODEL_HASHES["t1t2"],
    },
    retry_if_failed=3,
)


def _model_is_installed(model_name):
    nnunet_results = os.environ.get("nnUNet_results")
    if not nnunet_results or not os.path.isdir(nnunet_results):
        return False
    dataset = _MODEL_DATASETS[model_name]
    return any(
        entry.startswith(f"Dataset{dataset}_") for entry in os.listdir(nnunet_results)
    )


def ensure_model_installed(model_name):
    """Download (if needed) and install an nnUNet model weight package.

    Parameters
    ----------
    model_name : str
        One of ``"t1"``, ``"t2"``, ``"t1t2"``.

    Raises
    ------
    OSError
        If the ``nnUNet_results`` environment variable is not set.
    ValueError
        If ``model_name`` is not one of the known models.
    """
    if model_name not in _MODEL_DATASETS:
        raise ValueError(
            f"Unknown model: {model_name!r}. " f"Choose from {list(_MODEL_DATASETS)}"
        )

    if not os.environ.get("nnUNet_results"):
        raise OSError(
            "The nnUNet_results environment variable must be set so that "
            "nnUNet can find pretrained model weights."
        )

    if _model_is_installed(model_name):
        return

    filenames = {
        "t1": "t1_model.zip",
        "t2": "t2_model.zip",
        "t1t2": "t1t2_model.zip",
    }
    zip_path = _MODEL_POOCH.fetch(filenames[model_name])

    subprocess.run(
        ["nnUNetv2_install_pretrained_model_from_zip", zip_path],
        check=True,
    )


def fetch_ref_img():
    return REGISTRY.fetch("ref.nii.gz")


def fetch_test_data():
    return {
        "t1w": REGISTRY.fetch("sub-01_ses-forrestgump_T1w.nii.gz"),
        "t2w": REGISTRY.fetch("sub-01_ses-forrestgump_T2w.nii.gz"),
    }
