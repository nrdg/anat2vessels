import pooch

import ants

try:
    import ray
except ImportError:
    ray = None
try:
    import antspynet
except ImportError:
    antspynet = None

REF_IMG_PATH = pooch.retrieve(
    url="https://huggingface.co/huggingbrain/AnatomicalVesselSeg/resolve/main/ref.nii.gz",
    known_hash="sha256:a73a27eb80db1bdd36e33adb843da21e5df145402c773695193c56ade1fa30b0",
    path=pooch.os_cache("anat2vessels"),
    fname="ref.nii.gz",
)


def skull_strip(img, modality="t1"):
    """
    Skull stripping with antspynet.

    Parameters
    ----------
    img : antspy image class instance.
        Target image for skull-stripping
    modality : str, optional. One of ['t1', 't2']
        The image modality.

    Returns
    -------
    An antspy image class instance. Skull stripped result.
    """
    probability_mask = antspynet.brain_extraction(img, modality=modality)
    brain_mask = ants.get_mask(probability_mask, low_thresh=0.5)
    brain_extracted = img * brain_mask
    return brain_extracted


def preprocess_img(in_file, out_file, modality="t1", skull_strip=False):
    """
    Skull stripping (optional), registration to ref, resampling, cropping.

    Parameters
    ----------
    in_file : str
        Full path to the input file.

    out_file : str
        Full path to the output file.

    modality : str, optional. One of ['t1', 't2']

    skull_strip : bool, optional
        Whether to strip the skull from the brain. Default: False.
    """
    img = ants.image_read(in_file)
    ref_img = ants.image_read(REF_IMG_PATH)

    if skull_strip:
        img = skull_strip(img, modality=modality)

    # Uses rigid so size doesn't change:
    img_reg = ants.registration(fixed=ref_img, moving=img, type_of_transform="Rigid")
    img = img_reg["warpedmovout"]
    img = ants.resample_image_to_target(img, ref_img, interp_type="linear")
    img = ants.crop_image(img, ref_img)
    ants.image_write(img, out_file)
