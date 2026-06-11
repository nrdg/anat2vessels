import ants

try:
    import ray
except ImportError:
    ray = None
try:
    import antspynet
except ImportError:
    antspynet = None

_REF_IMG_PATH = None


def _get_ref_path():
    global _REF_IMG_PATH
    if _REF_IMG_PATH is None:
        from anat2vessels.data.fetch import fetch_ref_img

        _REF_IMG_PATH = fetch_ref_img()
    return _REF_IMG_PATH


def __getattr__(name):
    if name == "REF_IMG_PATH":
        return _get_ref_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def preprocess_img(
    in_file, out_file, modality="t1", do_skull_strip=False, ref_path=None
):
    """
    Skull stripping (optional), registration to ref, resampling, cropping.

    Parameters
    ----------
    in_file : str
        Full path to the input file.

    out_file : str
        Full path to the output file.

    modality : str, optional. One of ['t1', 't2']

    do_skull_strip : bool, optional
        Whether to strip the skull from the brain. Default: False.

    ref_path : str, optional
        Path to reference image. Default: REF_IMG_PATH.
    """
    img = ants.image_read(in_file)
    if ref_path is None:
        ref_path = _get_ref_path()
    ref_img = ants.image_read(ref_path)

    if do_skull_strip:
        img = skull_strip(img, modality=modality)

    # Uses rigid so size doesn't change:
    img_reg = ants.registration(fixed=ref_img, moving=img, type_of_transform="Rigid")
    img = img_reg["warpedmovout"]
    img = ants.resample_image_to_target(img, ref_img, interp_type="linear")
    img = ants.crop_image(img, ref_img)
    ants.image_write(img, out_file)
