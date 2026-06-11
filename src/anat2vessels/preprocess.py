import os
import os.path as op
import argparse
import ants
import ray
try:
    import antspynet
except ImportError:
    antspynet = None
try:
    from bids import BIDSLayout
except ImportError:
    BIDSLayout = None

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


def preprocess_bids(
    bids_dir, output_dir, model="t1t2", skull_strip=False, use_ray=True
):
    if BIDSLayout is None:
        raise ImportError("pybids is required. Install with: pip install pybids")

    layout = BIDSLayout(bids_dir)
    os.makedirs(output_dir, exist_ok=True)

    subjects = layout.get_subjects()
    if not subjects:
        raise ValueError(f"No subjects found in {bids_dir}")

    n_threads = os.cpu_count() - 1
    if use_ray and n_threads > 1:
        ray.init(num_cpus=n_threads, ignore_reinit_error=True)
        remote_func = ray.remote(_preprocess_subject)
        futures = []
        for sub in subjects:
            futures.append(
                remote_func.remote(layout, sub, model, skull_strip, output_dir)
            )
        ray.get(futures)
    else:
        for sub in subjects:
            _preprocess_subject(layout, sub, model, skull_strip, output_dir)


def _preprocess_subject(layout, subject, model, skull_strip, output_dir):
    if model in ("t1", "t1t2"):
        t1_file = layout.get(
            subject=subject, suffix="T1w", extension=".nii.gz", return_type="file"
        )
        if not t1_file:
            t1_file = layout.get(
                subject=subject, suffix="T1w", extension=".nii", return_type="file"
            )
        if t1_file:
            out = op.join(output_dir, f"{subject}_0000.nii.gz")
            if not op.exists(out):
                preprocess_img(t1_file[0], out, modality="t1", do_skull_strip=skull_strip)

    if model == "t2":
        t2_file = layout.get(
            subject=subject, suffix="T2w", extension=".nii.gz", return_type="file"
        )
        if not t2_file:
            t2_file = layout.get(
                subject=subject, suffix="T2w", extension=".nii", return_type="file"
            )
        if t2_file:
            out = op.join(output_dir, f"{subject}_0000.nii.gz")
            if not op.exists(out):
                preprocess_img(t2_file[0], out, modality="t2", do_skull_strip=skull_strip)

    if model == "t1t2":
        t2_file = layout.get(
            subject=subject, suffix="T2w", extension=".nii.gz", return_type="file"
        )
        if not t2_file:
            t2_file = layout.get(
                subject=subject, suffix="T2w", extension=".nii", return_type="file"
            )
        if t2_file:
            out = op.join(output_dir, f"{subject}_0001.nii.gz")
            if not op.exists(out):
                preprocess_img(t2_file[0], out, modality="t2", do_skull_strip=skull_strip)


def run():
    parser = argparse.ArgumentParser(
        description="Preprocess anatomical MRI for vessel segmentation"
    )
    parser.add_argument("--bids_dir", required=True, help="BIDS dataset directory")
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for preprocessed images"
    )
    parser.add_argument(
        "--model",
        choices=["t1", "t2", "t1t2"],
        default="t1t2",
        help="Target model (determines which modalities to process)",
    )
    parser.add_argument(
        "--skull_strip", action="store_true", help="Apply skull stripping"
    )
    parser.add_argument(
        "--no_ray", action="store_true", help="Disable parallel processing"
    )
    args = parser.parse_args()
    preprocess_bids(
        args.bids_dir,
        args.output_dir,
        args.model,
        args.skull_strip,
        use_ray=not args.no_ray,
    )
