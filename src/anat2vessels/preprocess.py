import os
import os.path as op
import argparse
import ants
import ray

try:
    import antspynet
except ImportError:
    antspynet = None
from bids import BIDSLayout

_REF_IMG_PATH = None


def _get_ref_path():
    """Return the path to the reference image, fetching and caching it on first call.

    Returns
    -------
    str
        Path to the cached reference NIfTI file.
    """
    global _REF_IMG_PATH
    if _REF_IMG_PATH is None:
        from anat2vessels.data.fetch import fetch_ref_img

        _REF_IMG_PATH = fetch_ref_img()
    return _REF_IMG_PATH


def __getattr__(name):
    """Module-level fallback attribute access.

    Provides backward-compatible ``REF_IMG_PATH`` via lazy fetch.

    Parameters
    ----------
    name : str
        Attribute name.

    Returns
    -------
    str
        Path to the reference image (if ``name == "REF_IMG_PATH"``).

    Raises
    ------
    AttributeError
        For any other attribute name.
    """
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
    """Preprocess all subjects in a BIDS dataset.

    For each subject, the requested modalities are registered to the reference
    image, resampled, and cropped. Processing can be parallelized with Ray.

    Parameters
    ----------
    bids_dir : str
        Path to the BIDS dataset directory.
    output_dir : str
        Directory where preprocessed images are written.
    model : {"t1", "t2", "t1t2"}, optional
        Determines which modalities to process.
    skull_strip : bool, optional
        Whether to apply skull stripping.
    use_ray : bool, optional
        Enable parallel processing with Ray.

    Raises
    ------
    ValueError
        If no subjects are found in the BIDS directory.
    """
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
    """Preprocess all modalities for a single subject.

    Parameters
    ----------
    layout : BIDSLayout
        The BIDS layout object.
    subject : str
        Subject identifier (e.g. ``"01"``).
    model : {"t1", "t2", "t1t2"}
        Target model.
    skull_strip : bool
        Whether to apply skull stripping.
    output_dir : str
        Output directory.
    """
    if model in ("t1", "t1t2"):
        _process_modality(
            layout,
            subject,
            "T1w",
            [".nii.gz", ".nii"],
            output_dir,
            "t1",
            "0000",
            skull_strip,
        )

    if model in ("t2", "t1t2"):
        _process_modality(
            layout,
            subject,
            "T2w",
            [".nii.gz", ".nii"],
            output_dir,
            "t2",
            "0001" if model == "t1t2" else "0000",
            skull_strip,
        )


def _process_modality(
    layout, subject, suffix, extensions, output_dir, modality, suffix_ext, skull_strip
):
    """Find and preprocess a single modality file for one subject.

    Searches for files matching the given suffix and extension in the BIDS
    layout, then runs :func:`preprocess_img` if the output does not already
    exist.

    Parameters
    ----------
    layout : BIDSLayout
        The BIDS layout object.
    subject : str
        Subject identifier.
    suffix : str
        BIDS suffix (e.g. ``"T1w"``, ``"T2w"``).
    extensions : list of str
        File extensions to try (e.g. ``[".nii.gz", ".nii"]``).
    output_dir : str
        Output directory.
    modality : {"t1", "t2"}
        Modality passed to :func:`preprocess_img`.
    suffix_ext : str
        Output file suffix (e.g. ``"0000"``, ``"0001"``).
    skull_strip : bool
        Whether to apply skull stripping.
    """
    for ext in extensions:
        file = layout.get(
            subject=subject, suffix=suffix, extension=ext, return_type="file"
        )
        if file:
            out = op.join(output_dir, f"{subject}_{suffix_ext}.nii.gz")
            if not op.exists(out):
                preprocess_img(
                    file[0], out, modality=modality, do_skull_strip=skull_strip
                )
            return


def run():
    """CLI entry point for ``a2v-preprocess``.

    Parses command-line arguments and calls :func:`preprocess_bids`.
    """
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
