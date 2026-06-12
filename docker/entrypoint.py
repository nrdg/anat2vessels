#!/usr/bin/env python3
import os
import subprocess
import argparse

NNUNET_PARAMS = {
    "t1": {
        "dataset": "076",
        "trainer": "nnUNetTrainer",
        "config": "3d_fullres",
        "plans": "nnUNetResEncUNetLPlans",
    },
    "t2": {
        "dataset": "086",
        "trainer": "nnUNetTrainer",
        "config": "3d_fullres",
        "plans": "nnUNetResEncUNetLPlans",
    },
    "t1t2": {
        "dataset": "096",
        "trainer": "nnUNetTrainerCLDLoss",
        "config": "3d_fullres",
        "plans": "nnUNetResEncUNetMPlans",
    },
}


def cmd_preprocess(args):
    """Run the preprocessing step of the pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Must have ``bids_dir``, ``output_dir``, ``model``, ``skull_strip``,
        and ``no_ray`` attributes.
    """
    from anat2vessels.preprocess import preprocess_bids

    preprocess_bids(
        bids_dir=args.bids_dir,
        output_dir=args.output_dir,
        model=args.model,
        skull_strip=args.skull_strip,
        use_ray=not args.no_ray,
    )


def cmd_predict(args):
    """Run nnUNet prediction on preprocessed images.

    Parameters
    ----------
    args : argparse.Namespace
        Must have ``input_dir``, ``output_dir``, ``model``, ``device``,
        and ``folds`` attributes.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    params = NNUNET_PARAMS[args.model]
    env = os.environ.copy()
    cmd = [
        "nnUNetv2_predict",
        "-d",
        params["dataset"],
        "-i",
        args.input_dir,
        "-o",
        args.output_dir,
        "-f",
    ]
    if args.folds:
        cmd += args.folds.split()
    else:
        cmd += ["0", "1", "2", "3", "4"]

    cmd += [
        "-tr",
        params["trainer"],
        "-c",
        params["config"],
        "-p",
        params["plans"],
    ]
    if args.device:
        cmd += ["--device", args.device]
    subprocess.run(cmd, env=env, check=True)


def cmd_features(args):
    """Run vessel feature extraction on prediction outputs.

    Parameters
    ----------
    args : argparse.Namespace
        Must have ``input_dir``, ``output_path``, and ``no_ray`` attributes.
    """
    from anat2vessels.vessels2csv import run_feature_extraction

    run_feature_extraction(
        input_dir=args.input_dir,
        output_path=args.output_path,
        use_ray=not args.no_ray,
    )


def cmd_all(args):
    """Run the full pipeline: preprocess, predict, and extract features.

    Parameters
    ----------
    args : argparse.Namespace
        Must have ``bids_dir``, ``output_dir``, ``model``, ``skull_strip``,
        ``no_ray``, and ``device`` attributes.
    """
    preproc_dir = os.path.join(args.output_dir, "preprocessed")
    pred_dir = os.path.join(args.output_dir, "predictions")
    features_csv = os.path.join(args.output_dir, "features.csv")

    preproc_args = argparse.Namespace(
        bids_dir=args.bids_dir,
        output_dir=preproc_dir,
        model=args.model,
        skull_strip=args.skull_strip,
        no_ray=args.no_ray,
    )
    cmd_preprocess(preproc_args)

    predict_args = argparse.Namespace(
        input_dir=preproc_dir,
        output_dir=pred_dir,
        model=args.model,
        device=args.device,
    )
    cmd_predict(predict_args)

    features_args = argparse.Namespace(
        input_dir=pred_dir,
        output_path=features_csv,
        no_ray=args.no_ray,
    )
    cmd_features(features_args)

    print(f"Pipeline complete. Results: {features_csv}")


def main():
    """CLI entry point for the Docker pipeline (preprocess | predict | features | all).

    Parses command-line arguments and dispatches to the appropriate subcommand.
    """
    parser = argparse.ArgumentParser(description="Anat2Vessels full pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_pre = subparsers.add_parser("preprocess", help="Preprocess BIDS dataset")
    p_pre.add_argument("--bids_dir", required=True)
    p_pre.add_argument("--output_dir", required=True)
    p_pre.add_argument("--model", choices=["t1", "t2", "t1t2"], default="t1t2")
    p_pre.add_argument("--skull_strip", action="store_true")
    p_pre.add_argument("--no_ray", action="store_true")
    p_pre.set_defaults(func=cmd_preprocess)

    p_pred = subparsers.add_parser("predict", help="Run nnUNet inference")
    p_pred.add_argument("--input_dir", required=True)
    p_pred.add_argument("--output_dir", required=True)
    p_pred.add_argument("--model", choices=["t1", "t2", "t1t2"], required=True)
    p_pred.add_argument("--device", default=None)
    p_pred.add_argument("--folds", default=None)
    p_pred.set_defaults(func=cmd_predict)

    p_feat = subparsers.add_parser("features", help="Extract vessel features")
    p_feat.add_argument("--input_dir", required=True)
    p_feat.add_argument("--output_path", required=True)
    p_feat.add_argument("--no_ray", action="store_true")
    p_feat.set_defaults(func=cmd_features)

    p_all = subparsers.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--bids_dir", required=True)
    p_all.add_argument("--output_dir", required=True)
    p_all.add_argument("--model", choices=["t1", "t2", "t1t2"], default="t1t2")
    p_all.add_argument("--skull_strip", action="store_true")
    p_all.add_argument("--no_ray", action="store_true")
    p_all.add_argument("--device", default=None)
    p_all.set_defaults(func=cmd_all)

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
