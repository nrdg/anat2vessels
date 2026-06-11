# Anat2Vessel
![Anat2Vess Logo](/figures/Logo.png)
Anat2Vess is a deep learning based tool which allows for users to extract information about brain vasculature from anatomical (T1 & T2) MRI scans. The primary use case for this tool is to add vasculature information to existing dataset in which vessel imaging was not included.

For further details on training, validation and performance of our model, see our preprint [here](https://www.biorxiv.org/content/10.1101/2025.05.06.652518v1).

## Usage

The pipeline assumes your data is organized according to the
[BIDS standard](https://bids.neuroimaging.io/).

### Docker (Recommended)

Build the image:

```bash
docker build -f docker/dockerfile -t anat2vessels:latest .
```

Run the full pipeline (preprocess + inference + feature extraction) in one command:

```bash
docker run --gpus=1 \
  -v /path/to/bids:/data/bids \
  -v /path/to/results:/data/results \
  anat2vessels:latest all \
  --bids_dir /data/bids \
  --output_dir /data/results \
  --model t1t2 \
  --skull_strip
```

Or run individual steps:

```bash
# Step 1: Preprocess
docker run --gpus=1 -v /path/to/bids:/data/bids -v /path/to/results:/data/results \
  anat2vessels:latest preprocess \
  --bids_dir /data/bids --output_dir /data/results/preprocessed \
  --model t1t2 --skull_strip

# Step 2: nnUNet inference
docker run --gpus=1 -v /path/to/results:/data/results \
  anat2vessels:latest predict \
  --input_dir /data/results/preprocessed --output_dir /data/results/predictions \
  --model t1t2

# Step 3: Feature extraction
docker run --gpus=1 -v /path/to/results:/data/results \
  anat2vessels:latest features \
  --input_dir /data/results/predictions --output_path /data/results/features.csv
```

| `--model` | Description |
|-----------|-------------|
| `t1` | T1-weighted only |
| `t2` | T2-weighted only |
| `t1t2` | Combined T1 + T2 (default) |

Note: GPU support requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Local installation

Install the package:

```bash
pip install anat2vessels
```

Preprocess a BIDS dataset:

```bash
a2v-preprocess --bids_dir ./bids --output_dir ./preprocessed --model t1t2 --skull_strip
```

Run inference with nnUNet (requires separate nnUNet installation):

```bash
nnUNetv2_predict -d 096 -i ./preprocessed -o ./predictions -f 0 1 2 3 4 -tr nnUNetTrainerCLDLoss -c 3d_fullres -p nnUNetResEncUNetMPlans
```

Extract features:

```bash
a2v-features --input_dir ./predictions --output_path ./features.csv
```


## Citation

If you used this tool to assist with work related to a publication please
cite the following reference:
```
@article {Gilmore2025.05.06.652518,
	author = {Gilmore, Asa and Eshun, Anita Esi and Wu, Yue and Lee, Aaron Y and Rokem, Ariel},
	title = {Vessels hiding in plain sight: quantifying brain vascular morphology in anatomical MR images using deep learning},
	elocation-id = {2025.05.06.652518},
	year = {2025},
	doi = {10.1101/2025.05.06.652518},
	URL = {https://www.biorxiv.org/content/early/2025/05/11/2025.05.06.652518},
	journal = {bioRxiv}
}
```