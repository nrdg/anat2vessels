# Anat2Vessel
![Anat2Vess Logo](/figures/Logo.png)
Anat2Vess is a deep learning based tool which allows for users to extract information about brain vasculature from anatomical (T1 & T2) MRI scans. The primary use case for this tool is to add vasculature information to existing dataset in which vessel imaging was not included.

For further details on training, validation and performance of our model, see our paper [here](https://www.biorxiv.org/content/10.1101/2025.05.06.652518v1).

## Step 1: Data Preprocessing

Before running the segmentation model on your dataset it's necessary to preprocess
the data.

We provide a python script which automates this process. To run this script,
we recommend using a virtual environment like conda for this.

Example:

```bash
pip install -r requirements.txt
python preprocess_imgs.py --t1_dir ./t1w --t2_dir ./t2w --output_dir ./preprocessed --id_delim "_"
```

This script will register all images to a reference image, and resample them to a shape of (512,512,160) with 1mm isotropic voxels.

The Anat2Vessel model requires the data to be skull stripped,
this can be done with the preprocess_imgs.py script by adding the --skull_strip
argument to the command. This requires antspynet and tensorflow to be installed.
If using nvidia gpu for this, make sure to install tensorflow with
```bash
pip install 'tensorflow[and-cuda]'
```

If running inference on a large dataset, optionally install ray,
and the script with run in parallel using all available cores.
To disable this use the --no_ray flag.
## Step 2: Setting up nnUNet

To run inference on the Anat2Vessel model, you can either install
nnUNet locally, or use the docker containers provided.

### local installation
First, install nnUNet according to their instruction found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).

Then download the model .zip file of choice from our repository found [here](https://huggingface.co/huggingbrain/AnatomicalVesselSeg), and install running $nnUNetv2_install_pretrained_model_from_zip /path/to/model_file.zip

finally to run inference on a preprocessed folder run on of the following four commands based on the model you downloaded:

t2 only model:
```bash
nnUNetv2_predict -d 086 -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlans
```
t1 only model:
```bash
nnUNetv2_predict -d 076 -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlans
```
t1 + t2:
```bash
nnUNetv2_predict -d 096 -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainerCLDLoss -c 3d_fullres -p nnUNetResEncUNetMPlans
```

### Docker usage

For usage of the docker containers provided [here](https://hub.docker.com/repository/docker/asaagilmore/anatomical_vessel_seg/tags),
simply clone the tagged docker image for the model you want (t1,t2,t1t2), and then
run it with your directory bound as follows. Note each container will only work with
one of the models (76,86,96), so make sure to get the right container.

```bash
docker run -it --gpus=1 -v /path/to/data:/data asaagilmore/anatomical_vessel_seg:t1t2 ...INFERENCE_COMMAND_FROM_ABOVE
```
Note that this requires the NVIDIA container toolkit to be installed on your machine
for GPU support.

## Step 3: Feature extraction

We provide the script used in our paper to extract anatomical features from the model predictions.
This script will extract the features from the model predictions and save them to a .csv file.

Feature extraction can be easily performed using the csv_from_predictions.py script.

To do simply run the following command:
```bash
python csv_from_predictions.py --input_dir /path/to/predictions --output_path path/to/save/features.csv
```

If ray is installed it will run in parallel using all available cores,
to disable this use the --no_ray flag.


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