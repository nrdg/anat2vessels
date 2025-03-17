### Vessel Segmentation

## Inferance Instructions

# To run inference please first run the preprocess_imgs.py script on your data

Example: $python preprocess_imgs.py --t1_dir ./t1w --t2_dir ./t2w --output_dir ./preprocessed --id_delim "_"

This script will regester all images to a reference image, and resample them to a shape of (512,512,160)

Optionally add --skull_strip if your data is not skull stripped already. This requires antspynet to be installed
aswell as tensorflow, to use gpus for this make sure you install $pip install 'tensorflow[and-cuda]'.

# Setting up nnUNet

Install nnUNet according to their instruction found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).

Then download the model zip of choice from our repository found [here](https://doi.org/10.6084/m9.figshare.27040633.v1), and installing running $nnUNetv2_install_pretrained_model_from_zip model_file.zip

finally to run inference on a preprocessed folder run on of the following four commands based on the model you downloaded:

t2 only model:

$ nnUNetv2_predict -d 086 -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlans

t1 only model:

$ nnUNetv2_predict -d 076 -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlans

t1 + t2:

$ nnUNetv2_predict -d 096 -i INPUT_FOLDER -o OUTPUT_FOLDER -f  0 1 2 3 4 -tr nnUNetTrainerCLDLoss -c 3d_fullres -p nnUNetResEncUNetMPlans


For usage of the docker containers provided [here](https://hub.docker.com/repository/docker/asaagilmore/anatomical_vessel_seg/tags),
simply clone the tagged docker image for the model you want (t1,t2,t1t2), and then
run it with your directory bound as follows. Note each container will only work with
one of the models (76,86,96), so make sure to get the right container.

docker run -it --gpus=1 -v /path/to/data:/data asaagilmore/anatomical_vessel_seg:t1t2 ...INFERENCE_COMMAND_FROM_ABOVE

## Feature extraction

Feature extraction can be easily performed using the csv_from_predictions.py script. You may need to use cython to build the feature_extraction model if the prebuilt binaries do not work.
To do this navigate to the feature_extraction folder and run "$python setup.py build_ext --inplace". Once installed you can simply run "$python csv_from_predictions.py --input_dir /path/to/predictions --output_path path/to/save/features.csv"
Its recommended, though not required, to install ray using "$pip install ray" to run this feature extraction cores.

