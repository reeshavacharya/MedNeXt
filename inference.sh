#!/bin/bash -l

#SBATCH --job-name=MedNeXt_S_inference_BTCV
# Use the Quick partition (GPU1 is only in Quick)
#SBATCH --partition=Quick

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
# Quick partition has 1-day limit
#SBATCH --time=1-00:00:00

# Request one A100 GPU on a Quick node
# (best available GPU type for this training)
#SBATCH --gres=gpu:A100:1

#SBATCH --output=inference_std_out
#SBATCH --error=inference_std_err



# Simple inference script for BTCV (Task017) using the trained
# MedNeXt S-variant (3d_fullres, fold 0).
# It runs prediction on the BTCV test set (imagesTs) and writes
# segmentations into a dedicated output folder.

set -e

# Root of this project
PROJECT_ROOT="/home/r/reeshav/MedNeXt"
cd "$PROJECT_ROOT"

# Activate project virtual environment so mednextv1/nnUNet come from .venv
source "$PROJECT_ROOT/.venv/bin/activate"

# nnUNet environment paths (must match those used for training)
# Raw data base is now on the shared /data location
export nnUNet_raw_data_base="/data/reeshav/MedNeXt_dataset/Abdomen"
export nnUNet_preprocessed="$PROJECT_ROOT/nnUNet_preprocessed"
export RESULTS_FOLDER="$PROJECT_ROOT/nnUNet_results"

# Task and model configuration (BTCV = Task017)
TASK_NAME="Task017_AbdominalOrganSegmentation"
NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_MedNeXt_S_kernel3"
PLANS_ID="nnUNetPlansv2.1_trgSp_1x1x1"

# Input BTCV test images (nnUNet raw layout)
INPUT_FOLDER="$nnUNet_raw_data_base/nnUNet_raw_data/${TASK_NAME}/imagesTs"

# Output folder for predicted segmentations
OUTPUT_FOLDER="$PROJECT_ROOT/predictions/BTCV_Task017_MedNeXt_S_kernel3_fold0"
mkdir -p "$OUTPUT_FOLDER"

echo "Running inference for BTCV Task017 using MedNeXt S (fold 0)"

echo "Input imagesTs  : $INPUT_FOLDER"
echo "Output folder   : $OUTPUT_FOLDER"

# Use mednextv1_predict (predict_simple) which locates the model
# from RESULTS_FOLDER/nnUNet based on task, network and trainer.
#
# -i: input folder with *_0000.nii.gz test images
# -o: output folder for segmentations
# -t: task name or ID (here: Task017_AbdominalOrganSegmentation)
# -m: model type (2d, 3d_lowres, 3d_fullres, 3d_cascade_fullres)
# -tr: trainer class name
# -p: plans identifier
# -f 0: use fold_0 only
# --overwrite_existing: overwrite any existing segmentations in OUTPUT_FOLDER

srun mednextv1_predict \
  -i "$INPUT_FOLDER" \
  -o "$OUTPUT_FOLDER" \
  -t "$TASK_NAME" \
  -m "$NETWORK" \
  -tr "$TRAINER" \
  -p "$PLANS_ID" \
  -f 0 \
  --overwrite_existing

echo "Inference finished. Segmentations are in: $OUTPUT_FOLDER" 
