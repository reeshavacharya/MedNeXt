#!/bin/bash -l

#SBATCH --job-name=zip-table5
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
#SBATCH --gres=gpu:1

#SBATCH --output=std_out
#SBATCH --error=std_err

set -e

cd /home/r/reeshav/MedNeXt

# activate project virtual environment so nnunet/mednextv1 come from .venv
source /home/r/reeshav/MedNeXt/.venv/bin/activate

export nnUNet_raw_data_base=/data/reeshav/MedNeXt_dataset/Abdomen
export nnUNet_preprocessed=/home/r/reeshav/MedNeXt/nnUNet_preprocessed
export RESULTS_FOLDER=/home/r/reeshav/MedNeXt/nnUNet_results

# echo "Starting nnUNet planning and preprocessing for Task 17 (1x1x1 spacing)" 
# srun mednextv1_plan_and_preprocess -t 17 \
# 	-pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 \
# 	-pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1

echo "Starting MedNeXt S-variant training (3d_fullres, Task017, fold 0)" 
srun mednextv1_train 3d_fullres \
	nnUNetTrainerV2_MedNeXt_S_kernel3 \
	Task017_AbdominalOrganSegmentation \
	0 \
	-p nnUNetPlansv2.1_trgSp_1x1x1 \
	-c