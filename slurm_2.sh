#!/bin/bash -l

#SBATCH --job-name=zip-table5
#SBATCH --partition=Quick

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00

# Request ONE GPU of ANY type
#SBATCH --gres=gpu:1

#SBATCH --output=std_out
#SBATCH --error=std_err

set -e

cd /home/r/reeshav/MedNeXt
source /home/r/reeshav/MedNeXt/.venv/bin/activate

export nnUNet_raw_data_base=/home/r/reeshav/MedNeXt/dataset
export nnUNet_preprocessed=/home/r/reeshav/MedNeXt/nnUNet_preprocessed
export RESULTS_FOLDER=/home/r/reeshav/MedNeXt/nnUNet_results

echo "Starting MedNeXt S-variant training (3d_fullres, Task017, fold 0)"
srun mednextv1_train 3d_fullres \
    nnUNetTrainerV2_MedNeXt_S_kernel3 \
    Task017_AbdominalOrganSegmentation \
    0 \
    -p nnUNetPlansv2.1_trgSp_1x1x1 \
    -c
