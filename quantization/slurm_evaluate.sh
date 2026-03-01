#!/bin/bash -l

#SBATCH --job-name=quant-net
# Use the Quick partition but allow any available GPU node
#SBATCH --partition=Quick

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00

# Request a single generic GPU on any eligible node in the partition
# (scheduler will choose an available GPU type)
#SBATCH --gres=gpu:1

#SBATCH --output=std_out_evaluate_20
#SBATCH --error=std_err_evaluate_20

set -e

cd /home/r/reeshav/MedNeXt
source .venv/bin/activate

export nnUNet_raw_data_base=/data/reeshav/MedNeXt_dataset/Abdomen
export nnUNet_preprocessed=/home/r/reeshav/MedNeXt/nnUNet_preprocessed
export RESULTS_FOLDER=/home/r/reeshav/MedNeXt/nnUNet_results

srun python -m quantization.evaluate