#!/bin/bash -l

#SBATCH --job-name=eval-onnx
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

#SBATCH --output=std_out_onnx_eval
#SBATCH --error=std_err_onnx_eval

set -euo pipefail

# Explicitly set project root (do NOT rely on SLURM's copy location)
ROOT_DIR="/home/r/reeshav/MedNeXt"
cd "${ROOT_DIR}"

# Activate venv
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[run_onnx] .venv not found in ${ROOT_DIR}. Please activate your env manually." >&2
fi

# nnUNet paths (adjust if your setup differs)
export nnUNet_raw_data_base=/data/reeshav/MedNeXt_dataset/Abdomen
export nnUNet_preprocessed="${ROOT_DIR}/nnUNet_preprocessed"
export RESULTS_FOLDER="${ROOT_DIR}/nnUNet_results"


# 1) Export FP32 ONNX model
FP32_ONNX="ONNX_quantization/mednext_task017_fp32.onnx"
INT8_ONNX="ONNX_quantization/mednext_task017_int8.onnx"

srun python -m ONNX_quantization.evaluate_onnx \
  --int8_model ONNX_quantization/mednext_task017_int8.onnx \
  --num_cases 10