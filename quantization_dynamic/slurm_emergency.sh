#!/bin/bash -l

#SBATCH --job-name=quant-reeshav
# Change to 'general' since GPU6, 13, 14 are IDLE
#SBATCH --partition=general

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00

# Request any GPU in the general partition
#SBATCH --gres=gpu:1

#SBATCH --output=std_out_quant_emergency
#SBATCH --error=std_err_quant_emergency

set -e

cd /home/r/reeshav/MedNeXt
source .venv/bin/activate

export nnUNet_raw_data_base=/data/reeshav/MedNeXt_dataset/Abdomen
export nnUNet_preprocessed=/home/r/reeshav/MedNeXt/nnUNet_preprocessed
export RESULTS_FOLDER=/home/r/reeshav/MedNeXt/nnUNet_results


echo "[SLURM] Job started at $(date)"
echo "[SLURM] Running on host $(hostname)"
echo "[SLURM] CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "[SLURM] GPU status before run:"
nvidia-smi || echo "[SLURM] nvidia-smi not available"

echo "[SLURM] =========================================================="
echo "[SLURM] Step 1/5: FP32 model loading + environment setup (inside Python)"
echo "[SLURM] Step 2/5: Calibration and activation statistics collection"
echo "[SLURM] Step 3/5: Weight quantization and (optional) bit-packing"
echo "[SLURM] Step 4/5: ONNX export and TensorRT INT8 engine build"
echo "[SLURM] Step 5/5: TensorRT inference on validation / test data"
echo "[SLURM] =========================================================="

echo "[SLURM] Launching Python quantization + TensorRT pipeline via quantization_dynamic.quantize"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python -m quantization_dynamic.quantize \
	--quant_dtype int8 \
	--calibration_method minmax \
	--per_channel \
	--num_cases 1 \
	--output_dir dynamic_trt_results \
	--export_tensorrt_engine \
	--run_trt_inference \
	--onnx_path quantization_dynamic/model_int8.onnx \
	--engine_path quantization_dynamic/model_int8.engine


echo "[SLURM] Verifying generated ONNX and TensorRT engine files from Python pipeline"
ls -lh quantization_dynamic/model_int8.onnx quantization_dynamic/model_int8.engine || \
	echo "[SLURM] Warning: ONNX/engine files not found as expected"

echo "[SLURM] Job finished at $(date)"



