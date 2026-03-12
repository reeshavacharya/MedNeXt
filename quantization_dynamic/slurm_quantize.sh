#!/bin/bash -l

#SBATCH --job-name=quant-net
# Use the Quick partition but allow any available GPU node
#SBATCH --partition=Quick

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00

# Request a single generic GPU on any eligible node in the partition
# (scheduler will choose an available GPU type)
#SBATCH --gres=gpu:A100:1

#SBATCH --output=std_out_quant
#SBATCH --error=std_err_quant

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


