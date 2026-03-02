"""Static PTQ to INT8 for MedNeXt ONNX model using ONNX Runtime.

This script takes the exported FP32 ONNX model from export_onnx.py and
produces an INT8-quantized ONNX model using calibration data drawn from
nnUNet's preprocessed Task017 data.

Requirements (install in your venv):

    pip install onnx onnxruntime onnxruntime-tools

Usage (from project root, with nnUNet env vars set as usual):

    python -m ONNX_quantization.quantize_int8 \
        --fp32_model ONNX_quantization/mednext_task017_fp32.onnx \
        --int8_model ONNX_quantization/mednext_task017_int8.onnx \
        --num_calib 50

The resulting INT8 model can then be run with onnxruntime (optionally
using CUDAExecutionProvider for GPU INT8, if available).
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subfiles,
)

# nnUNet stores the preprocessed base directory in `preprocessing_output_dir`
# within nnunet_mednext.paths. Alias it here for clarity.
from nnunet_mednext.paths import preprocessing_output_dir as nnUNet_preprocessed
from nnunet_mednext.training.model_restore import load_model_and_checkpoint_files


def load_trainer_and_patch_info() -> Tuple[object, Tuple[int, int, int], int]:
    """Load trainer to get patch_size and in_channels.

    We reuse nnUNet's trainer to discover the correct input shape for
    the ONNX model: (1, in_channels, *patch_size).
    """

    from nnunet_mednext.paths import network_training_output_dir

    if network_training_output_dir is None:
        raise RuntimeError(
            "network_training_output_dir is None. Set RESULTS_FOLDER before loading the model."
        )

    task_name = "Task017_AbdominalOrganSegmentation"
    trainer_name = "nnUNetTrainerV2_MedNeXt_S_kernel3"
    plans_id = "nnUNetPlansv2.1_trgSp_1x1x1"

    model_folder = join(
        network_training_output_dir,
        "3d_fullres",
        task_name,
        f"{trainer_name}__{plans_id}",
    )
    if not os.path.isdir(model_folder):
        raise RuntimeError(f"Model folder not found: {model_folder}")

    print(f"[quantize_int8] Loading trainer from {model_folder}")
    trainer, params = load_model_and_checkpoint_files(
        model_folder,
        folds=[0],
        mixed_precision=False,
        checkpoint_name="model_final_checkpoint",
    )
    if not params:
        raise RuntimeError(
            "No checkpoint parameters returned by load_model_and_checkpoint_files"
        )

    trainer.load_checkpoint_ram(params[0], False)
    net = trainer.network

    in_channels = getattr(net, "in_channels", None)
    if in_channels is None:
        in_channels = 1

    patch_size = getattr(trainer, "patch_size", None)
    if patch_size is None:
        raise RuntimeError("trainer.patch_size is None; cannot determine calibration patch size")

    return trainer, tuple(patch_size), int(in_channels)


def get_calibration_files(max_samples: int) -> List[str]:
    """Collect up to max_samples preprocessed .npz files for calibration."""

    if nnUNet_preprocessed is None:
        raise RuntimeError(
            "nnUNet_preprocessed is None. Set nnUNet_preprocessed before running calibration."
        )

    task_name = "Task017_AbdominalOrganSegmentation"
    stage_dir = join(
        nnUNet_preprocessed,
        task_name,
        "nnUNetData_plans_v2.1_trgSp_1x1x1_stage1",
    )
    if not os.path.isdir(stage_dir):
        raise RuntimeError(f"Preprocessed stage1 folder not found: {stage_dir}")

    files = subfiles(stage_dir, suffix=".npz", join=True, sort=True)
    if not files:
        raise RuntimeError(f"No .npz files found in {stage_dir}")

    return files[:max_samples]


def center_crop_or_pad(data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center-crop or pad a (C, D, H, W) volume to (C, *target_shape)."""

    c, d, h, w = data.shape
    td, th, tw = target_shape

    # Pad if needed
    pad_d_before = max((td - d) // 2, 0)
    pad_h_before = max((th - h) // 2, 0)
    pad_w_before = max((tw - w) // 2, 0)

    pad_d_after = max(td - d - pad_d_before, 0)
    pad_h_after = max(th - h - pad_h_before, 0)
    pad_w_after = max(tw - w - pad_w_before, 0)

    if any(p > 0 for p in (pad_d_before, pad_d_after, pad_h_before, pad_h_after, pad_w_before, pad_w_after)):
        data = np.pad(
            data,
            (
                (0, 0),
                (pad_d_before, pad_d_after),
                (pad_h_before, pad_h_after),
                (pad_w_before, pad_w_after),
            ),
            mode="constant",
        )
        _, d, h, w = data.shape

    # Now crop if needed
    start_d = max((d - td) // 2, 0)
    start_h = max((h - th) // 2, 0)
    start_w = max((w - tw) // 2, 0)

    data = data[:, start_d : start_d + td, start_h : start_h + th, start_w : start_w + tw]
    return data


def build_calibration_reader(
    onnx_model_path: Path,
    num_calib: int,
    patch_size: Tuple[int, int, int],
    in_channels: int,
):
    """Create an ONNX Runtime CalibrationDataReader for static PTQ.

    This function is kept local so that we only import onnxruntime when
    actually doing quantization.
    """

    from onnxruntime.quantization import CalibrationDataReader
    import onnx

    model = onnx.load(str(onnx_model_path))
    # Assume a single input named "input"; fall back to first if different
    if len(model.graph.input) == 0:
        raise RuntimeError("ONNX model has no inputs")
    input_name = model.graph.input[0].name

    calib_files = get_calibration_files(num_calib)
    print(f"[quantize_int8] Using {len(calib_files)} calibration samples from preprocessed data")

    class _NNUNetCalibrationDataReader(CalibrationDataReader):
        def __init__(self, files: List[str]):
            self.files = files
            self.index = 0

        def get_next(self):
            if self.index >= len(self.files):
                return None
            f = self.files[self.index]
            self.index += 1

            arr = np.load(f)["data"]  # shape: (C, D, H, W)
            arr = arr[:in_channels]
            arr = center_crop_or_pad(arr, patch_size)
            # Add batch dimension: (1, C, D, H, W)
            arr = np.expand_dims(arr.astype(np.float32), 0)
            return {input_name: arr}

    return _NNUNetCalibrationDataReader(calib_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp32_model",
        type=str,
        required=True,
        help="Path to FP32 ONNX model (exported by export_onnx.py)",
    )
    parser.add_argument(
        "--int8_model",
        type=str,
        default="mednext_task017_int8.onnx",
        help="Output INT8 ONNX model path",
    )
    parser.add_argument(
        "--num_calib",
        type=int,
        default=50,
        help="Number of calibration samples from preprocessed data",
    )
    args = parser.parse_args()

    fp32_model_path = Path(args.fp32_model).resolve()
    int8_model_path = Path(args.int8_model).resolve()

    if not fp32_model_path.is_file():
        raise FileNotFoundError(f"FP32 ONNX model not found: {fp32_model_path}")

    _, patch_size, in_channels = load_trainer_and_patch_info()
    print(f"[quantize_int8] Patch size: {patch_size}, in_channels: {in_channels}")

    calib_reader = build_calibration_reader(fp32_model_path, args.num_calib, patch_size, in_channels)

    print(f"[quantize_int8] Quantizing {fp32_model_path} -> {int8_model_path}")

    from onnxruntime.quantization import (
        quantize_static,
        CalibrationMethod,
        QuantType,
        QuantFormat,
    )

    int8_model_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(fp32_model_path),
        model_output=str(int8_model_path),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )

    print(f"[quantize_int8] INT8 model saved to {int8_model_path}")
    print(
        "You can now run inference with onnxruntime using this INT8 model. "
        "For GPU INT8, ensure CUDAExecutionProvider with INT8 support is available."
    )


if __name__ == "__main__":
    main()
