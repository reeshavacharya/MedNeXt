"""MODEL QUANTIZATION MODULE

This module converts an FP32 PyTorch model into a quantized model using
calibration statistics computed in :mod:`calibrate`. It supports
integer precisions:

- INT8
- INT6 (simulated using int8 containers)
- INT4 (two logical values per int8 via bit-packing)

There are two main entry points:

1. :func:`quantize_model` – a library helper that expects an already
    loaded model and precomputed calibration parameters.
2. A command-line interface (``python -m quantization_dynamic.quantize``)
    that orchestrates the full dynamic pipeline:

    FP32 model → calibration → quantization (+bit-packing) → ONNX export
    → TensorRT engine build → TensorRT inference.

For now the CLI is focused on INT8, min-max calibration, per-layer
scales, and a fixed MedNeXt Task017 BTCV configuration.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import argparse
import logging
import os
from os.path import join

import numpy as np
import torch
import torch.nn as nn

from batchgenerators.utilities.file_and_folder_operations import subfiles

from nnunet_mednext.paths import (
    preprocessing_output_dir,
    network_training_output_dir,
)
from nnunet_mednext.training.model_restore import (
    load_model_and_checkpoint_files,
)

from .bitpacking import pack_int4, pack_int6
from .calibrate import run_calibration
from .onnx_export import export_quantized_onnx
from .tensorrt_engine import build_tensorrt_engine
from .trt_inference import TensorRTInferenceSession
from .model_wrappers import PatchInferenceWrapper, QuantizedModelWrapper
from .quant_utils import SUPPORTED_QUANT_DTYPES, default_tensorrt_config


logger = logging.getLogger(__name__)


def quantize_model(
        model: nn.Module,
        quant_params: Dict[str, Dict[str, float]],
        quant_dtype: str = "int8",
        pack_weights: bool = True,
        pack_dim: int = 1,
        save_path: Optional[str] = None,
) -> QuantizedModelWrapper:
    """Convert an FP32 model into a quantized model.

    Parameters
    ----------
    model:
        Pretrained FP32 PyTorch model (already loaded by caller).
    quant_params:
        Dictionary of per-layer calibration parameters produced by
        :func:`calibrate.run_calibration`. Expected keys per layer:
        ``{"scale", "zero_point", "min_val", "max_val"}``.
    quant_dtype:
        Quantization dtype ("int8", "int6", "int4").
    pack_weights:
        If True, apply INT6/INT4 bit-packing for storage.
    pack_dim:
        Dimension along which to pack INT4 values (for Conv weights this is
        typically the input-channel dimension: 1).
    save_path:
        Optional path to ``torch.save`` the resulting quantized model
        (state_dict).

    Returns
    -------
    quantized_model:
        A :class:`QuantizedModelWrapper` instance wrapping the quantized
        model. The underlying weights have been quantize–dequantized in
        FP32 for approximate behavior, and integer weights + packed
        representations are stored as buffers on each layer.
    """

    qd = quant_dtype.lower()
    if qd not in SUPPORTED_QUANT_DTYPES:
        raise ValueError(
            f"Unsupported quantization dtype '{quant_dtype}'. Supported: {SUPPORTED_QUANT_DTYPES}"
        )

    logger.info("Creating QuantizedModelWrapper (dtype=%s)", qd)
    q_model = QuantizedModelWrapper(model, quant_params, quant_dtype=qd)

    if pack_weights and qd in ("int6", "int4"):
        logger.info("Applying bit-packing for %s weights", qd)
        # QuantizedModelWrapper attaches 'weight_qint' buffers to each
        # quantized Conv/Linear. We pack those into tighter int8
        # representations suitable for downstream export.
        for name, module in q_model.model.named_modules():
            q_w = getattr(module, "weight_qint", None)
            if q_w is None:
                continue

            if qd == "int6":
                packed = pack_int6(q_w)
                module.register_buffer("weight_packed_int6", packed.cpu())
                logger.debug("Packed INT6 weights for layer '%s'", name)
            elif qd == "int4":
                # Pack along the specified dimension (e.g., in_channels)
                orig_len = q_w.shape[pack_dim]
                packed = pack_int4(q_w, dim=pack_dim)
                module.register_buffer("weight_packed_int4", packed.cpu())
                module.register_buffer(
                    "weight_packed_int4_orig_len",
                    torch.tensor(orig_len, dtype=torch.int32),
                )
                logger.debug("Packed INT4 weights for layer '%s' (dim=%d)", name, pack_dim)

    if save_path is not None:
        logger.info("Saving quantized model to %s", save_path)
        torch.save(q_model.state_dict(), save_path)

    return q_model


# ---------------------------------------------------------------------------
# CLI orchestration: FP32 → calibration → quantization → ONNX → TRT → infer
# ---------------------------------------------------------------------------


def _get_model_folder() -> str:
    """Locate the trained MedNeXt-S model folder for Task017 (BTCV).

    This mirrors the default nnUNet/MedNeXt training layout.
    """

    if network_training_output_dir is None:
        raise RuntimeError(
            "network_training_output_dir is None. Make sure RESULTS_FOLDER "
            "is set before running the dynamic quantization pipeline."
        )

    task_name = "Task017_AbdominalOrganSegmentation"
    trainer = "nnUNetTrainerV2_MedNeXt_S_kernel3"
    plans_id = "nnUNetPlansv2.1_trgSp_1x1x1"

    model_folder = join(
        network_training_output_dir,
        "3d_fullres",
        task_name,
        f"{trainer}__{plans_id}",
    )

    if not os.path.isdir(model_folder):
        raise RuntimeError(
            f"Model folder not found: {model_folder}. "
            f"Check RESULTS_FOLDER/nnUNet/3d_fullres paths."
        )
    return model_folder


def _get_calibration_files(max_cases: int) -> Iterable[str]:
    """Return a list of preprocessed training cases for calibration."""

    if preprocessing_output_dir is None:
        raise RuntimeError(
            "preprocessing_output_dir is None. Make sure nnUNet_preprocessed "
            "is set before running the dynamic quantization pipeline."
        )

    task_dir = join(
        preprocessing_output_dir,
        "Task017_AbdominalOrganSegmentation",
        "nnUNetData_plans_v2.1_trgSp_1x1x1_stage1",
    )
    if not os.path.isdir(task_dir):
        raise RuntimeError(
            f"Preprocessed Task017 data not found at {task_dir}. "
            "Run mednextv1_plan_and_preprocess first."
        )

    files = subfiles(task_dir, suffix=".npz", join=True, sort=True)
    if not files:
        raise RuntimeError(
            f"No .npz files found in preprocessed Task017 directory: {task_dir}"
        )

    return files[: max_cases or len(files)]


def _build_calibration_loader(
        calib_files: Iterable[str],
        stem_in_ch: Optional[int],
        device: torch.device,
        model: nn.Module,
        patch_size: tuple[int, int, int] = (128, 128, 128),
        stride: tuple[int, int, int] = (128, 128, 128),
) -> Iterable[torch.Tensor]:
    """Create a calibration loader that yields fixed-size 3D patches.

    For each preprocessed volume, this function:

    1. Loads the nnUNet-style ``data`` array with shape (C, D, H, W).
    2. Optionally slices channels to ``stem_in_ch`` to match the model.
    3. Uses :class:`PatchInferenceWrapper`'s ``generate_patches`` method
       to extract "safe" patches of size ``patch_size`` (default
       128 x 128 x 128) with the given ``stride``.

    Each yielded element is a single patch tensor of shape
    (C, pD, pH, pW) moved to ``device``. These patches are then fed
    directly to :func:`run_calibration`, which wraps ``model`` in a
    :class:`CalibrationWrapper` and runs forward passes on patches
    instead of full volumes.
    """

    patch_wrapper = PatchInferenceWrapper(model, patch_size=patch_size, stride=stride, device=device)

    for f in calib_files:
        data = np.load(f)["data"]  # same format used by nnUNet inference
        if stem_in_ch is not None:
            data = data[:stem_in_ch]
        # data has shape (C, D, H, W)
        vol = torch.from_numpy(data.astype(np.float32, copy=False)).to(device)

        # Use PatchInferenceWrapper's sliding-window patch generator.
        for patch, _coords in patch_wrapper.generate_patches(vol):
            yield patch


def _load_fp32_model(device: torch.device) -> nn.Module:
    """Load the trained MedNeXt-S FP32 model (Task017, fold 0)."""

    model_folder = _get_model_folder()
    logger.info("[CLI] Loading FP32 model from %s", model_folder)
    trainer, params = load_model_and_checkpoint_files(
        model_folder,
        folds=[0],
        mixed_precision=False,
        checkpoint_name="model_final_checkpoint",
    )
    if not params:
        raise RuntimeError("No checkpoint parameters returned by load_model_and_checkpoint_files")

    trainer.load_checkpoint_ram(params[0], False)
    net = trainer.network
    net.eval()
    net.to(device)
    return net


def _get_stem_in_channels(net: nn.Module) -> Optional[int]:
    stem_in_ch = getattr(getattr(net, "stem", None), "in_channels", None)
    if stem_in_ch is None:
        logger.warning("[CLI] Could not determine stem in_channels; using data channels as-is")
    else:
        logger.info("[CLI] Network stem expects %d input channel(s)", stem_in_ch)
    return stem_in_ch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic INT8 PTQ + TensorRT pipeline for MedNeXt Task017")
    parser.add_argument("--quant_dtype", default="int8", choices=["int8"], help="Quantization dtype (currently only int8 is wired up)")
    parser.add_argument("--calibration_method", default="minmax", choices=["minmax"], help="Activation calibration method")
    parser.add_argument("--per_channel", action="store_true", help="(Reserved) Per-channel activation scales (not used yet)")
    parser.add_argument("--num_cases", type=int, default=10, help="Number of calibration cases to use")
    parser.add_argument("--output_dir", default="dynamic_trt_results", help="Output directory for artifacts")
    parser.add_argument("--onnx_path", default="quantization_dynamic/model_int8.onnx", help="Path to save ONNX model")
    parser.add_argument("--engine_path", default="quantization_dynamic/model_int8.engine", help="Path to save TensorRT engine")
    parser.add_argument("--export_tensorrt_engine", action="store_true", help="Build TensorRT engine from ONNX")
    parser.add_argument("--run_trt_inference", action="store_true", help="Run a quick TensorRT inference sanity check")
    return parser.parse_args()


def main() -> None:
    """CLI entry: run full INT8 PTQ + TensorRT pipeline for MedNeXt Task017."""

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[CLI] Using device: %s", device)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("[CLI] Output directory: %s", args.output_dir)

    # 1) Load FP32 model
    net = _load_fp32_model(device)
    stem_in_ch = _get_stem_in_channels(net)

    # 2) Build calibration loader (patch-based) and run calibration (INT8, minmax)
    calib_files = list(_get_calibration_files(args.num_cases))
    logger.info("[CLI] Using %d calibration cases", len(calib_files))
    calib_loader = _build_calibration_loader(calib_files, stem_in_ch, device, model=net)

    logger.info("[CLI] Starting calibration (method=%s)", args.calibration_method)
    quant_params = run_calibration(
        net,
        calib_loader,
        quant_dtype=args.quant_dtype,
        calibration_method=args.calibration_method,
        symmetric=True,
        device=device,
    )
    logger.info("[CLI] Calibration produced quant params for %d layers", len(quant_params))

    # 3) Quantize model (weights) and optionally apply bit-packing
    logger.info("[CLI] Quantizing model weights (dtype=%s)", args.quant_dtype)
    q_model = quantize_model(net, quant_params, quant_dtype=args.quant_dtype, pack_weights=False)

    # 4) Export ONNX
    onnx_path = args.onnx_path
    logger.info("[CLI] Exporting quantized model to ONNX: %s", onnx_path)
    export_quantized_onnx(q_model, output_path=onnx_path)

    # 5) Build TensorRT engine (optional)
    if args.export_tensorrt_engine:
        logger.info("[CLI] Building TensorRT engine at %s", args.engine_path)
        trt_config = default_tensorrt_config()
        build_tensorrt_engine(onnx_path, engine_path=args.engine_path, config=trt_config)

    # 6) Quick TensorRT inference sanity check (optional)
    if args.run_trt_inference:
        logger.info("[CLI] Running a quick TensorRT inference sanity check")
        trt_session = TensorRTInferenceSession(args.engine_path)
        # Use one of the calibration volumes as a test volume
        if not calib_files:
            logger.warning("[CLI] No calibration files available for TRT test; skipping inference")
        else:
            data = np.load(calib_files[0])["data"]
            if stem_in_ch is not None:
                data = data[:stem_in_ch]
            vol = data.astype(np.float32, copy=False)
            # TRT expects (N,C,D,H,W)
            input_batch = vol[None, ...]
            pred = trt_session.infer(input_batch)
            logger.info("[CLI] TensorRT inference output shape: %s", tuple(pred.shape))

    logger.info("[CLI] Dynamic INT8 quantization + TensorRT pipeline completed")


if __name__ == "__main__":
    main()

