from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union
import logging
import numpy as np
import torch


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported quantization dtypes and ranges
# ---------------------------------------------------------------------------

SUPPORTED_QUANT_DTYPES: Tuple[str, ...] = ("int8", "int6", "int4")

_DTYPE_TO_RANGE: Dict[str, Tuple[int, int]] = {
    "int8": (-128, 127),
    "int6": (-32, 31),
    "int4": (-8, 7),
}


def get_quant_range(quant_dtype: str) -> Tuple[int, int]:
    """Return integer range (qmin, qmax) for a given quantization dtype.

    Parameters
    ----------
    quant_dtype:
        One of "int8", "int6", "int4" (case-insensitive).
    """

    dtype_key = quant_dtype.lower()
    if dtype_key not in _DTYPE_TO_RANGE:
        raise ValueError(f"Unsupported quantization dtype: {quant_dtype}")
    return _DTYPE_TO_RANGE[dtype_key]


# ---------------------------------------------------------------------------
# Calibration methods and configuration
# ---------------------------------------------------------------------------


SUPPORTED_CALIBRATION_METHODS: Tuple[str, ...] = (
    "minmax",
    "percentile",
    "kl",
    "mse",
    "omse",
    "aciq",
)


def normalize_calibration_method(method: str) -> str:
    """Validate and normalize calibration method string."""

    method_l = method.lower()
    if method_l not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(
            f"Unsupported calibration method '{method}'. "
            f"Supported: {SUPPORTED_CALIBRATION_METHODS}"
        )
    logger.info("Calibration method: %s", method_l)
    return method_l


@dataclass
class PatchSettings:
    """Patch / sliding-window inference settings used during calibration.

    These mirror the arguments typically passed to nnUNet's
``predict_preprocessed_data_return_seg_and_softmax``.
    """

    use_sliding_window: bool = True
    step_size: float = 0.5
    use_gaussian: bool = True
    pad_mode: str = "constant"
    pad_constant: float = 0.0
    do_mirroring: bool = True
    mirror_axes: Tuple[int, ...] = (0, 1, 2)


def default_patch_settings() -> PatchSettings:
    """Return default patch settings for 3D segmentation calibration."""

    return PatchSettings()


@dataclass
class TensorRTConfig:
    """Basic TensorRT configuration options used when building engines."""

    int8: bool = True
    fp16: bool = False
    max_workspace_size_bytes: int = 4 * 1024**3  # 4 GB
    strict_types: bool = False


def default_tensorrt_config() -> TensorRTConfig:
    """Return default TensorRT configuration for INT8 engines."""

    logger.info("Using default TensorRT configuration (INT8 enabled)")
    return TensorRTConfig()


@dataclass
class OnnxExportConfig:
    """ONNX export configuration shared across the pipeline."""

    opset_version: int = 17
    do_constant_folding: bool = True
    input_names: Tuple[str, ...] = ("input",)
    output_names: Tuple[str, ...] = ("output",)
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None


def default_onnx_export_config() -> OnnxExportConfig:
    """Return default ONNX export configuration for segmentation models."""

    logger.info("Using default ONNX export config (opset=17)")
    return OnnxExportConfig()


# ---------------------------------------------------------------------------
# Scale and zero-point computation
# ---------------------------------------------------------------------------


def compute_scale(
        min_val: Union[float, np.ndarray, torch.Tensor],
        max_val: Union[float, np.ndarray, torch.Tensor],
        qmax: int,
) -> float:
    """Compute symmetric quantization scale from min/max.

    Scale is defined as::

        scale = max(abs(min_val), abs(max_val)) / qmax

    If the range is zero (both min and max are 0), a fallback
    ``scale = 1.0`` is returned to avoid division by zero.
    """

    min_f = float(np.min(min_val))
    max_f = float(np.max(max_val))
    max_abs = max(abs(min_f), abs(max_f))
    if max_abs == 0.0:
        logger.warning("compute_scale: zero range detected, using scale=1.0")
        return 1.0
    logger.info("Computing quantization scale (symmetric)")
    return max_abs / float(qmax)


def compute_zero_point(
        min_val: Union[float, np.ndarray, torch.Tensor],
        max_val: Union[float, np.ndarray, torch.Tensor],
        scale: float,
        qmin: int,
        qmax: int,
        symmetric: bool = True,
) -> int:
    """Compute zero-point for symmetric or asymmetric quantization.

    For symmetric quantization, this simply returns 0. For asymmetric
    quantization, it returns a clamped version of::

        zero_point = round(-min_val / scale)
    """

    if symmetric:
        return 0
    min_f = float(np.min(min_val))
    logger.info("Computing asymmetric zero-point")
    zp = int(round(-min_f / float(scale)))
    return int(max(qmin, min(qmax, zp)))


# ---------------------------------------------------------------------------
# Float ↔ int quantization utilities
# ---------------------------------------------------------------------------


def quantize_tensor(
        tensor: torch.Tensor,
        scale: Union[float, torch.Tensor],
        qmin: int,
        qmax: int,
        zero_point: int = 0,
        dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Quantize a floating tensor into integer codes.

    Implements::

        q = round(x / scale) + zero_point
        q = clamp(q, qmin, qmax)
    """

    if isinstance(scale, torch.Tensor):
        scaled = tensor / scale
    else:
        scaled = tensor / float(scale)

    logger.info("Quantizing tensor")
    q = torch.round(scaled) + int(zero_point)
    q = torch.clamp(q, qmin, qmax)
    return q.to(dtype)


def dequantize_tensor(
        q_tensor: torch.Tensor,
        scale: Union[float, torch.Tensor],
        zero_point: int = 0,
) -> torch.Tensor:
    """Dequantize integer tensor back to floating point.

    Uses::

        x = (q - zero_point) * scale
    """

    logger.info("Dequantizing tensor")
    if isinstance(scale, torch.Tensor):
        return (q_tensor.to(torch.float32) - float(zero_point)) * scale
    return (q_tensor.to(torch.float32) - float(zero_point)) * float(scale)


# ---------------------------------------------------------------------------
# Per-channel quantization helpers
# ---------------------------------------------------------------------------


def compute_channel_min_max(
        weight: torch.Tensor,
        channel_axis: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel min and max for a weight tensor.

    Example for Conv3D weights of shape

        (out_channels, in_channels, D, H, W)

    call with ``channel_axis=0``.
    """

    if channel_axis != 0:
        weight = weight.transpose(0, channel_axis)
    flat = weight.contiguous().view(weight.shape[0], -1)
    min_vals = flat.min(dim=1).values
    max_vals = flat.max(dim=1).values
    return min_vals, max_vals


def compute_channel_scales(
        min_vals: torch.Tensor,
        max_vals: torch.Tensor,
        qmax: int,
) -> torch.Tensor:
    """Compute per-channel symmetric scales from per-channel min/max."""

    logger.info("Computing per-channel quantization scales")
    max_abs = torch.max(min_vals.abs(), max_vals.abs())
    scales = max_abs / float(qmax)
    scales[max_abs == 0] = 1.0
    return scales


# ---------------------------------------------------------------------------
# Clipping utilities
# ---------------------------------------------------------------------------


def apply_percentile_clipping(
        values: Union[torch.Tensor, np.ndarray],
        percentile: float,
) -> Union[torch.Tensor, np.ndarray]:
    """Apply symmetric percentile clipping to values.

    Steps::

    1. Compute absolute values
    2. Compute percentile threshold
    3. Clip to [-threshold, threshold]
    """

    logger.info("Applying percentile clipping at %.4f percentile", percentile)
    is_torch = isinstance(values, torch.Tensor)
    if is_torch:
        v = values.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        v = np.asarray(values, dtype=np.float32)

    abs_v = np.abs(v.reshape(-1))
    if abs_v.size == 0:
        return values
    thr = np.percentile(abs_v, percentile)
    if thr <= 0:
        return values

    if is_torch:
        return torch.clamp(values, min=-float(thr), max=float(thr))
    return np.clip(values, -float(thr), float(thr))


# ---------------------------------------------------------------------------
# MSE / OMSE error metrics
# ---------------------------------------------------------------------------


def mse_reconstruction_error(
        tensor: torch.Tensor,
        scale: Union[float, torch.Tensor],
        qmin: int,
        qmax: int,
        zero_point: int = 0,
) -> float:
    """Compute MSE between original and quantized-dequantized tensor."""

    q = quantize_tensor(tensor, scale, qmin, qmax, zero_point)
    rec = dequantize_tensor(q, scale, zero_point)
    err = torch.mean((rec - tensor) ** 2).item()
    logger.info("MSE reconstruction error: %.6f", err)
    return err


def omse_reconstruction_error(
        tensor: torch.Tensor,
        scale_candidates: Sequence[float],
        qmin: int,
        qmax: int,
        zero_point: int = 0,
) -> Tuple[float, float]:
    """Compute OMSE: choose scale from candidates that minimizes MSE.

    Returns
    -------
    (best_scale, best_error)
    """

    best_err = float("inf")
    best_scale = float(scale_candidates[0])
    for s in scale_candidates:
        err = mse_reconstruction_error(tensor, s, qmin, qmax, zero_point)
        if err < best_err:
            best_err = err
            best_scale = float(s)
    logger.info("OMSE best scale: %.6f (MSE=%.6f)", best_scale, best_err)
    return best_scale, best_err


# ---------------------------------------------------------------------------
# KL divergence utilities
# ---------------------------------------------------------------------------


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence KL(p || q) for discrete distributions."""

    eps = 1e-8
    p_safe = p + eps
    q_safe = q + eps
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def compute_kl_threshold(
        values: Union[torch.Tensor, np.ndarray],
        num_bins: int = 2048,
        num_quant_bins: int = 128,
) -> float:
    """Search for a clipping threshold that minimizes KL divergence.

    Operates on absolute values of ``values`` and returns a positive
    threshold suitable for symmetric clipping.
    """

    logger.info("Computing KL divergence threshold")
    if isinstance(values, torch.Tensor):
        v = values.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        v = np.asarray(values, dtype=np.float32)

    abs_v = np.abs(v.reshape(-1))
    if abs_v.size == 0:
        return 0.0
    max_val = float(abs_v.max())
    if max_val == 0.0:
        return 0.0

    hist, bin_edges = np.histogram(abs_v, bins=num_bins, range=(0.0, max_val))
    if not np.any(hist):
        return 0.0

    hist = hist.astype(np.float64)
    total = hist.sum()
    hist /= total

    best_kl = float("inf")
    best_threshold = max_val

    for clip_bin in range(num_quant_bins, num_bins + 1):
        p = hist.copy()
        tail_mass = p[clip_bin:].sum()
        p[clip_bin - 1] += tail_mass
        p[clip_bin:] = 0.0

        p_slice = p[:clip_bin]
        if not np.any(p_slice):
            continue

        region_size = clip_bin / float(num_quant_bins)
        q = np.zeros_like(p_slice)
        for qi in range(num_quant_bins):
            start = int(round(qi * region_size))
            end = int(round((qi + 1) * region_size))
            if end <= start:
                continue
            mass = p_slice[start:end].sum()
            if mass == 0.0:
                continue
            avg = mass / float(end - start)
            q[start:end] = avg

        kl = _kl_divergence(p_slice, q)
        if kl < best_kl:
            best_kl = kl
            best_threshold = bin_edges[clip_bin]

    return float(best_threshold)


# ---------------------------------------------------------------------------
# ACIQ threshold utility
# ---------------------------------------------------------------------------


def compute_aciq_threshold(
        values: Union[torch.Tensor, np.ndarray],
        num_bits: int = 8,
) -> float:
    """Compute ACIQ clipping threshold under Gaussian assumption.

    Assumes activations follow a zero-mean Gaussian with standard
    deviation ``sigma`` and returns ``alpha * sigma``, where ``alpha``
    depends on the bit width.
    """

    if isinstance(values, torch.Tensor):
        v = values.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        v = np.asarray(values, dtype=np.float32)

    abs_v = np.abs(v.reshape(-1))
    if abs_v.size == 0:
        return 0.0

    sigma = float(abs_v.std())
    if sigma == 0.0:
        return 0.0

    if num_bits >= 8:
        alpha = 2.575  # ~99% Gaussian mass
    elif num_bits >= 6:
        alpha = 2.0
    else:  # 4 bits and lower
        alpha = 1.3

    thr = alpha * sigma
    logger.info("ACIQ threshold (bits=%d): %.6f", num_bits, thr)
    return float(thr)
