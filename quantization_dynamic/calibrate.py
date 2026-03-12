"""Calibration utilities for post-training quantization (PTQ).

This module orchestrates activation collection (via
``ActivationObserver`` and ``CalibrationWrapper``) and converts these
statistics into per-layer quantization parameters (scale and
zero-point) using various calibration methods.

The design is tailored for 3D segmentation networks (e.g. MedNeXt,
3D U-Net) where inference typically uses sliding-window patch
inference. The caller is responsible for driving the model with the
*same* patch sampling strategy that will be used at inference time;
this module simply consumes the resulting activation statistics.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import logging

import numpy as np
import torch
import torch.nn as nn

from .activation_observer import ActivationObserver
from .model_wrappers import CalibrationWrapper
from .quant_utils import (
    SUPPORTED_CALIBRATION_METHODS,
    apply_percentile_clipping,
    compute_aciq_threshold,
    compute_kl_threshold,
    compute_scale,
    compute_zero_point,
    get_quant_range,
    normalize_calibration_method,
)


logger = logging.getLogger(__name__)


def _compute_range_minmax(layer_stats: Dict[str, object]) -> tuple[float, float]:
    """Return (min_val, max_val) using raw running min/max."""

    min_val = layer_stats.get("min")
    max_val = layer_stats.get("max")
    if min_val is None or max_val is None:
        return 0.0, 0.0
    return float(min_val), float(max_val)


def _compute_range_percentile(
        layer_stats: Dict[str, object], percentile: float
) -> tuple[float, float]:
    """Return (min_val, max_val) using symmetric percentile clipping."""

    values = layer_stats.get("values")
    if values is None or len(values) == 0:
        return 0.0, 0.0
    abs_vals = np.asarray(values, dtype=np.float32)
    thr = float(np.percentile(abs_vals, percentile))
    if thr <= 0.0:
        return -thr, thr
    return -thr, thr


def _compute_range_kl(layer_stats: Dict[str, object]) -> tuple[float, float]:
    """Return symmetric range using KL-divergence-based threshold."""

    values = layer_stats.get("values")
    if values is None or len(values) == 0:
        return 0.0, 0.0
    thr = compute_kl_threshold(values)
    return -thr, thr


def _compute_range_aciq(layer_stats: Dict[str, object], num_bits: int) -> tuple[float, float]:
    """Return symmetric range using ACIQ analytical threshold."""

    values = layer_stats.get("values")
    if values is None or len(values) == 0:
        return 0.0, 0.0
    thr = compute_aciq_threshold(values, num_bits=num_bits)
    return -thr, thr


def _select_range_for_layer(
        layer_stats: Dict[str, object],
        method: str,
        percentile: Optional[float],
        quant_dtype: str,
) -> tuple[float, float]:
    """Compute (min_val, max_val) for a single layer based on method."""

    method = normalize_calibration_method(method)
    if method == "minmax":
        return _compute_range_minmax(layer_stats)
    if method == "percentile":
        pct = 99.9 if percentile is None else float(percentile)
        return _compute_range_percentile(layer_stats, pct)
    if method == "kl":
        return _compute_range_kl(layer_stats)
    if method == "aciq":
        bits = {"int8": 8, "int6": 6, "int4": 4}.get(quant_dtype.lower(), 8)
        return _compute_range_aciq(layer_stats, bits)

    # MSE / OMSE fall back to minmax range but log intent.
    if method in {"mse", "omse"}:
        logger.warning(
            "Calibration methods 'mse' and 'omse' are not explicitly "
            "implemented for activations here; falling back to minmax range."
        )
        return _compute_range_minmax(layer_stats)

    # Should not reach here due to normalize_calibration_method
    return _compute_range_minmax(layer_stats)


def compute_quant_params_from_stats(
        activation_stats: Dict[str, Dict[str, object]],
        quant_dtype: str = "int8",
        calibration_method: str = "minmax",
        percentile: Optional[float] = None,
        symmetric: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Convert activation statistics into per-layer quantization params.

    Parameters
    ----------
    activation_stats:
        Dictionary returned by ``ActivationObserver.get_stats``.
    quant_dtype:
        Quantization dtype ("int8", "int6", or "int4").
    calibration_method:
        One of SUPPORTED_CALIBRATION_METHODS.
    percentile:
        Optional percentile for percentile-based calibration.
    symmetric:
        If True, use symmetric quantization (zero-point = 0).

    Returns
    -------
    quant_params:
        Mapping from layer name to dict with keys: "scale", "zero_point",
        "min_val", "max_val".
    """

    logger.info("Computing quantization parameters from activation statistics")
    calibration_method = normalize_calibration_method(calibration_method)
    qmin, qmax = get_quant_range(quant_dtype)
    quant_params: Dict[str, Dict[str, float]] = {}

    for layer_name, stats in activation_stats.items():
        min_val, max_val = _select_range_for_layer(
            stats, calibration_method, percentile, quant_dtype
        )
        if min_val == 0.0 and max_val == 0.0:
            # Degenerate layer (e.g., always zero); use unit scale.
            scale = 1.0
            zp = 0
        else:
            scale = compute_scale(min_val, max_val, qmax)
            zp = compute_zero_point(min_val, max_val, scale, qmin, qmax, symmetric)

        quant_params[layer_name] = {
            "scale": float(scale),
            "zero_point": float(zp),
            "min_val": float(min_val),
            "max_val": float(max_val),
        }

    return quant_params


def run_calibration(
        model: nn.Module,
        calib_loader: Iterable[torch.Tensor],
        quant_dtype: str = "int8",
        calibration_method: str = "minmax",
        percentile: Optional[float] = None,
        symmetric: bool = True,
        device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """High-level calibration workflow.

    Workflow:

    1. Wrap FP32 model with :class:`CalibrationWrapper`
    2. Register activation observers
    3. Run model on calibration dataset (using caller's patch strategy)
    4. Collect activation statistics
    5. Compute per-layer quantization parameters
    """

    logger.info("Starting calibration")
    logger.info("Calibration method: %s", calibration_method)
    if percentile is not None:
        logger.info("Using percentile clipping: %.4f", percentile)

    wrapper = CalibrationWrapper(model, ActivationObserver(), device=device)

    # Run forward passes over calibration data
    for batch in calib_loader:
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
        else:
            inputs = batch
        logger.info("Collecting activation stats on calibration batch")
        _ = wrapper(inputs)

    # Detach hooks once calibration is done
    wrapper.remove_hooks()
    activation_stats = wrapper.get_activation_stats()

    quant_params = compute_quant_params_from_stats(
        activation_stats,
        quant_dtype=quant_dtype,
        calibration_method=calibration_method,
        percentile=percentile,
        symmetric=symmetric,
    )

    logger.info("Calibration completed for %d layers", len(quant_params))
    return quant_params
