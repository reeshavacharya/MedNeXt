"""ONNX export utilities for the quantization pipeline.

This module exports the (quantized) PyTorch model to ONNX format for
subsequent TensorRT compilation. The actual model architecture should
be wrapped using :class:`BaseModelWrapper` before export so that only
the raw forward graph (without Python-level patch loops) is captured.
"""

from __future__ import annotations

from typing import Optional, Tuple

import logging

import torch
import torch.nn as nn

from .model_wrappers import BaseModelWrapper
from .quant_utils import OnnxExportConfig, default_onnx_export_config


logger = logging.getLogger(__name__)


def export_quantized_onnx(
        model: nn.Module,
        output_path: str = "quantized_model.onnx",
        dummy_input_shape: Tuple[int, int, int, int, int] = (1, 1, 128, 128, 128),
        config: Optional[OnnxExportConfig] = None,
) -> None:
    """Export (quantized) model to ONNX file.

    Parameters
    ----------
    model:
        Model to export. Typically a :class:`QuantizedModelWrapper` or the
        underlying nn.Module. It will be wrapped in :class:`BaseModelWrapper`
        for export.
    output_path:
        Destination ONNX path. Defaults to ``quantized_model.onnx``.
    dummy_input_shape:
        Shape of the dummy input tensor for tracing, typically
        ``(N, C, D, H, W)`` for 3D segmentation.
    config:
        Optional :class:`OnnxExportConfig` instance. If ``None``, the
        default configuration from :func:`default_onnx_export_config` is
        used.
    """

    if config is None:
        config = default_onnx_export_config()

    logger.info("Preparing model for ONNX export to %s", output_path)
    # Ensure a clean forward graph without patch-based loops.
    if not isinstance(model, BaseModelWrapper):
        wrapped = BaseModelWrapper(model)
    else:
        wrapped = model

    wrapped.eval()
    params = list(wrapped.parameters())
    device = params[0].device if params else torch.device("cpu")
    wrapped.to(device)

    dummy_input = torch.randn(*dummy_input_shape, device=device)

    logger.info("Exporting model to ONNX (opset=%d)", config.opset_version)
    torch.onnx.export(
        wrapped,
        dummy_input,
        output_path,
        opset_version=config.opset_version,
        do_constant_folding=config.do_constant_folding,
        input_names=list(config.input_names),
        output_names=list(config.output_names),
        dynamic_axes=config.dynamic_axes,
    )

    logger.info("ONNX model saved to %s", output_path)
