"""TensorRT engine builder.

This module converts an ONNX model into a TensorRT engine that can
execute optimized INT8 inference on NVIDIA GPUs. It is a thin wrapper
around the TensorRT API and is configured via :class:`TensorRTConfig`
from :mod:`quant_utils`.
"""

from __future__ import annotations

from typing import Optional

import logging
import os

import tensorrt as trt

from .quant_utils import TensorRTConfig, default_tensorrt_config


logger = logging.getLogger(__name__)


def build_tensorrt_engine(
        onnx_path: str,
        engine_path: str = "model.engine",
        config: Optional[TensorRTConfig] = None,
) -> None:
    """Build and serialize a TensorRT engine from an ONNX model.

    Parameters
    ----------
    onnx_path:
        Path to the input ONNX model.
    engine_path:
        Path where the serialized TensorRT engine will be saved.
    config:
        Optional :class:`TensorRTConfig` instance controlling INT8/FP16
        flags and workspace size. If ``None``,
        :func:`default_tensorrt_config` is used.
    """

    if config is None:
        config = default_tensorrt_config()

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    logger.info("Building TensorRT engine from ONNX: %s", onnx_path)
    logger.info("Engine output path: %s", engine_path)

    logger_trt = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger_trt)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger_trt)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("TensorRT ONNX parser error: %s", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model for TensorRT")

    build_config = builder.create_builder_config()
    build_config.max_workspace_size = int(config.max_workspace_size_bytes)

    if config.int8:
        logger.info("Enabling INT8 mode for TensorRT engine")
        build_config.set_flag(trt.BuilderFlag.INT8)
    if config.fp16:
        logger.info("Enabling FP16 mode for TensorRT engine")
        build_config.set_flag(trt.BuilderFlag.FP16)
    if config.strict_types:
        logger.info("Enabling strict type constraints for TensorRT engine")
        build_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    engine = builder.build_engine(network, build_config)
    if engine is None:
        raise RuntimeError("TensorRT builder failed to create an engine")

    logger.info("Serializing TensorRT engine to %s", engine_path)
    serialized = engine.serialize()
    with open(engine_path, "wb") as f:
        f.write(serialized)

    logger.info("TensorRT engine successfully saved")
