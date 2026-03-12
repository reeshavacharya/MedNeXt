"""TensorRT inference module.

This module loads a serialized TensorRT engine and performs inference
using optimized GPU kernels. It also provides a simple sliding-window
helper for 3D medical image segmentation.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import logging
import os

import numpy as np
import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda


logger = logging.getLogger(__name__)


def _trt_dtype_to_np(trt_dtype: trt.DataType) -> np.dtype:
    """Map TensorRT dtypes to NumPy dtypes."""
    if trt_dtype == trt.DataType.FLOAT:
        return np.float32
    if trt_dtype == trt.DataType.HALF:
        return np.float16
    if trt_dtype == trt.DataType.INT32:
        return np.int32
    if trt_dtype == trt.DataType.INT8:
        return np.int8
    if trt_dtype == trt.DataType.BOOL:
        return np.bool_
    raise TypeError(f"Unsupported TensorRT data type: {trt_dtype}")


class TensorRTInferenceSession:
    """Convenience wrapper around a TensorRT engine for inference.

    This class:

    - loads a serialized engine from disk
    - creates a runtime and execution context
    - manages GPU buffers for inputs and outputs
    - exposes methods for single-pass and sliding-window inference
    """

    def __init__(self, engine_path: str) -> None:
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        logger.info("Loading TensorRT engine from %s", engine_path)
        logger_trt = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger_trt)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        # Assume a single input and single output binding.
        self.input_binding, self.output_binding = self._find_io_bindings()
        self.stream = cuda.Stream()

    def _find_io_bindings(self) -> Tuple[int, int]:
        input_indices: List[int] = []
        output_indices: List[int] = []
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                input_indices.append(i)
            else:
                output_indices.append(i)

        if len(input_indices) != 1 or len(output_indices) != 1:
            raise RuntimeError(
                "TensorRTInferenceSession expects exactly one input and one output binding"
            )
        return input_indices[0], output_indices[0]

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        """Run a single inference pass.

        Parameters
        ----------
        input_array:
            Input tensor as a NumPy array. Shape must be compatible with the
            engine's input binding. Typically this is ``(N, C, D, H, W)`` for
            3D segmentation models exported with explicit batch.

        Returns
        -------
        output_array:
            Model predictions as a NumPy array.
        """

        if not isinstance(input_array, np.ndarray):
            raise TypeError("input_array must be a NumPy array")

        # Ensure contiguous float32 input by default.
        if input_array.dtype != np.float32:
            input_array = input_array.astype(np.float32, copy=False)

        input_array = np.ascontiguousarray(input_array)

        input_idx = self.input_binding
        output_idx = self.output_binding

        # Handle dynamic shapes by setting the binding shape from input.
        if -1 in tuple(self.engine.get_binding_shape(input_idx)):
            self.context.set_binding_shape(input_idx, tuple(input_array.shape))

        input_nbytes = int(input_array.nbytes)

        output_shape = tuple(self.context.get_binding_shape(output_idx))
        output_dtype = _trt_dtype_to_np(self.engine.get_binding_dtype(output_idx))
        output_size = int(np.prod(output_shape))
        output_array = np.empty(output_size, dtype=output_dtype)

        output_nbytes = int(output_array.nbytes)

        # Allocate device memory.
        device_input = cuda.mem_alloc(input_nbytes)
        device_output = cuda.mem_alloc(output_nbytes)

        bindings: List[int] = [0] * self.engine.num_bindings
        bindings[input_idx] = int(device_input)
        bindings[output_idx] = int(device_output)

        # Transfer input to device.
        cuda.memcpy_htod_async(device_input, input_array, self.stream)

        # Execute inference.
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        # Transfer predictions back.
        cuda.memcpy_dtoh_async(output_array, device_output, self.stream)
        self.stream.synchronize()

        # Free device memory.
        device_input.free()
        device_output.free()

        return output_array.reshape(output_shape)

    def sliding_window_inference(
        self,
        volume: np.ndarray,
        patch_size: Sequence[int],
        stride: Sequence[int],
    ) -> np.ndarray:
        """Run sliding-window inference over a 3D volume.

        Parameters
        ----------
        volume:
            Input volume as ``(C, D, H, W)`` or ``(1, C, D, H, W)`` NumPy array.
        patch_size:
            Tuple/list of three ints ``(pD, pH, pW)``.
        stride:
            Tuple/list of three ints ``(sD, sH, sW)``.

        Returns
        -------
        pred:
            Output prediction volume as ``(C_out, D, H, W)``.
        """

        if volume.ndim == 5:
            # Assume (N, C, D, H, W) with N == 1
            if volume.shape[0] != 1:
                raise ValueError("sliding_window_inference expects batch size 1 when 5D input is given")
            volume = volume[0]
        elif volume.ndim != 4:
            raise ValueError("volume must have shape (C, D, H, W) or (1, C, D, H, W)")

        volume = np.asarray(volume, dtype=np.float32)

        C, D, H, W = volume.shape
        pD, pH, pW = patch_size
        sD, sH, sW = stride

        # Determine output channel dimension from a single forward pass.
        # Use a central patch (with padding via clamping of start indices if needed).
        start_D = max(0, (D - pD) // 2)
        start_H = max(0, (H - pH) // 2)
        start_W = max(0, (W - pW) // 2)
        patch = volume[
            :,
            start_D : start_D + pD,
            start_H : start_H + pH,
            start_W : start_W + pW,
        ]
        patch = patch[None, ...]  # add batch dim
        probe_out = self.infer(patch)
        if probe_out.ndim != 5:
            raise RuntimeError("TensorRT engine output must have shape (N, C_out, D, H, W)")
        C_out = probe_out.shape[1]

        pred_sum = np.zeros((C_out, D, H, W), dtype=np.float32)
        pred_count = np.zeros((1, D, H, W), dtype=np.float32)

        for z in range(0, max(D - pD + 1, 1), sD):
            z_end = min(z + pD, D)
            z = z_end - pD
            for y in range(0, max(H - pH + 1, 1), sH):
                y_end = min(y + pH, H)
                y = y_end - pH
                for x in range(0, max(W - pW + 1, 1), sW):
                    x_end = min(x + pW, W)
                    x = x_end - pW

                    patch = volume[:, z:z_end, y:y_end, x:x_end][None, ...]
                    out_patch = self.infer(patch)  # (1, C_out, pD, pH, pW)
                    out_patch = out_patch[0]

                    pred_sum[:, z:z_end, y:y_end, x:x_end] += out_patch
                    pred_count[:, z:z_end, y:y_end, x:x_end] += 1.0

        # Avoid division by zero just in case.
        pred_count[pred_count == 0] = 1.0
        pred = pred_sum / pred_count

        return pred
