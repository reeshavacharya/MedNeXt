"""
MODEL WRAPPER MODULE FOR QUANTIZATION PIPELINE

This module provides wrapper classes around the original FP32 model
to make it compatible with the quantization pipeline.

The wrapper solves several common issues that arise when applying
post-training quantization (PTQ) to complex segmentation models such
as MedNeXt or 3D U-Net.

Responsibilities of this module include:

    • standardizing model input/output interface
    • enabling patch-based inference
    • simplifying activation observer attachment
    • ensuring ONNX export compatibility
    • preparing model for quantized inference

This module does NOT modify the architecture of the original model.
Instead it wraps the model inside helper classes.

------------------------------------------------------------
WHY MODEL WRAPPERS ARE NECESSARY
------------------------------------------------------------

Segmentation networks often include custom forward pipelines such as:

    - sliding window inference
    - patch merging
    - post-processing steps

These operations are often implemented outside the model graph.

However:

    • calibration requires running the raw forward pass
    • ONNX export requires a clean forward graph
    • TensorRT cannot export Python loops

Therefore we define wrapper classes to control how the model is executed
during each stage of the pipeline.

------------------------------------------------------------
PACKAGE IMPORTS
------------------------------------------------------------

import torch
import torch.nn as nn
import logging
import numpy as np

from quant_config import QuantConfig
from activation_observer import ActivationObserver

------------------------------------------------------------
CORE WRAPPER CLASSES
------------------------------------------------------------

This module should implement the following wrappers:

1. BaseModelWrapper
2. CalibrationWrapper
3. QuantizedModelWrapper
4. PatchInferenceWrapper

Each wrapper serves a different stage of the pipeline.

------------------------------------------------------------
1. BASE MODEL WRAPPER
------------------------------------------------------------

Class: BaseModelWrapper

This wrapper provides a consistent interface around the original model.

Constructor arguments:

    model (nn.Module)

Responsibilities:

    • store reference to original model
    • expose standardized forward() function
    • ensure model is in evaluation mode
    • optionally move model to target device

Typical behavior:

    self.model = model
    self.model.eval()

Forward function:

    def forward(self, x):
        return self.model(x)

This wrapper ensures all pipeline stages interact with the model
through the same interface.

------------------------------------------------------------
2. CALIBRATION WRAPPER
------------------------------------------------------------

Class: CalibrationWrapper

This wrapper is used during the calibration phase.

It integrates the ActivationObserver to capture activation
statistics during forward passes.

Constructor arguments:

    model
    observer

Example usage:

    observer = ActivationObserver()
    wrapper = CalibrationWrapper(model, observer)

Responsibilities:

    • register activation observers
    • run forward passes for calibration dataset
    • collect activation statistics

Hook registration:

    iterate through model modules and attach observers to:

        nn.Conv3d
        nn.Linear
        nn.BatchNorm3d

Forward pass should:

    • run original model forward
    • allow hooks to collect activations

After calibration:

    observer.get_statistics()

should return activation distributions used by calibrate.py.

------------------------------------------------------------
3. QUANTIZED MODEL WRAPPER
------------------------------------------------------------

Class: QuantizedModelWrapper

This wrapper is used after quantization has been applied.

It stores quantization parameters and ensures weights
are properly interpreted during inference.

Constructor arguments:

    model
    quant_params
    quant_dtype

Responsibilities:

    • hold quantized weights
    • store scale factors
    • perform optional dequantization during testing

This wrapper may simulate quantized inference using:

    quantize → integer ops → dequantize

However real integer inference will occur in TensorRT,
so this wrapper is mainly used for verification.

------------------------------------------------------------
4. PATCH INFERENCE WRAPPER
------------------------------------------------------------

Class: PatchInferenceWrapper

This wrapper implements sliding-window inference
for large 3D volumes.

Segmentation models often cannot process full volumes
due to GPU memory limitations.

Instead inference proceeds by:

    1. dividing the volume into patches
    2. running inference per patch
    3. stitching patch predictions together

Constructor arguments:

    model
    patch_size
    stride

Example:

    patch_size = (128,128,128)
    stride = (64,64,64)

Responsibilities:

    • divide input volume into patches
    • run model inference on each patch
    • reconstruct full output volume

------------------------------------------------------------
PATCH EXTRACTION ALGORITHM
------------------------------------------------------------

Given input volume:

    (C, D, H, W)

Compute patch coordinates using sliding window.

For each patch:

    patch = volume[:, z:z+pz, y:y+py, x:x+px]

Run model forward:

    prediction = model(patch)

Store prediction in output volume.

When overlapping patches exist:

    average predictions or apply weighting.

------------------------------------------------------------
IMPORTANT FOR QUANTIZATION
------------------------------------------------------------

Calibration must use the SAME patch sampling strategy
used during inference.

Otherwise activation statistics may be inaccurate.

Therefore PatchInferenceWrapper should optionally expose
patch generator used during calibration.

Example method:

    generate_patches(volume)

This ensures calibration and inference observe the
same data distribution.

------------------------------------------------------------
ONNX EXPORT COMPATIBILITY
------------------------------------------------------------

When exporting the model to ONNX:

    only the base model forward graph should be exported.

Patch inference logic must NOT be part of the ONNX graph,
because TensorRT cannot compile Python loops.

Therefore ONNX export should use BaseModelWrapper
instead of PatchInferenceWrapper.

------------------------------------------------------------
DEVICE MANAGEMENT
------------------------------------------------------------

Wrappers should support moving model and inputs
to correct device.

Example:

    device = torch.device("cuda")

During calibration and inference.

------------------------------------------------------------
LOGGING
------------------------------------------------------------

Provide informative logs such as:

    logging.info("Wrapping model for calibration")
    logging.info("Registering activation observers")
    logging.info("Running patch-based inference")
    logging.info("Preparing model for ONNX export")

------------------------------------------------------------
EXPECTED WORKFLOW IN PIPELINE
------------------------------------------------------------

Training (already done):

    FP32 model

Calibration:

    model → CalibrationWrapper → calibrate.py

Quantization:

    quantize.py modifies model weights

ONNX export:

    BaseModelWrapper → onnx_export.py

TensorRT build:

    tensorrt_engine.py

Inference:

    TensorRT engine → trt_inference.py
    combined with PatchInferenceWrapper

------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import logging

import numpy as np
import torch
import torch.nn as nn

from .activation_observer import ActivationObserver
from .quant_utils import (
    SUPPORTED_QUANT_DTYPES,
    compute_channel_min_max,
    compute_channel_scales,
    compute_scale,
    dequantize_tensor,
    get_quant_range,
    quantize_tensor,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. BaseModelWrapper
# ---------------------------------------------------------------------------


class BaseModelWrapper(nn.Module):
    """Thin wrapper exposing a standardized forward interface.

    This wrapper does not change the underlying architecture; it simply
    ensures the model is in evaluation mode and that inputs are moved to
    the correct device and shaped consistently (NCDHW for 3D volumes).
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        if device is not None:
            self.model.to(device)

    @property
    def device(self) -> torch.device:
        params = list(self.model.parameters())
        if params:
            return params[0].device
        return torch.device("cpu")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run a forward pass through the wrapped model.

        Accepts either (C, D, H, W) or (N, C, D, H, W) tensors and always
        returns the raw model output without post-processing.
        """

        if x.dim() == 4:
            x = x.unsqueeze(0)
        elif x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")

        x = x.to(self.device, non_blocking=True)
        logger.info("Running base model forward")
        out = self.model(x)
        return out


# ---------------------------------------------------------------------------
# 2. CalibrationWrapper
# ---------------------------------------------------------------------------


class CalibrationWrapper(BaseModelWrapper):
    """Wrapper used during calibration with ActivationObserver.

    The wrapper itself only standardizes input handling; the heavy lifting
    for statistics is done by :class:`ActivationObserver` via forward
    hooks attached to the underlying model.
    """

    def __init__(
        self,
        model: nn.Module,
        observer: Optional[ActivationObserver] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        logger.info("Wrapping model for calibration")
        super().__init__(model, device=device)
        self.observer = observer or ActivationObserver()
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_observers()

    def _register_observers(self) -> None:
        logger.info("Registering activation observers")
        self._handles = self.observer.register_hooks(self.model)

    def reset_observer(self) -> None:
        """Clear all collected statistics and re-register hooks."""

        self.observer.reset()
        self._register_observers()

    def get_activation_stats(self) -> Dict[str, Dict[str, object]]:
        """Return activation statistics collected so far."""

        return self.observer.get_stats()

    def remove_hooks(self) -> None:
        """Detach all observer hooks (no further stats will be collected)."""

        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# 3. QuantizedModelWrapper
# ---------------------------------------------------------------------------


class QuantizedModelWrapper(BaseModelWrapper):
    """Wrapper around a model whose weights have been quantized.

    This class keeps a reference to quantization parameters (typically
    per-layer activation scales) and can apply a simple weight
    quantize–dequantize step for verification in PyTorch. Real integer
    inference is expected to happen in TensorRT; this wrapper mainly
    serves for sanity-checking quantized weights and activations.
    """

    def __init__(
        self,
        model: nn.Module,
        quant_params: Dict[str, Dict[str, object]],
        quant_dtype: str = "int8",
        device: Optional[torch.device] = None,
    ) -> None:
        if quant_dtype.lower() not in SUPPORTED_QUANT_DTYPES:
            raise ValueError(
                f"Unsupported quantization dtype '{quant_dtype}'. "
                f"Supported: {SUPPORTED_QUANT_DTYPES}"
            )

        logger.info("Preparing quantized model wrapper (dtype=%s)", quant_dtype)
        super().__init__(model, device=device)
        self.quant_params = quant_params
        self.quant_dtype = quant_dtype.lower()
        self._qmin, self._qmax = get_quant_range(self.quant_dtype)

        # Optionally apply a quantize-dequantize step to weights using
        # provided per-layer scales. This does not attempt to simulate
        # exact integer arithmetic but gives a realistic approximation
        # of quantization effects.
        self._apply_weight_quantization()

    def _apply_weight_quantization(self) -> None:
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                continue
            if name not in self.quant_params:
                continue

            params = self.quant_params[name]
            scale = params.get("scale")
            if scale is None:
                continue

            w = getattr(module, "weight", None)
            if w is None:
                continue

            if isinstance(scale, torch.Tensor):
                w_scale = scale.to(w.device)
            else:
                w_scale = torch.tensor(scale, device=w.device, dtype=torch.float32)

            logger.info("Quantizing weights for layer '%s'", name)
            q_w = quantize_tensor(w.data, w_scale, self._qmin, self._qmax, zero_point=0, dtype=torch.int8)
            w_deq = dequantize_tensor(q_w, w_scale, zero_point=0)
            w.data.copy_(w_deq)
            # Keep integer weights for debugging/analysis
            module.register_buffer("weight_qint", q_w.cpu())
            module.register_buffer("weight_scale", w_scale.detach().cpu())


# ---------------------------------------------------------------------------
# 4. PatchInferenceWrapper
# ---------------------------------------------------------------------------


class PatchInferenceWrapper(BaseModelWrapper):
    """Sliding-window (patch-based) inference for large 3D volumes.

    This wrapper operates on 3D volumes of shape (C, D, H, W) or
    (N, C, D, H, W), extracts overlapping patches, runs the model on
    each patch, and stitches the predictions back into a full-resolution
    output volume by averaging overlapping regions.
    """

    def __init__(
        self,
        model: nn.Module,
        patch_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        device: Optional[torch.device] = None,
    ) -> None:
        logger.info("Initializing patch-based inference wrapper")
        super().__init__(model, device=device)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.stride = tuple(int(x) for x in stride)

    @staticmethod
    def _compute_starts(dim: int, patch: int, stride: int) -> List[int]:
        if dim <= patch:
            return [0]
        starts = list(range(0, dim - patch + 1, stride))
        if starts[-1] != dim - patch:
            starts.append(dim - patch)
        return starts

    def generate_patches(self, volume: torch.Tensor) -> Iterable[Tuple[torch.Tensor, Tuple[int, int, int]]]:
        """Yield (patch, (z, y, x)) tuples for a given volume.

        Volume should have shape (C, D, H, W). The returned patches are
        4D tensors compatible with the base model wrapper (a batch
        dimension will be added in ``forward``).
        """

        if volume.dim() != 4:
            raise ValueError(f"Expected 4D tensor (C,D,H,W), got {tuple(volume.shape)}")

        _, D, H, W = volume.shape
        pz, py, px = self.patch_size
        sz, sy, sx = self.stride

        z_starts = self._compute_starts(D, pz, sz)
        y_starts = self._compute_starts(H, py, sy)
        x_starts = self._compute_starts(W, px, sx)

        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = volume[:, z : z + pz, y : y + py, x : x + px]
                    yield patch, (z, y, x)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run sliding-window inference over a 3D volume.

        Parameters
        ----------
        x:
            Input tensor of shape (C, D, H, W) or (N, C, D, H, W).

        Returns
        -------
        Tensor of shape (C_out, D, H, W) with averaged predictions.
        """

        logger.info("Running patch-based inference")
        added_batch = False
        if x.dim() == 4:
            # (C,D,H,W) -> (1,C,D,H,W)
            x = x.unsqueeze(0)
            added_batch = True
        elif x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")

        # We operate on the first (or only) batch element for patching.
        B, C, D, H, W = x.shape
        if B != 1:
            logger.warning("PatchInferenceWrapper currently assumes batch size 1; using first element only.")
        volume = x[0]

        pz, py, px = self.patch_size
        device = self.device

        # Prepare output and count volumes lazily after first patch
        output: Optional[torch.Tensor] = None
        count: Optional[torch.Tensor] = None

        for patch, (z, y, xx) in self.generate_patches(volume):
            patch_in = patch.unsqueeze(0).to(device, non_blocking=True)
            pred = self.model(patch_in)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            if pred.dim() == 5:
                # (N,C,D,H,W) -> (C,D,H,W)
                pred = pred[0]

            if output is None:
                C_out = pred.shape[0]
                output = torch.zeros((C_out, D, H, W), dtype=pred.dtype, device=device)
                count = torch.zeros((1, D, H, W), dtype=torch.float32, device=device)

            output[:, z : z + pz, y : y + py, xx : xx + px] += pred
            count[:, z : z + pz, y : y + py, xx : xx + px] += 1.0

        if output is None or count is None:
            raise RuntimeError("No patches were generated for the given input volume")

        output = output / count.clamp(min=1.0)
        return output
