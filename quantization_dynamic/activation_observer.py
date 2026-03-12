"""Activation observer for calibration.

This module implements an ``ActivationObserver`` that attaches forward
hooks to selected layers and collects activation statistics for
post-training quantization. It is deliberately lightweight and only
stores aggregate statistics and histograms to control memory usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import logging

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class LayerStats:
    """Container for per-layer activation statistics.

    Attributes
    ----------
    min_val, max_val:
        Running min/max over all observed activations.
    abs_values:
        Concatenated absolute values used for percentile estimation.
    hist:
        Global histogram counts over |activation|.
    hist_range:
        Tuple (min_abs, max_abs) used to define histogram range.
    """

    min_val: float = float("inf")
    max_val: float = float("-inf")
    abs_values: List[np.ndarray] = field(default_factory=list)
    hist: Optional[np.ndarray] = None
    hist_range: Optional[Tuple[float, float]] = None


class ActivationObserver:
    """Collect activation statistics via forward hooks during calibration.

    Usage pattern::

        observer = ActivationObserver(num_bins=2048)
        handles = observer.register_hooks(model)
        # run calibration forward passes
        ...
        for h in handles:
            h.remove()
        stats = observer.get_stats()

    The returned ``stats`` dictionary can then be consumed by
    calibration code to derive quantization scales and thresholds.
    """

    def __init__(self, num_bins: int = 2048) -> None:
        self.num_bins = int(num_bins)
        self.activation_stats: Dict[str, LayerStats] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def reset(self) -> None:
        """Clear all collected statistics and remove existing hooks."""

        logger.info("Resetting activation observer statistics")
        self.activation_stats.clear()
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _get_or_create_stats(self, name: str) -> LayerStats:
        if name not in self.activation_stats:
            self.activation_stats[name] = LayerStats()
        return self.activation_stats[name]

    def _hook_fn(self, name: str):
        """Create a forward hook function bound to a layer name."""

        def hook(module: nn.Module, inp, out):  # noqa: D401
            # Collect activation statistics from layer output.
            logger.debug("Collecting activation statistics for layer '%s'", name)
            if isinstance(out, (tuple, list)):
                if not out:
                    return
                act = out[0]
            else:
                act = out

            if not isinstance(act, torch.Tensor):
                return

            with torch.no_grad():
                act_detached = act.detach()
                if act_detached.numel() == 0:
                    return

                # Move to CPU, flatten, and convert to numpy
                arr = act_detached.cpu().numpy().astype(np.float32, copy=False)
                flat = arr.reshape(-1)
                stats = self._get_or_create_stats(name)

                # Update running min/max
                cur_min = float(np.min(flat))
                cur_max = float(np.max(flat))
                stats.min_val = min(stats.min_val, cur_min)
                stats.max_val = max(stats.max_val, cur_max)

                # Percentile tracking uses absolute values
                stats.abs_values.append(np.abs(flat))

                # Histogram accumulation over |activation|
                abs_flat = np.abs(flat)
                max_abs = float(abs_flat.max())
                if max_abs == 0.0:
                    return

                if stats.hist is None:
                    # Initialize histogram range lazily based on first batch
                    stats.hist_range = (0.0, max_abs)
                    hist, _ = np.histogram(
                        abs_flat,
                        bins=self.num_bins,
                        range=stats.hist_range,
                    )
                    stats.hist = hist.astype(np.int64)
                else:
                    # Recompute histogram using existing range
                    min_edge, max_edge = stats.hist_range
                    # If new activations exceed current range, extend range and
                    # rebuild histogram from stored abs_values at next query.
                    if max_abs > max_edge:
                        stats.hist_range = (0.0, max_abs)
                        stats.hist = None
                    else:
                        hist, _ = np.histogram(
                            abs_flat,
                            bins=self.num_bins,
                            range=stats.hist_range,
                        )
                        stats.hist += hist.astype(np.int64)

        return hook

    def register_hooks(self, model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
        """Attach forward hooks to common quantizable layers in ``model``.

        Currently observes outputs of:

        - nn.Conv1d/2d/3d
        - nn.Linear
        - nn.BatchNorm1d/2d/3d
        - nn.ReLU / nn.LeakyReLU (useful for activation ranges)
        """

        logger.info("Registering activation observers")
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(
                module,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.Linear,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.ReLU,
                    nn.LeakyReLU,
                ),
            ):
                h = module.register_forward_hook(self._hook_fn(name))
                self._handles.append(h)

        return self._handles

    def get_stats(self) -> Dict[str, Dict[str, object]]:
        """Return collected activation statistics in a consumable format.

        The structure is::

            {
                layer_name: {
                    "values": np.ndarray (concatenated activations, optional),
                    "histogram": np.ndarray or None,
                    "hist_range": (min_abs, max_abs) or None,
                    "min": float,
                    "max": float,
                },
                ...
            }
        """

        logger.info("Calibration pass completed. Collecting activation statistics")
        result: Dict[str, Dict[str, object]] = {}
        for name, stats in self.activation_stats.items():
            if stats.abs_values:
                all_abs = np.concatenate(stats.abs_values, axis=0)
            else:
                all_abs = np.empty(0, dtype=np.float32)

            # If histogram needs to be rebuilt because range changed
            if stats.hist is None and all_abs.size > 0 and stats.hist_range is not None:
                min_edge, max_edge = stats.hist_range
                hist, _ = np.histogram(
                    all_abs,
                    bins=self.num_bins,
                    range=(0.0, max_edge),
                )
                stats.hist = hist.astype(np.int64)

            result[name] = {
                "values": all_abs,
                "histogram": stats.hist,
                "hist_range": stats.hist_range,
                "min": None if stats.min_val == float("inf") else stats.min_val,
                "max": None if stats.max_val == float("-inf") else stats.max_val,
            }

        return result
