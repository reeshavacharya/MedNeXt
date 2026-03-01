"""Fake-quantized model wrapper using precomputed activation scales.

Job: Define a "fake quantized" version of the trained MedNeXt model.

This module provides a wrapper that applies uniform fake quantization to
the outputs of Conv/Linear layers using the activation ranges computed
by quantization/calibrate.py (stored in quantization/scales.json).

Key idea: we do not actually run INT8 tensors; instead, for a given
layer output x and scale S, we apply the operation

	q = clamp(round(x / S), qmin, qmax)
	x_fake = q * S

This constrains the activations to the same 256 discrete steps (for
INT8) that a true quantized pipeline would use, mimicking quantization
error while still running in FP32.

Usage (high-level):

	from quantization.quantized_model import load_fake_quantized_trainer

	trainer = load_fake_quantized_trainer()
	# Use trainer.predict_preprocessed_data_return_seg_and_softmax(...) as
	# usual; the internal network now applies fake quantization at each
	# Conv/Linear layer.
"""

import json
import os
from typing import Dict, Any

import torch
import torch.nn as nn

from batchgenerators.utilities.file_and_folder_operations import join

from nnunet_mednext.paths import network_training_output_dir
from nnunet_mednext.training.model_restore import load_model_and_checkpoint_files


class FakeQuantizeActivation(nn.Module):

	"""Layer that applies per-tensor fake quantization with given S, Z.

	It assumes symmetric signed INT8 by default (qmin=-128, qmax=127),
	and uses the calibration-derived scale S and zero-point Z.
	"""

	def __init__(self, scale: float, zero_point: int = 0,
				 qmin: int = -128, qmax: int = 127):
		super().__init__()
		self.register_buffer("scale", torch.tensor(float(scale)))
		self.register_buffer("zero_point", torch.tensor(int(zero_point)))
		self.qmin = int(qmin)
		self.qmax = int(qmax)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if not self.training and self.scale.item() > 0.0:
			s = self.scale
			z = self.zero_point
			q = torch.round(x / s + z)
			q = torch.clamp(q, self.qmin, self.qmax)
			x = (q - z) * s
		return x


def _load_scales() -> Dict[str, Any]:
	"""Load quantization/scales.json produced by calibrate.py."""

	this_dir = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(this_dir, "scales.json")
	if not os.path.isfile(path):
		raise FileNotFoundError(
			f"scales.json not found at {path}. Run quantization.calibrate first."
		)
	with open(path, "r") as f:
		data = json.load(f)
	return data


def _wrap_model_with_fake_quant(net: nn.Module, scales: Dict[str, Any]) -> nn.Module:
	"""Insert FakeQuantizeActivation modules after Conv/Linear layers.

	For each named module that is Conv or Linear and appears in
	scales["layers"], we wrap it in a small Sequential that applies the
	original module followed by FakeQuantizeActivation using that
	layer's scale/zero-point.
	"""

	layer_scales = scales.get("layers", {})

	for name, module in list(net.named_modules()):
		if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
			continue

		if name not in layer_scales:
			continue

		info = layer_scales[name]
		scale = float(info.get("scale", 1.0))
		zero_point = int(info.get("zero_point", 0))

		parent = net
		parts = name.split(".")
		for p in parts[:-1]:
			parent = getattr(parent, p)

		last = parts[-1]
		orig = getattr(parent, last)

		fq = FakeQuantizeActivation(scale=scale, zero_point=zero_point)
		wrapped = nn.Sequential(orig, fq)
		setattr(parent, last, wrapped)

	return net


def _quantize_module_weights(net: nn.Module,
							   qmin: int = -128,
							   qmax: int = 127) -> None:
	"""Fake-quantize Conv/Linear weights in-place.

	This simulates INT8 weight quantization by:

	1. Computing a per-tensor symmetric scale S_w for each Conv/Linear
	   weight tensor based on its max absolute value.
	2. Quantizing to integers q_w = round(w / S_w) clamped to [qmin, qmax].
	3. Dequantizing back to FP32: w_fake = q_w * S_w.

	The network still runs in FP32, but its weights are constrained to the
	values that an INT8 representation (with per-tensor scaling) could
	represent.
	"""

	for module in net.modules():
		if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
			continue

		w = getattr(module, "weight", None)
		if w is None:
			continue

		with torch.no_grad():
			max_abs = w.data.abs().max()
			if max_abs == 0:
				continue

			scale = float(max_abs) / float(qmax)
			q_w = torch.round(w.data / scale)
			q_w = torch.clamp(q_w, qmin, qmax)
			w.data.copy_(q_w * scale)


def load_fake_quantized_trainer():
	"""Load the Task017 MedNeXt-S trainer with fake-quantized network.

	Returns the nnUNet trainer instance whose network has been modified
	in-place to insert FakeQuantizeActivation modules according to
	quantization/scales.json.
	"""

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

	print(f"[quantized_model] Loading FP32 model from {model_folder}")
	trainer, params = load_model_and_checkpoint_files(
		model_folder,
		folds=[0],
		mixed_precision=False,
		checkpoint_name="model_final_checkpoint",
	)
	if not params:
		raise RuntimeError("No checkpoint parameters returned by load_model_and_checkpoint_files")

	trainer.load_checkpoint_ram(params[0], False)

	# Fake-quantize Conv/Linear weights in-place (per-tensor symmetric INT8)
	net = trainer.network
	_quantize_module_weights(net)

	# Load calibration scales
	scales = _load_scales()
	print("[quantized_model] Loaded scales for",
		  len(scales.get("layers", {})), "layer(s)")

	# Wrap network in-place with activation fake quant
	_wrap_model_with_fake_quant(net, scales)
	net.eval()

	return trainer
