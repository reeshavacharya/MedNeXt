"""Calibration script for INT8 quantization.

Job: Determine the dynamic range (min/max) of activations for the
trained MedNeXt model on Task017 (BTCV) and derive per-layer
quantization parameters (scale S and zero-point Z).

Because we aren't retraining, we estimate ranges from a calibration
subset of the (already preprocessed) training data.

High-level steps (matching the comments in this file):

- Input: Saved .model (final checkpoint) and a calibration subset of
  the training data (here: preprocessed volumes in
  nnUNet_preprocessed/Task017_AbdominalOrganSegmentation/..._stage1).
- Process:
	* Load the trained model (MedNeXt-S, 3d_fullres, fold 0).
	* Register forward hooks on every Conv and Linear layer.
	* Run several forward passes on real preprocessed data.
	* For each such layer, track the global min/max of its outputs.
- Output: quantization/scales.json containing, per layer:
	* observed min / max activation
	* scale S and zero-point Z for symmetric int8 quantization.

Run with:

	cd /home/r/reeshav/MedNeXt
	source .venv/bin/activate
	python -m quantization.calibrate

Environment variables nnUNet_raw_data_base, nnUNet_preprocessed and
RESULTS_FOLDER must be set the same way they were during training so
that paths.py resolves correctly.
"""

import json
import os
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from batchgenerators.utilities.file_and_folder_operations import (
	join,
	subfiles,
)

from nnunet_mednext.paths import (
	preprocessing_output_dir,
	network_training_output_dir,
)
from nnunet_mednext.training.model_restore import (
	load_model_and_checkpoint_files,
)


def _get_model_folder() -> str:
	"""Return the training output folder for Task017 MedNeXt-S (fold 0).

	This mirrors where training stored model_final_checkpoint.model.
	"""

	if network_training_output_dir is None:
		raise RuntimeError(
			"network_training_output_dir is None. Make sure RESULTS_FOLDER "
			"is set before running calibration."
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


def _get_calibration_files(max_cases: int = 50):
	"""Return a list of preprocessed training cases for calibration.

	We use the 3D full-resolution preprocessed data (stage1) for
	Task017, which matches the data seen during training.
	"""

	if preprocessing_output_dir is None:
		raise RuntimeError(
			"preprocessing_output_dir is None. Make sure nnUNet_preprocessed "
			"is set before running calibration."
		)

	task_dir = join(
		preprocessing_output_dir,
		"Task017_AbdominalOrganSegmentation",
		"nnUNetData_plans_v2.1_trgSp_1x1x1_stage1",
	)
	if not os.path.isdir(task_dir):
		raise RuntimeError(
			f"Preprocessed Task017 data not found at {task_dir}. "
			f"Run mednextv1_plan_and_preprocess first."
		)

	files = subfiles(task_dir, suffix=".npz", join=True, sort=True)
	if not files:
		raise RuntimeError(
			f"No .npz files found in preprocessed Task017 directory: {task_dir}"
		)

	return files[: max_cases or len(files)]


def _register_activation_hooks(
	model: nn.Module,
) -> (Dict[str, Dict[str, float]], list):
	"""Register forward hooks on all Conv/Linear layers.

	Returns a (stats_dict, hooks) tuple where stats_dict[layer_name]
	accumulates global min/max over all calibration forwards.
	"""

	activation_stats: Dict[str, Dict[str, float]] = {}
	hooks = []

	def make_hook(name: str):
		def hook(_module, _inp, out):
			# out can be Tensor or tuple/list; handle the common cases.
			if isinstance(out, (tuple, list)):
				if not out:
					return
				out = out[0]
			if not isinstance(out, torch.Tensor):
				return
			if out.numel() == 0:
				return

			with torch.no_grad():
				data = out.detach()
				cur_min = float(data.min().item())
				cur_max = float(data.max().item())

			stats = activation_stats.get(name)
			if stats is None:
				activation_stats[name] = {"min": cur_min, "max": cur_max}
			else:
				stats["min"] = min(stats["min"], cur_min)
				stats["max"] = max(stats["max"], cur_max)

		return hook

	for name, module in model.named_modules():
		if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
			h = module.register_forward_hook(make_hook(name))
			hooks.append(h)

	return activation_stats, hooks


def _compute_scales(activation_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
	"""Given per-layer min/max, compute int8 (symmetric) scale & zero-point.

	We assume signed INT8 with range [-128, 127] and symmetric
	quantization around 0 (Z = 0). For each layer:

		max_abs = max(|min|, |max|)
		S = max_abs / 127
		Z = 0

	If max_abs is 0 (e.g. layer output is constant 0), we fall back to
	S = 1.0, Z = 0 to avoid division by zero.
	"""

	qmin, qmax = -128, 127
	layers: Dict[str, Any] = {}

	for name, stats in activation_stats.items():
		vmin = float(stats["min"])
		vmax = float(stats["max"])
		max_abs = max(abs(vmin), abs(vmax))
		if max_abs == 0.0:
			scale = 1.0
		else:
			scale = max_abs / float(qmax)
		zero_point = 0
		layers[name] = {
			"min": vmin,
			"max": vmax,
			"scale": scale,
			"zero_point": zero_point,
		}

	return {
		"meta": {
			"dtype": "int8",
			"scheme": "symmetric",
			"qmin": qmin,
			"qmax": qmax,
			"note": "S, Z computed per-layer from observed activation min/max",
		},
		"layers": layers,
	}


def main(max_cases: int = 50):
	# 1) Locate model and calibration data
	model_folder = _get_model_folder()
	calib_files = _get_calibration_files(max_cases=max_cases)

	# 2) Load trainer and final checkpoint in FP32 (no mixed precision)
	print(f"[calibrate] Loading model from: {model_folder}", flush=True)
	trainer, params = load_model_and_checkpoint_files(
		model_folder,
		folds=[0],
		mixed_precision=False,
		checkpoint_name="model_final_checkpoint",
	)
	if not params:
		raise RuntimeError("No checkpoint parameters returned by load_model_and_checkpoint_files")

	# Load the (only) fold parameters into the trainer network
	trainer.load_checkpoint_ram(params[0], False)

	net = trainer.network
	net.eval()

	# Determine how many input channels the network actually expects
	# (for MedNeXt this is the stem conv in_channels). Some
	# preprocessed arrays may have extra channels (for example from
	# auxiliary information), so we will slice the calibration data to
	# match this count.
	stem_in_ch = getattr(getattr(net, "stem", None), "in_channels", None)
	if stem_in_ch is None:
		# Fallback: assume first dimension of data is the correct
		# channel count (no slicing). This should not happen for
		# MedNeXt but keeps the script robust.
		print("[calibrate] Could not determine stem in_channels; using data channels as-is", flush=True)
		stem_in_ch = None
	else:
		print(f"[calibrate] Network stem expects {stem_in_ch} input channel(s)", flush=True)

	# 3) Register activation hooks on Conv/Linear layers
	activation_stats, hooks = _register_activation_hooks(net)
	print(f"[calibrate] Registered hooks on {len(activation_stats)} layer(s)", flush=True)

	# 4) Run forward passes on calibration volumes using the trainer's
	#    prediction function so that shapes/modalities are handled
	#    exactly as during training/inference.
	print(f"[calibrate] Running calibration on {len(calib_files)} preprocessed cases...", flush=True)
	with torch.no_grad():
		for idx, f in enumerate(calib_files, start=1):
			data = np.load(f)["data"]  # same format used by nnUNet inference
			# nnUNet expects 'data' as a numpy array of shape
			# (C, Z, Y, X). If this array has more channels than the
			# network input (for example, extra auxiliary channels),
			# slice to the number of channels the stem expects so that
			# we mirror the configuration used during training.
			if stem_in_ch is not None and data.shape[0] > stem_in_ch:
				print(
					f"[calibrate] Slicing data channels from {data.shape[0]} to {stem_in_ch} "
					f"for file {os.path.basename(f)}",
					flush=True,
				)
				data = data[:stem_in_ch]
			elif stem_in_ch is not None and data.shape[0] < stem_in_ch:
				raise RuntimeError(
					f"Calibration data has fewer channels ({data.shape[0]}) "
					f"than the network expects ({stem_in_ch})."
				)

			_ = trainer.predict_preprocessed_data_return_seg_and_softmax(
				data,
				do_mirroring=False,
				mirror_axes=trainer.data_aug_params.get("mirror_axes", (0, 1, 2)),
				use_sliding_window=True,
				step_size=0.5,
				use_gaussian=True,
				all_in_gpu=False,
				mixed_precision=False,
			)
			print(f"[calibrate] Processed {idx}/{len(calib_files)}: {os.path.basename(f)}", flush=True)

	# Remove hooks to avoid potential reference cycles
	for h in hooks:
		h.remove()

	# 5) Derive per-layer scales and zero-points
	result = _compute_scales(activation_stats)
	print(f"[calibrate] Computed scales for {len(result['layers'])} layer(s)", flush=True)

	# 6) Save to quantization/scales.json (overwrite on each run)
	this_dir = os.path.dirname(os.path.abspath(__file__))
	out_path = os.path.join(this_dir, "scales.json")
	with open(out_path, "w") as f:
		json.dump(result, f, indent=2)

	print(f"[calibrate] Saved calibration scales to: {out_path}", flush=True)


if __name__ == "__main__":
	# Default: use up to 50 calibration cases. Adjust if desired.
	main(max_cases=50)
