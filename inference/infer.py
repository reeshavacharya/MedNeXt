"""Single-case inference and metric evaluation on BTCV (Task017).

This script:

1. Locates the trained MedNeXt-S nnUNet model for Task017
   (using the standard nnunet_mednext.paths configuration).
2. Picks **one random training case** from the BTCV imagesTr/labelsTr
   folders.
3. Runs nnUNet-style inference for that case to generate a segmentation
   NIfTI.
4. Computes Dice, Accuracy, and other default metrics using
   :class:`nnunet_mednext.evaluation.evaluator.Evaluator`.
5. Writes the metrics to a JSON file named
   ``inference_{timestamp}.json`` in this ``inference/`` folder.

Usage (from project root, with nnUNet env vars set):

    python -m inference.infer

Environment variables required (as usual for nnUNet/MedNeXt):

    nnUNet_raw_data_base
    nnUNet_preprocessed
    RESULTS_FOLDER
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from datetime import datetime
from typing import List

import SimpleITK as sitk

from batchgenerators.utilities.file_and_folder_operations import (
	join,
	load_pickle,
	maybe_mkdir_p,
	subfiles,
)

from nnunet_mednext.evaluation.evaluator import Evaluator
from nnunet_mednext.inference.predict import (
	check_input_folder_and_return_caseIDs,
	preprocess_multithreaded,
)
from nnunet_mednext.inference.segmentation_export import save_segmentation_nifti
from nnunet_mednext.inference.segmentation_export import save_segmentation_nifti
from nnunet_mednext.paths import nnUNet_raw_data, network_training_output_dir
from nnunet_mednext.training.model_restore import load_model_and_checkpoint_files



TASK_NAME = "Task017_AbdominalOrganSegmentation"
TRAINER_NAME = "nnUNetTrainerV2_MedNeXt_S_kernel3"
PLANS_ID = "nnUNetPlansv2.1_trgSp_1x1x1"

def _predict_cases_with_trainer(
	trainer,
	model_folder: str,
	list_of_lists: List[List[str]],
	output_filenames: List[str],
	num_threads_preprocessing: int = 4,
	num_threads_nifti_save: int = 2,
	lowres_segmentations: List[str] = None,
	tta: bool = True,
	mixed_precision: bool = False,
	overwrite_existing: bool = True,
	step_size: float = 0.5,
	disable_postprocessing: bool = True,
) -> None:
	"""Minimal copy of nnUNet prediction loop for a single trainer.

	This mirrors the behavior of predict_cases (without ensembling) but
	operates on an already-loaded trainer instance. It saves NIfTI
	segmentations for each case in list_of_lists.
	"""
	from multiprocessing import Pool

	import numpy as np
	from batchgenerators.utilities.file_and_folder_operations import isfile
	from nnunet_mednext.inference.predict import (
		load_postprocessing,
		load_remove_save,
	)

	local_timings: dict[str, float] = {}
	t_func_start = time.perf_counter()
	assert len(list_of_lists) == len(output_filenames), "Mismatched inputs/outputs"
	maybe_mkdir_p(os.path.dirname(output_filenames[0]))

	segs_from_prev_stage = lowres_segmentations

	print("Starting preprocessing...")
	t_preproc_predict_start = time.perf_counter()
	preprocess_iterator = preprocess_multithreaded(
		trainer,
		list_of_lists,
		output_filenames,
		num_threads_preprocessing,
		segs_from_prev_stage,
	)
	local_timings["preprocess_iterator_creation_sec"] = (
		time.perf_counter() - t_preproc_predict_start
	)
	print(
		"[infer][timing] Created preprocess iterator: "
		f"{local_timings['preprocess_iterator_creation_sec']:.2f} s"
	)

	print("Starting prediction...")
	results = []
	pool = Pool(num_threads_nifti_save)

	total_cases = len(list_of_lists)
	print(f"[evaluate] Prediction loop over {total_cases} case(s)...")
	sum_pred_sec = 0.0
	t_first_case_start = time.perf_counter()

	for idx, preprocessed in enumerate(preprocess_iterator, 1):
		output_filename, (d, dct) = preprocessed
		print(f"[evaluate] ({idx}/{total_cases}) predicting case {output_filename}")
		# Match nnUNet's predict.py behavior: d may be a np.ndarray or a
		# temporary .npy filename created for large arrays.
		if isinstance(d, str):
			data = np.load(d)
			os.remove(d)
		else:
			data = d

		# Time just the sliding-window + TTA prediction call.
		t_pred_start = time.perf_counter()
		seg, _ = trainer.predict_preprocessed_data_return_seg_and_softmax(
			data,
			do_mirroring=tta,
			mirror_axes=trainer.data_aug_params.get("mirror_axes", (0, 1, 2)),
			use_sliding_window=True,
			step_size=step_size,
			use_gaussian=True,
			all_in_gpu=False,
			mixed_precision=mixed_precision,
		)
		this_pred_sec = time.perf_counter() - t_pred_start
		sum_pred_sec += this_pred_sec
		print(
			"[infer][timing] Sliding-window prediction (with TTA) for case "
			f"{output_filename}: {this_pred_sec:.2f} s"
		)

		transpose_forward = trainer.plans.get("transpose_forward")
		if transpose_forward is not None:
			transpose_backward = trainer.plans.get("transpose_backward")
			seg = seg.transpose([i for i in transpose_backward])

		results.append(
			pool.starmap_async(
				save_segmentation_nifti,
				((seg, output_filename, dct, 0, None),),
			)
		)

	t_preproc_predict_end = time.perf_counter()
	local_timings["sliding_window_prediction_total_sec"] = sum_pred_sec
	local_timings["preprocess_and_predict_no_export_sec"] = (
		t_preproc_predict_end - t_preproc_predict_start
	)
	print(
		"[infer][timing] Total sliding-window prediction (all cases): "
		f"{local_timings['sliding_window_prediction_total_sec']:.2f} s"
	)
	print(
		"[infer][timing] Preprocessing + prediction (no export): "
		f"{local_timings['preprocess_and_predict_no_export_sec']:.2f} s"
	)

	print("Inference done. Waiting for NIfTI export to finish...")
	t_export_start = time.perf_counter()
	_ = [i.get() for i in results]
	t_export_end = time.perf_counter()
	local_timings["nifti_export_sec"] = t_export_end - t_export_start
	print(
		f"[infer][timing] NIfTI export (all cases): "
		f"{local_timings['nifti_export_sec']:.2f} s"
	)

	if not disable_postprocessing:
		pp_file = join(model_folder, "postprocessing.json")
		if os.path.isfile(pp_file):
			print("Running postprocessing...")
			shutil.copy(pp_file, os.path.dirname(output_filenames[0]))
			for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
			pp_results = []
			t_pp_start = time.perf_counter()
			pp_results.append(
				pool.starmap_async(
					load_remove_save,
					zip(
						output_filenames,
						output_filenames,
						[for_which_classes] * len(output_filenames),
						[min_valid_obj_size] * len(output_filenames),
					),
				),
			)
			_ = [i.get() for i in pp_results]
			local_timings["postprocessing_sec"] = time.perf_counter() - t_pp_start
			print(
				f"[infer][timing] Postprocessing (all cases): "
				f"{local_timings['postprocessing_sec']:.2f} s"
			)
		else:
			print(
				"WARNING: postprocessing.json not found; skipping postprocessing."
			)

	pool.close()
	pool.join()

	# Aggregate overall timings for this helper.
	local_timings["preprocess_slidingwindow_export_total_sec"] = (
		t_export_end - t_preproc_predict_start
	)
	# Everything that is not explicit sliding-window prediction or export
	# is counted as preprocessing + orchestration overhead.
	pre_overhead = (
		local_timings["preprocess_slidingwindow_export_total_sec"]
		- local_timings["sliding_window_prediction_total_sec"]
		- local_timings["nifti_export_sec"]
	)
	if pre_overhead < 0:
		pre_overhead = 0.0
	local_timings["preprocessing_and_overhead_sec"] = pre_overhead
	local_timings["predict_helper_total_sec"] = time.perf_counter() - t_func_start
	print(
		"[infer][timing] Preprocessing + sliding-window + export (helper total): "
		f"{local_timings['preprocess_slidingwindow_export_total_sec']:.2f} s"
	)
	print(
		"[infer][timing] Preprocessing + overhead (approx.): "
		f"{local_timings['preprocessing_and_overhead_sec']:.2f} s"
	)
	print(
		"[infer][timing] _predict_cases_with_trainer total runtime: "
		f"{local_timings['predict_helper_total_sec']:.2f} s"
	)

	return local_timings

def _get_model_folder() -> str:
	"""Return the training output folder for Task017 MedNeXt-S (fold 0)."""

	if network_training_output_dir is None:
		raise RuntimeError(
			"network_training_output_dir is None. Make sure RESULTS_FOLDER is set "
			"before running inference."
		)

	model_folder = join(
		network_training_output_dir,
		"3d_fullres",
		TASK_NAME,
		f"{TRAINER_NAME}__{PLANS_ID}",
	)
	if not os.path.isdir(model_folder):
		raise RuntimeError(
			f"Model folder not found: {model_folder}. "
			"Check RESULTS_FOLDER/nnUNet/3d_fullres paths."
		)
	return model_folder


def _pick_random_case(input_folder: str, model_folder: str) -> str:
	"""Pick a single random case ID from BTCV imagesTr.

	Uses plans.pkl to determine the expected number of modalities and
	`check_input_folder_and_return_caseIDs` to derive valid case IDs.
	"""

	plans_path = join(model_folder, "plans.pkl")
	if not os.path.isfile(plans_path):
		raise RuntimeError(f"plans.pkl not found in model folder: {plans_path}")
	plans = load_pickle(plans_path)
	expected_num_modalities = plans["num_modalities"]

	all_case_ids: List[str] = list(
		check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)
	)
	if not all_case_ids:
		raise RuntimeError(f"No valid case IDs found in {input_folder}")

	case_id = random.choice(all_case_ids)
	print(f"[infer] Selected random case: {case_id}")
	return case_id


def _build_case_file_lists(input_folder: str, case_id: str) -> List[List[str]]:
	"""Build nnUNet-style list_of_lists for a single case.

	The case may have multiple modalities, e.g. *_0000.nii.gz, *_0001.nii.gz.
	"""

	all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
	case_files = [
		join(input_folder, f)
		for f in all_files
		if f[: len(case_id)].startswith(case_id) and len(f) == (len(case_id) + 12)
	]
	if not case_files:
		raise RuntimeError(f"No image files found for case ID {case_id} in {input_folder}")
	return [case_files]


def _run_single_case_inference() -> dict:
	"""Run inference on one random BTCV case and compute metrics.

	Returns a dictionary with case ID and per-label metrics suitable for
	JSON serialization.
	"""

	timings: dict[str, float] = {}
	t_overall_start = time.perf_counter()

	if network_training_output_dir is None or nnUNet_raw_data is None:
		raise RuntimeError(
			"nnUNet paths are not configured. Make sure nnunet_mednext.paths "
			"has been set up and environment variables are exported."
		)

	model_folder = _get_model_folder()
	input_folder = join(nnUNet_raw_data, TASK_NAME, "imagesTr")
	gt_folder = join(nnUNet_raw_data, TASK_NAME, "labelsTr")
	if not os.path.isdir(input_folder) or not os.path.isdir(gt_folder):
		raise RuntimeError(
			f"Expected imagesTr and labelsTr under {join(nnUNet_raw_data, TASK_NAME)}"
		)

	# Pick a random case and build its modality file list
	t0 = time.perf_counter()
	case_id = _pick_random_case(input_folder, model_folder)
	list_of_lists = _build_case_file_lists(input_folder, case_id)
	timings["case_selection_and_filelist_sec"] = time.perf_counter() - t0
	print(
		f"[infer][timing] Case selection + file list: "
		f"{timings['case_selection_and_filelist_sec']:.2f} s"
	)

	# Prepare output folder for the prediction NIfTI
	this_dir = os.path.dirname(os.path.abspath(__file__))
	pred_dir = join(this_dir, "predictions_single")
	maybe_mkdir_p(pred_dir)
	output_files = [join(pred_dir, case_id + ".nii.gz")]

	t0 = time.perf_counter()
	print("[infer] Loading FP32 trainer...")
	trainer, params = load_model_and_checkpoint_files(
		model_folder,
		folds=[0],
		mixed_precision=False,
		checkpoint_name="model_final_checkpoint",
	)
	if not params:
		raise RuntimeError("No checkpoint parameters returned by load_model_and_checkpoint_files")
	trainer.load_checkpoint_ram(params[0], False)
	trainer.network.eval()
	timings["model_load_sec"] = time.perf_counter() - t0
	print(f"[infer][timing] Model load (trainer + checkpoint): {timings['model_load_sec']:.2f} s")

	t0 = time.perf_counter()
	print("[infer] Running nnUNet-style inference for the selected case...")
	predict_timings = _predict_cases_with_trainer(
		trainer,
		model_folder,
		list_of_lists,
		output_files,
		num_threads_preprocessing=2,
		num_threads_nifti_save=1,
		lowres_segmentations=None,
		tta=True,
		mixed_precision=False,
		overwrite_existing=True,
		step_size=0.5,
		disable_postprocessing=True,
	)
	timings["preprocessing_prediction_export_total_sec"] = time.perf_counter() - t0
	for k, v in predict_timings.items():
		timings[f"predict_{k}"] = v
	print(
		"[infer][timing] Preprocessing + sliding-window prediction + NIfTI export (outer): "
		f"{timings['preprocessing_prediction_export_total_sec']:.2f} s"
	)

	# Load predicted and reference segmentations
	pred_path = output_files[0]
	gt_path = join(gt_folder, case_id + ".nii.gz")
	if not os.path.isfile(gt_path):
		raise RuntimeError(f"Ground-truth label file not found: {gt_path}")

	print(f"[infer] Loading prediction from {pred_path}")
	print(f"[infer] Loading ground truth from {gt_path}")
	pred_img = sitk.ReadImage(pred_path)
	gt_img = sitk.ReadImage(gt_path)

	t0 = time.perf_counter()
	pred = sitk.GetArrayFromImage(pred_img)
	gt = sitk.GetArrayFromImage(gt_img)
	timings["nifti_read_and_to_array_sec"] = time.perf_counter() - t0
	print(
		"[infer][timing] Load NIfTI + to numpy arrays: "
		f"{timings['nifti_read_and_to_array_sec']:.2f} s"
	)

	# Compute metrics using nnUNet's Evaluator
	print("[infer] Computing evaluation metrics (Dice, Accuracy, etc.)...")
	t0 = time.perf_counter()
	evaluator = Evaluator(test=pred, reference=gt)
	metrics_per_label = evaluator.to_dict()
	timings["metrics_evaluation_sec"] = time.perf_counter() - t0
	print(
		f"[infer][timing] Metric computation (Evaluator): "
		f"{timings['metrics_evaluation_sec']:.2f} s"
	)

	# Aggregate simple global metrics (mean over labels)
	metric_names = set()
	for label_result in metrics_per_label.values():
		metric_names.update(label_result.keys())

	aggregate = {}
	for m in sorted(metric_names):
		vals = [float(label_result[m]) for label_result in metrics_per_label.values()]
		if not vals:
			continue
		aggregate[m] = float(sum(vals) / len(vals))

	timings["total_pipeline_sec"] = time.perf_counter() - t_overall_start
	print(
		f"[infer][timing] TOTAL pipeline time (case selection -> metrics): "
		f"{timings['total_pipeline_sec']:.2f} s"
	)

	return {
		"case_id": case_id,
		"prediction_path": pred_path,
		"ground_truth_path": gt_path,
		"metrics_per_label": metrics_per_label,
		"metrics_aggregate_mean": aggregate,
		"timings_sec": timings,
	}


def main() -> None:
	"""Entry point for random single-case inference and metric dump."""

	result = _run_single_case_inference()

	this_dir = os.path.dirname(os.path.abspath(__file__))
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_path = join(this_dir, f"inference.json")
	print(f"[infer] Writing metrics JSON to {out_path}")
	with open(out_path, "w") as f:
		json.dump(result, f, indent=2)

	print("[infer] Done.")


if __name__ == "__main__":
	main()
