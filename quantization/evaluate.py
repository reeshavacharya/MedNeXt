"""Side-by-side evaluation of clean vs fake-quantized MedNeXt.

This script:

1. Loads the standard FP32 trainer for Task017.
2. Loads a fake-quantized trainer (Conv/Linear activations quantized
   according to quantization/scales.json).
3. Runs inference for both trainers on the BTCV imagesTr set.
4. Evaluates Dice scores against labelsTr using the standard nnUNet
   evaluation utilities.
5. Writes two JSON files in the quantization/ directory:
   - mednext_validation.json (clean model)
   - mednext_quantized_validation.json (fake-quantized model)

Usage (from project root, with nnUNet env vars set):

    python -m quantization.evaluate
"""

import json
import os
import random
import shutil
from copy import deepcopy
from typing import List, Tuple

from batchgenerators.utilities.file_and_folder_operations import (
	join,
	maybe_mkdir_p,
	subfiles,
	load_pickle,
)

from nnunet_mednext.evaluation.evaluator import evaluate_folder, aggregate_scores
from nnunet_mednext.inference.predict import (
	check_input_folder_and_return_caseIDs,
	preprocess_multithreaded,
	save_segmentation_nifti,
)
from nnunet_mednext.paths import network_training_output_dir, nnUNet_raw_data
from nnunet_mednext.training.model_restore import load_model_and_checkpoint_files

from .quantized_model import load_fake_quantized_trainer


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

	assert len(list_of_lists) == len(output_filenames), "Mismatched inputs/outputs"
	maybe_mkdir_p(os.path.dirname(output_filenames[0]))

	segs_from_prev_stage = lowres_segmentations

	print("Starting preprocessing...")
	preprocess_iterator = preprocess_multithreaded(
		trainer,
		list_of_lists,
		output_filenames,
		num_threads_preprocessing,
		segs_from_prev_stage,
	)

	print("Starting prediction...")
	results = []
	pool = Pool(num_threads_nifti_save)

	total_cases = len(list_of_lists)
	print(f"[evaluate] Prediction loop over {total_cases} case(s)...")

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

	print("Inference done. Waiting for NIfTI export to finish...")
	_ = [i.get() for i in results]

	if not disable_postprocessing:
		pp_file = join(model_folder, "postprocessing.json")
		if os.path.isfile(pp_file):
			print("Running postprocessing...")
			shutil.copy(pp_file, os.path.dirname(output_filenames[0]))
			for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
			pp_results = []
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
		else:
			print(
				"WARNING: postprocessing.json not found; skipping postprocessing."
			)

	pool.close()
	pool.join()


def main():
	if network_training_output_dir is None or nnUNet_raw_data is None:
		raise RuntimeError(
			"nnUNet paths are not configured. Make sure nnunet_mednext.paths "
			"has been set up and environment variables are exported."
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

	input_folder = join(nnUNet_raw_data, task_name, "imagesTr")
	gt_folder = join(nnUNet_raw_data, task_name, "labelsTr")
	if not os.path.isdir(input_folder) or not os.path.isdir(gt_folder):
		raise RuntimeError(
			f"Expected imagesTr and labelsTr under {join(nnUNet_raw_data, task_name)}"
		)

	this_dir = os.path.dirname(os.path.abspath(__file__))
	out_dir_clean = join(this_dir, "predictions_clean")
	out_dir_quant = join(this_dir, "predictions_quantized")
	maybe_mkdir_p(out_dir_clean)
	maybe_mkdir_p(out_dir_quant)

	print("[evaluate] Preparing case list from imagesTr...")
	plans_path = join(model_folder, "plans.pkl")
	assert os.path.isfile(plans_path), "plans.pkl not found in model folder"
	plans = load_pickle(plans_path)
	expected_num_modalities = plans["num_modalities"]
	all_case_ids = list(
		check_input_folder_and_return_caseIDs(
			input_folder, expected_num_modalities
		)
	)

	# Subsample a fixed number of cases for quick side-by-side eval
	n_samples = 20
	if len(all_case_ids) <= n_samples:
		case_ids = all_case_ids
	else:
		# Use a fixed seed for reproducibility
		random.seed(42)
		case_ids = sorted(random.sample(all_case_ids, n_samples))

	print(
		f"[evaluate] Using {len(case_ids)} cases out of {len(all_case_ids)} for evaluation."
	)

	all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
	list_of_lists = [
		[
			join(input_folder, i)
			for i in all_files
			if i[: len(j)].startswith(j) and len(i) == (len(j) + 12)
		]
		for j in case_ids
	]
	output_files_clean = [join(out_dir_clean, i + ".nii.gz") for i in case_ids]
	output_files_quant = [join(out_dir_quant, i + ".nii.gz") for i in case_ids]

	print("[evaluate] Loading clean FP32 trainer...")
	trainer_clean, params = load_model_and_checkpoint_files(
		model_folder,
		folds=[0],
		mixed_precision=False,
		checkpoint_name="model_final_checkpoint",
	)
	if not params:
		raise RuntimeError(
			"No checkpoint parameters returned by load_model_and_checkpoint_files"
		)
	trainer_clean.load_checkpoint_ram(params[0], False)
	trainer_clean.network.eval()

	print("[evaluate] Loading fake-quantized trainer...")
	trainer_quant = load_fake_quantized_trainer()

	print("[evaluate] Running inference with clean model...")
	_predict_cases_with_trainer(
		trainer_clean,
		model_folder,
		list_of_lists,
		output_files_clean,
		num_threads_preprocessing=4,
		num_threads_nifti_save=2,
		lowres_segmentations=None,
		tta=True,
		mixed_precision=False,
		overwrite_existing=True,
		step_size=0.5,
		disable_postprocessing=True,
	)

	print("[evaluate] Running inference with fake-quantized model...")
	_predict_cases_with_trainer(
		trainer_quant,
		model_folder,
		list_of_lists,
		output_files_quant,
		num_threads_preprocessing=4,
		num_threads_nifti_save=2,
		lowres_segmentations=None,
		tta=True,
		mixed_precision=False,
		overwrite_existing=True,
		step_size=0.5,
		disable_postprocessing=True,
	)

	print("[evaluate] Evaluating Dice scores for clean model predictions (subset)...")
	files_gt_subset = [cid + ".nii.gz" for cid in case_ids]
	test_ref_pairs_clean = [
		(join(out_dir_clean, f), join(gt_folder, f)) for f in files_gt_subset
	]
	res_clean = aggregate_scores(
		test_ref_pairs_clean,
		labels=None,
		num_threads=4,
		json_output_file=join(out_dir_clean, "summary.json"),
	)

	print("[evaluate] Evaluating Dice scores for quantized model predictions (subset)...")
	test_ref_pairs_quant = [
		(join(out_dir_quant, f), join(gt_folder, f)) for f in files_gt_subset
	]
	res_quant = aggregate_scores(
		test_ref_pairs_quant,
		labels=None,
		num_threads=4,
		json_output_file=join(out_dir_quant, "summary.json"),
	)

	clean_json = join(this_dir, "mednext_validation.json")
	quant_json = join(this_dir, "mednext_quantized_validation.json")

	with open(clean_json, "w") as f:
		json.dump(res_clean, f, indent=2)

	with open(quant_json, "w") as f:
		json.dump(res_quant, f, indent=2)

	print(f"[evaluate] Wrote clean model results to {clean_json}")
	print(f"[evaluate] Wrote quantized model results to {quant_json}")


if __name__ == "__main__":
	main()
