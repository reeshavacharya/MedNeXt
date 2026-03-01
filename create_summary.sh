python - << 'PY'
import os
from batchgenerators.utilities.file_and_folder_operations import subfiles, join
from nnunet_mednext.evaluation.evaluator import aggregate_scores

gt = "/data/reeshav/MedNeXt_dataset/Abdomen/nnUNet_raw_data/Task017_AbdominalOrganSegmentation/labelsTr"
pred = "/home/r/reeshav/MedNeXt/nnUNet_results/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/validation_raw"

# Use only the cases for which we actually have predictions
pred_cases = subfiles(pred, suffix=".nii.gz", join=False)
if not pred_cases:
	raise RuntimeError(f"No .nii.gz prediction files found in {pred}")

pairs = []
for c in pred_cases:
	gt_file = join(gt, c)
	if not os.path.isfile(gt_file):
		raise RuntimeError(f"Missing ground truth for predicted case {c}: expected {gt_file}")
	pairs.append((join(pred, c), gt_file))

# labels 0–13 for BTCV (0=background)
out_json = os.path.join(pred, "summary.json")
aggregate_scores(pairs, json_output_file=out_json, num_threads=8, labels=tuple(range(14)))
print(f"Wrote summary.json to {out_json}")
PY