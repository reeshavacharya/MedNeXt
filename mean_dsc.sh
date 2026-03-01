#!/bin/bash

# Compute mean DSC (Dice) from an nnUNet/MedNeXt summary.json.
# - By default, uses the validation_raw_postprocessed summary for
#   Task017 MedNeXt-S 3d_fullres fold_0.
# - You can also pass a different summary.json path as $1.
#
# Behavior:
# - Reads results["mean"][label]["Dice"]
# - Ignores label "0" (background) and any Dice values that are NaN
# - Prints mean DSC over the remaining labels.

set -e

PROJECT_ROOT="/home/r/reeshav/MedNeXt"
DEFAULT_SUMMARY="$PROJECT_ROOT/nnUNet_results/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/validation_raw_postprocessed/summary.json"

SUMMARY_PATH="${1:-$DEFAULT_SUMMARY}"

if [ ! -f "$SUMMARY_PATH" ]; then
  echo "summary.json not found at: $SUMMARY_PATH" >&2
  exit 1
fi

# Use the project virtualenv Python if available
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" - "$SUMMARY_PATH" << 'EOF'
import json, sys, math

summary_path = sys.argv[1]
with open(summary_path, 'r') as f:
    data = json.load(f)

results_mean = data["results"]["mean"]

dices = []
for label, metrics in results_mean.items():
    # skip background label 0 if present
    if str(label) == "0":
        continue
    d = metrics.get("Dice")
    if d is None:
        continue
    # json may encode NaN as a float nan
    try:
        if math.isnan(d):
            continue
    except TypeError:
        # non-numeric, skip
        continue
    dices.append(float(d))

if not dices:
    print("No valid Dice values found (after excluding NaNs/background)")
    sys.exit(1)

mean_dsc = sum(dices) / len(dices)
print(f"Mean DSC over {len(dices)} labels: {mean_dsc:.4f} ({mean_dsc*100:.2f}%)")
EOF
