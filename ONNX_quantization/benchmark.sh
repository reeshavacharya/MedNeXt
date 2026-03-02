#!/bin/bash

# Benchmark average Dice for FP32 vs INT8 ONNX MedNeXt.
#
# Reads clean_mednext.json and onnx_mednext.json (created by
# evaluate_onnx.py) and writes onnx_benchmark.json with:
#   {"clean_avg_dice": ..., "onnx_avg_dice": ...}
# averaging Dice across all foreground labels.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_JSON="$THIS_DIR/clean_mednext.json"
ONNX_JSON="$THIS_DIR/onnx_mednext.json"
OUT_JSON="$THIS_DIR/onnx_benchmark.json"

if [[ ! -f "$CLEAN_JSON" ]]; then
  echo "[benchmark] clean_mednext.json not found: $CLEAN_JSON" >&2
  exit 1
fi

if [[ ! -f "$ONNX_JSON" ]]; then
  echo "[benchmark] onnx_mednext.json not found: $ONNX_JSON" >&2
  exit 1
fi

python - "$CLEAN_JSON" "$ONNX_JSON" "$OUT_JSON" << 'EOF'
import json
import sys
from pathlib import Path

clean_path = Path(sys.argv[1])
onnx_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

clean = json.loads(clean_path.read_text())
onnx = json.loads(onnx_path.read_text())

clean_mean = clean.get("mean", {})
onnx_mean = onnx.get("mean", {})


def avg_dice(mean_dict):
    vals = []
    for cls_id, metrics in mean_dict.items():
        if str(cls_id) == "0":
            continue
        d = metrics.get("Dice")
        if d is not None:
            vals.append(d)
    return (sum(vals) / len(vals)) if vals else None


result = {
    "clean_avg_dice": avg_dice(clean_mean),
    "onnx_avg_dice": avg_dice(onnx_mean),
}

out_path.write_text(json.dumps(result, indent=2))
print(f"Wrote ONNX benchmark summary to {out_path}")
EOF
