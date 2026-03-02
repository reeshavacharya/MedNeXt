#!/bin/bash

# Benchmark clean vs quantized validation metrics.
# Reads mednext_validation.json and mednext_quantized_validation.json and
# writes quantized_benchmark.json with the foreground-class averages for
# all available scalar metrics (Dice, Accuracy, Jaccard, etc.) for both
# clean and quantized validations.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_JSON="$THIS_DIR/mednext_validation.json"
QUANT_JSON="$THIS_DIR/mednext_quantized_validation.json"
OUT_JSON="$THIS_DIR/quantized_benchmark.json"

if [[ ! -f "$CLEAN_JSON" ]]; then
  echo "Clean validation JSON not found: $CLEAN_JSON" >&2
  exit 1
fi

if [[ ! -f "$QUANT_JSON" ]]; then
  echo "Quantized validation JSON not found: $QUANT_JSON" >&2
  exit 1
fi

python - "$CLEAN_JSON" "$QUANT_JSON" "$OUT_JSON" << 'EOF'
import json
import sys
from pathlib import Path

clean_path = Path(sys.argv[1])
quant_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

clean_data = json.loads(clean_path.read_text())
quant_data = json.loads(quant_path.read_text())

clean_mean = clean_data.get("mean", {})
quant_mean = quant_data.get("mean", {})


def foreground_average_for_all_metrics(mean_dict):
  """Compute per-metric averages over all foreground classes.

  mean_dict is the "mean" section from nnUNet's aggregate_scores,
  mapping class_id -> {metric_name: value}.
  We skip class "0" (background) and, for each metric present in the
  first foreground class, compute the arithmetic mean across
  foreground classes where that metric is defined.
  """

  # Find one foreground class to discover available metric names
  first_fg_metrics = None
  for cls_id, metrics in mean_dict.items():
    if str(cls_id) == "0":
      continue
    first_fg_metrics = metrics
    break

  if first_fg_metrics is None:
    return {}

  metric_names = list(first_fg_metrics.keys())
  averages = {}

  for metric in metric_names:
    vals = []
    for cls_id, metrics in mean_dict.items():
      if str(cls_id) == "0":
        continue
      v = metrics.get(metric)
      if isinstance(v, (int, float)):
        vals.append(float(v))
    if vals:
      averages[metric] = sum(vals) / len(vals)

  return averages


result = {
  "clean": foreground_average_for_all_metrics(clean_mean),
  "quantized": foreground_average_for_all_metrics(quant_mean),
}

out_path.write_text(json.dumps(result, indent=2))
print(f"Wrote averaged metric benchmark to {out_path}")
EOF
