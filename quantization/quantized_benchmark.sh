#!/bin/bash

# Benchmark clean vs quantized validation metrics.
# Reads mednext_validation.json and mednext_quantized_validation.json and
# writes quantized_benchmark.json with only the average Dice across labels
# (foreground classes) for clean and quantized validations.

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

def avg_dice(mean_dict):
    values = []
    for cls_id, metrics in mean_dict.items():
        # skip background class "0"
        if str(cls_id) == "0":
            continue
        d = metrics.get("Dice")
        if d is not None:
            values.append(d)
    return (sum(values) / len(values)) if values else None

result = {
    "clean_avg_dice": avg_dice(clean_mean),
    "quantized_avg_dice": avg_dice(quant_mean),
}

out_path.write_text(json.dumps(result, indent=2))
print(f"Wrote average Dice benchmark to {out_path}")
EOF
