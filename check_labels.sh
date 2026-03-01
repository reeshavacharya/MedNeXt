cd /home/r/reeshav/MedNeXt
source .venv/bin/activate

python - << 'PY'
import os, glob
import numpy as np
import nibabel as nib

bases = [
    ("BTCV RawData Training labels", "/data/reeshav/MedNeXt_dataset/Abdomen/RawData/Training/label"),
    ("nnUNet Task017 labelsTr", "/data/reeshav/MedNeXt_dataset/Abdomen/nnUNet_raw_data/Task017_AbdominalOrganSegmentation/labelsTr"),
]

for name, d in bases:
    print(f"\n=== {name} ===")
    if not os.path.isdir(d):
        print(f"[WARN] directory does not exist: {d}")
        continue
    files = sorted(glob.glob(os.path.join(d, "*.nii.gz")))
    if not files:
        print(f"[WARN] no .nii.gz files in {d}")
        continue
    all_vals = set()
    for f in files:
        data = nib.load(f).get_fdata()
        all_vals.update(np.unique(data.astype(np.int16)).tolist())
    print("Union of label values:", sorted(all_vals))
PY