"""Compare FP32 nnUNet MedNeXt vs INT8 ONNX model on Task017.

This script runs inference for both:

1. The original trained nnUNet MedNeXt-S (kernel 3) FP32 model.
2. The INT8-quantized ONNX model produced by quantize_int8.py.

on the same subset of BTCV imagesTr, then evaluates Dice, Accuracy,
Jaccard, etc. against labelsTr using nnUNet's evaluator.

It writes two JSON files in ONNX_quantization/:

    clean_mednext.json   (PyTorch FP32)
    onnx_mednext.json    (INT8 ONNX)

Usage (from project root, with nnUNet env vars set as usual):

    python -m ONNX_quantization.evaluate_onnx \
        --int8_model ONNX_quantization/mednext_task017_int8.onnx \
        --num_cases 10
"""

import argparse
import os
import random
from pathlib import Path
from typing import List

import numpy as np

from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
)

from nnunet_mednext.evaluation.evaluator import evaluate_folder
from nnunet_mednext.inference.predict import (
    check_input_folder_and_return_caseIDs,
    save_segmentation_nifti,
)
from nnunet_mednext.paths import network_training_output_dir, nnUNet_raw_data
from nnunet_mednext.training.model_restore import load_model_and_checkpoint_files


def load_trainer():
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

    print(f"[evaluate_onnx] Loading FP32 trainer from {model_folder}")
    trainer, params = load_model_and_checkpoint_files(
        model_folder,
        folds=[0],
        mixed_precision=False,
        checkpoint_name="model_final_checkpoint",
    )
    if not params:
        raise RuntimeError(
            "No checkpoint parameters returned by load_model_and_checkpoint_files"
        )

    trainer.load_checkpoint_ram(params[0], False)
    trainer.network.eval()
    return trainer, model_folder


def get_cases(input_folder: str, model_folder: str, num_cases: int) -> List[str]:
    from batchgenerators.utilities.file_and_folder_operations import load_pickle

    plans_path = join(model_folder, "plans.pkl")
    assert os.path.isfile(plans_path), "plans.pkl not found in model folder"
    plans = load_pickle(plans_path)
    expected_num_modalities = plans["num_modalities"]

    all_case_ids = list(
        check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)
    )

    if len(all_case_ids) <= num_cases:
        case_ids = all_case_ids
    else:
        random.seed(42)
        case_ids = sorted(random.sample(all_case_ids, num_cases))

    print(
        f"[evaluate_onnx] Using {len(case_ids)} cases out of {len(all_case_ids)} for evaluation."
    )
    return case_ids


def run_fp32_inference(trainer, case_ids: List[str], input_folder: str, out_dir: Path):
    print("[evaluate_onnx] Running FP32 nnUNet inference...")
    from batchgenerators.utilities.file_and_folder_operations import isfile

    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, cid in enumerate(case_ids, 1):
        input_files = []
        # We know num_modalities from plans, but for BTCV it's 1 (CT)
        img_file = join(input_folder, f"{cid}_0000.nii.gz")
        if not isfile(img_file):
            raise RuntimeError(f"Expected image file not found: {img_file}")
        input_files.append(img_file)

        print(f"[FP32] ({idx}/{len(case_ids)}) preprocessing {cid}")
        d, s, properties = trainer.preprocess_patient(input_files)

        print(f"[FP32] ({idx}/{len(case_ids)}) predicting {cid}")
        seg, _ = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d,
            do_mirroring=trainer.data_aug_params.get("do_mirror", True),
            mirror_axes=trainer.data_aug_params.get("mirror_axes", (0, 1, 2)),
            use_sliding_window=True,
            step_size=0.5,
            use_gaussian=True,
            pad_border_mode="constant",
            pad_kwargs={"constant_values": 0},
            all_in_gpu=False,
            verbose=True,
            mixed_precision=False,
        )

        transpose_forward = trainer.plans.get("transpose_forward")
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get("transpose_backward")
            seg = seg.transpose([i for i in transpose_backward])

        out_file = out_dir / f"{cid}.nii.gz"
        save_segmentation_nifti(seg, str(out_file), properties, 0, None)


def _compute_steps_for_sliding_window(
    patch_size: tuple, image_size: tuple, step_size: float
):
    assert 0 < step_size <= 1, "step_size must be larger than 0 and <= 1"

    target_step_sizes_in_voxels = [i * step_size for i in patch_size]
    num_steps = [
        int(np.ceil((i - k) / j)) + 1
        for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)
    ]

    steps = []
    for dim in range(len(patch_size)):
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)

    return steps


def run_onnx_inference(
    int8_model: Path,
    trainer,
    case_ids: List[str],
    input_folder: str,
    out_dir: Path,
):
    print(f"[evaluate_onnx] Running INT8 ONNX inference with {int8_model}")
    import onnxruntime as ort
    from batchgenerators.utilities.file_and_folder_operations import isfile

    out_dir.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(
        str(int8_model),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Use the same patch_size and step_size as nnUNet
    patch_size = tuple(trainer.patch_size)
    step_size = 0.5

    for idx, cid in enumerate(case_ids, 1):
        input_files = []
        img_file = join(input_folder, f"{cid}_0000.nii.gz")
        if not isfile(img_file):
            raise RuntimeError(f"Expected image file not found: {img_file}")
        input_files.append(img_file)

        print(f"[INT8] ({idx}/{len(case_ids)}) preprocessing {cid}")
        d, s, properties = trainer.preprocess_patient(input_files)

        print(f"[INT8] ({idx}/{len(case_ids)}) running ONNX sliding-window for {cid}")

        # d: (C, D, H, W). Pad to at least patch_size, then do sliding-window
        # inference with simple averaging over overlapping tiles.
        data, slicer = pad_nd_image(
            d,
            patch_size,
            "constant",
            {"constant_values": 0},
            True,
            None,
        )
        data_shape = data.shape  # (C, X, Y, Z)

        steps = _compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)

        aggregated_results = None
        aggregated_nb_of_predictions = None

        for x0 in steps[0]:
            lb_x = x0
            ub_x = x0 + patch_size[0]
            for y0 in steps[1]:
                lb_y = y0
                ub_y = y0 + patch_size[1]
                for z0 in steps[2]:
                    lb_z = z0
                    ub_z = z0 + patch_size[2]

                    patch = data[
                        :,
                        lb_x:ub_x,
                        lb_y:ub_y,
                        lb_z:ub_z,
                    ]
                    x = patch.astype(np.float32)[np.newaxis, ...]
                    logits = sess.run([output_name], {input_name: x})[0]
                    # logits: (1, C_out, ps0, ps1, ps2)
                    if aggregated_results is None:
                        num_classes = logits.shape[1]
                        aggregated_results = np.zeros(
                            (num_classes,) + tuple(data_shape[1:]),
                            dtype=np.float32,
                        )
                        aggregated_nb_of_predictions = np.zeros_like(
                            aggregated_results,
                            dtype=np.float32,
                        )

                    predicted_patch = logits[0]  # (C_out, ps0, ps1, ps2)

                    aggregated_results[
                        :,
                        lb_x:ub_x,
                        lb_y:ub_y,
                        lb_z:ub_z,
                    ] += predicted_patch
                    aggregated_nb_of_predictions[
                        :,
                        lb_x:ub_x,
                        lb_y:ub_y,
                        lb_z:ub_z,
                    ] += 1.0

        # Crop away padding (reverse pad_nd_image slicer logic)
        slicer_full = tuple(
            [
                slice(0, aggregated_results.shape[i])
                for i in range(len(aggregated_results.shape) - (len(slicer) - 1))
            ]
            + list(slicer[1:])
        )
        aggregated_results = aggregated_results[slicer_full]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer_full]

        aggregated_results /= aggregated_nb_of_predictions
        seg = np.argmax(aggregated_results, axis=0).astype(np.uint8)

        transpose_forward = trainer.plans.get("transpose_forward")
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get("transpose_backward")
            seg = seg.transpose([i for i in transpose_backward])

        out_file = out_dir / f"{cid}.nii.gz"
        save_segmentation_nifti(seg, str(out_file), properties, 0, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--int8_model",
        type=str,
        required=True,
        help="Path to INT8 ONNX model (output of quantize_int8.py)",
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=10,
        help="Number of imagesTr cases to evaluate (subset)",
    )
    args = parser.parse_args()

    if network_training_output_dir is None or nnUNet_raw_data is None:
        raise RuntimeError(
            "nnUNet paths are not configured. Make sure nnunet_mednext.paths is set up."
        )

    int8_model = Path(args.int8_model).resolve()
    if not int8_model.is_file():
        raise FileNotFoundError(f"INT8 ONNX model not found: {int8_model}")

    task_name = "Task017_AbdominalOrganSegmentation"
    input_folder = join(nnUNet_raw_data, task_name, "imagesTr")
    gt_folder = join(nnUNet_raw_data, task_name, "labelsTr")
    if not os.path.isdir(input_folder) or not os.path.isdir(gt_folder):
        raise RuntimeError(
            f"Expected imagesTr and labelsTr under {join(nnUNet_raw_data, task_name)}"
        )

    trainer, model_folder = load_trainer()

    case_ids = get_cases(input_folder, model_folder, args.num_cases)

    this_dir = Path(__file__).resolve().parent
    out_clean = this_dir / "predictions_clean"
    out_onnx = this_dir / "predictions_int8"

    run_fp32_inference(trainer, case_ids, input_folder, out_clean)
    run_onnx_inference(int8_model, trainer, case_ids, input_folder, out_onnx)

    print("[evaluate_onnx] Evaluating Dice scores for FP32 predictions...")
    res_clean, _ = evaluate_folder(
        gt_folder,
        str(out_clean),
        labels=None,
        num_threads=4,
    )

    print("[evaluate_onnx] Evaluating Dice scores for INT8 ONNX predictions...")
    res_onnx, _ = evaluate_folder(
        gt_folder,
        str(out_onnx),
        labels=None,
        num_threads=4,
    )

    clean_json = this_dir / "clean_mednext.json"
    onnx_json = this_dir / "onnx_mednext.json"

    import json

    clean_json.write_text(json.dumps(res_clean, indent=2))
    onnx_json.write_text(json.dumps(res_onnx, indent=2))

    print(f"[evaluate_onnx] Wrote FP32 results to {clean_json}")
    print(f"[evaluate_onnx] Wrote INT8 ONNX results to {onnx_json}")


if __name__ == "__main__":
    main()
