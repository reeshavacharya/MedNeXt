"""Export MedNeXt Task017 model to ONNX for INT8 PTQ backends.

This script loads the trained nnUNet MedNeXt-S (kernel 3) model for
Task017_AbdominalOrganSegmentation and exports only the network
forward (no nnUNet trainer logic) to an ONNX file.

The resulting ONNX can then be quantized and executed with real INT8
backends such as TensorRT or ONNX Runtime's quantization tooling.

Usage (from project root, with nnUNet env vars set as usual):

    python -m ONNX_quantization.export_onnx \
        --output mednext_task017_fp32.onnx

If you omit --output, the default is
ONNX_quantization/mednext_task017_fp32.onnx.
"""

import argparse
import os
from pathlib import Path

import torch

from batchgenerators.utilities.file_and_folder_operations import join

from nnunet_mednext.paths import network_training_output_dir
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

    print(f"[export_onnx] Loading FP32 trainer from {model_folder}")
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
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="mednext_task017_fp32.onnx",
        help="Output ONNX filename (relative to ONNX_quantization folder)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version to use (>=13 recommended)",
    )
    args = parser.parse_args()

    this_dir = Path(__file__).resolve().parent
    out_path = (this_dir / args.output).resolve()

    trainer = load_trainer()
    # For ONNX export we run everything on CPU to avoid unsupported
    # CUDA fallbacks such as slow_conv3d_forward on some clusters.
    net = trainer.network.cpu()

    # Build a dummy input with the correct shape: (N, C, D, H, W)
    # Use trainer.patch_size and the network's input channels.
    in_channels = getattr(net, "in_channels", None)
    if in_channels is None:
        # Fallback: assume 1 input channel for Task017 CT
        in_channels = 1

    patch_size = getattr(trainer, "patch_size", None)
    if patch_size is None:
        raise RuntimeError("trainer.patch_size is None; cannot build dummy input")

    dummy_input = torch.randn(1, in_channels, *patch_size, dtype=torch.float32)
    dummy_input = dummy_input.cpu()

    print(f"[export_onnx] Exporting with dummy input shape: {tuple(dummy_input.shape)}")
    print(f"[export_onnx] Writing ONNX model to {out_path}")

    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export the model. We keep a simple static-shape export because nnUNet
    # uses a fixed patch_size for inference.
    torch.onnx.export(
        net,
        dummy_input,
        str(out_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamic_axes=None,
    )

    print("[export_onnx] Export complete.")
    print(
        "You can now apply INT8 PTQ using TensorRT or ONNX Runtime on the "
        "exported ONNX model."
    )


if __name__ == "__main__":
    main()
