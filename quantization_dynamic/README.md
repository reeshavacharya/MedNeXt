# Dynamic PTQ + TensorRT Pipeline (quantization_dynamic)

This directory implements a **fully dynamic, post-training quantization (PTQ) pipeline** for the MedNeXt / nnUNet Task017 (BTCV) model, targeting **INT8 TensorRT inference on GPU**.

The end-to-end runtime pipeline is:

1. **FP32 model (nnUNet / MedNeXt)**
2. **Calibration** (`calibrate.py`)
3. **Weight quantization (+ optional bit-packing)** (`quantize.py` + `bitpacking.py` + `model_wrappers.py`)
4. **ONNX export** (`onnx_export.py`)
5. **TensorRT engine build** (`tensorrt_engine.py`)
6. **TensorRT inference (single-pass + sliding-window)** (`trt_inference.py`)

All of these steps can be driven from a **single CLI entrypoint**:

```bash
python -m quantization_dynamic.quantize \
    --quant_dtype int8 \
    --calibration_method minmax \
    --num_cases 10 \
    --output_dir dynamic_trt_results \
    --export_tensorrt_engine \
    --run_trt_inference
```

> NOTE: The pipeline is currently specialized to **INT8**, **min-max calibration**, and the **Task017 MedNeXt-S** nnUNet configuration. It uses the same nnUNet paths (`RESULTS_FOLDER`, `nnUNet_preprocessed`) as the rest of the project.

---

## 1. High-Level Pipeline

### 1.1. Conceptual Flow

1. **Load FP32 model**
   - Load the trained MedNeXt-S Task017, 3d_fullres, fold 0, from nnUNet results.
   - Put model in `eval()` mode and move to GPU (if available).

2. **Calibration (activations only)**
   - Build a simple calibration loader over preprocessed Task017 `.npz` volumes.
   - Use the same shapes / modalities as nnUNet inference (stage1 preprocessed data).
   - Attach `ActivationObserver` hooks via `CalibrationWrapper` to collect per-layer activation stats.
   - For each calibration volume, run a forward pass and accumulate:
     - Running min / max
     - Histograms and absolute values (for more advanced calibration methods)
   - Convert these activations into **quantization parameters** (`scale`, `zero_point`, `min_val`, `max_val`) per layer via `compute_quant_params_from_stats`.

3. **Weight quantization (+ optional bit-packing)**
   - Wrap the FP32 model in `QuantizedModelWrapper`.
   - For each quantizable layer (Conv / Linear):
     - Use the layer’s activation scale to quantize weights:
       - `q_w = round(w / scale)` clamped to INT8 range.
       - `w_deq = q_w * scale` (quantize–dequantize approximation in FP32).
     - Replace original weights with `w_deq` and store:
       - `weight_qint` (integer codes, on CPU)
       - `weight_scale` (scale, on CPU)
   - For INT6/INT4, optionally **bit-pack** integer weights into `int8` containers via `bitpacking.py`. (For now, the CLI keeps INT8 un-packed by default.)

4. **ONNX export**
   - Wrap the (quantized) model in `BaseModelWrapper` if needed.
   - Export a clean forward graph (no Python loops, no patch-based control flow) to ONNX using `torch.onnx.export`.
   - The default ONNX path is `quantization_dynamic/model_int8.onnx`.

5. **TensorRT engine build**
   - Load the ONNX model using TensorRT’s ONNX parser.
   - Create a TensorRT builder and network with explicit batch.
   - Configure an **INT8 engine** via `TensorRTConfig` (INT8 flag on, optional FP16/strict-types flags, and workspace size).
   - Build and serialize the engine to `quantization_dynamic/model_int8.engine`.

6. **TensorRT inference**
   - Load the `.engine` with TensorRT Runtime.
   - Create an execution context and discover input/output bindings.
   - Allocate GPU buffers with PyCUDA, copy input to device, execute, and copy back predictions.
   - Provide both **single-pass** and **sliding-window** inference helpers for 3D volumes.

---

## 2. CLI: `python -m quantization_dynamic.quantize`

The main orchestrator is the CLI defined in `quantize.py`. It wires together all other modules into a single command.

### 2.1. Basic Usage

```bash
cd /home/r/reeshav/MedNeXt
source .venv/bin/activate

# Make sure these nnUNet variables are set (as in slurm_quantize.sh):
export nnUNet_raw_data_base=/data/reeshav/MedNeXt_dataset/Abdomen
export nnUNet_preprocessed=/home/r/reeshav/MedNeXt/nnUNet_preprocessed
export RESULTS_FOLDER=/home/r/reeshav/MedNeXt/nnUNet_results

python -m quantization_dynamic.quantize \
    --quant_dtype int8 \
    --calibration_method minmax \
    --num_cases 10 \
    --output_dir dynamic_trt_results \
    --export_tensorrt_engine \
    --run_trt_inference
```

This will:

1. Load the FP32 MedNeXt-S Task017 model.
2. Run calibration on 10 preprocessed training cases using min-max.
3. Quantize the model weights to INT8.
4. Export the quantized model to ONNX at `quantization_dynamic/model_int8.onnx`.
5. Build an INT8 TensorRT engine at `quantization_dynamic/model_int8.engine` (if `--export_tensorrt_engine`).
6. Run a simple TensorRT inference sanity check on one calibration volume and print the output shape (if `--run_trt_inference`).

### 2.2. CLI Arguments

All arguments are parsed and handled in `quantize.py`.

#### `--quant_dtype`

- **Purpose:** Select the quantization integer precision.
- **Current valid values:**
  - `int8`
- **Default:** `int8`
- **Behavior:**
  - Controls the quantization range used by `get_quant_range`.
  - For now, the CLI is wired only for `int8`. The core library helpers support `int6`/`int4`, but the main pipeline is focused on INT8 TensorRT.

#### `--calibration_method`

- **Purpose:** Select the method used to derive activation ranges from collected statistics.
- **Current valid values in the CLI:**
  - `minmax`
- **Default:** `minmax`
- **Behavior:**
  - `minmax` uses the raw running min and max per layer.
  - The calibration core (`calibrate.py`) also implements:
    - `percentile`, `kl`, `mse`, `omse`, `aciq`, but these are not exposed via the CLI yet to keep the surface minimal.

#### `--per_channel`

- **Purpose:** Reserved flag for future use (per-channel activation scales).
- **Type:** Boolean flag.
- **Default:** `False`.
- **Current behavior:**
  - Parsed but not actively used in the pipeline; all activation scales are effectively per-layer.
  - Included to keep the CLI shape compatible with possible future extensions.

#### `--num_cases`

- **Purpose:** Number of calibration volumes to use.
- **Type:** Integer.
- **Default:** `10`.
- **Valid range:**
  - Any positive integer up to the number of available preprocessed cases.
- **Behavior:**
  - Determines how many `.npz` files from the Task017 stage1 preprocessed directory are used in calibration.
  - Calibration files are taken from:
    - `${nnUNet_preprocessed}/Task017_AbdominalOrganSegmentation/nnUNetData_plans_v2.1_trgSp_1x1x1_stage1`.

#### `--output_dir`

- **Purpose:** Base directory for pipeline artifacts.
- **Type:** String (path).
- **Default:** `dynamic_trt_results`.
- **Behavior:**
  - Ensures the directory exists (`os.makedirs(..., exist_ok=True)`).
  - Intended location to store additional artifacts (quant params, logs, etc.).
  - Current pipeline uses fixed ONNX / engine paths; you can adapt them to be inside this folder if desired.

#### `--onnx_path`

- **Purpose:** Path where the exported ONNX model will be stored.
- **Type:** String (path).
- **Default:** `quantization_dynamic/model_int8.onnx`.
- **Behavior:**
  - Passed directly to `export_quantized_onnx`.
  - Must be writable by the current user.

#### `--engine_path`

- **Purpose:** Path where the TensorRT engine will be saved.
- **Type:** String (path).
- **Default:** `quantization_dynamic/model_int8.engine`.
- **Behavior:**
  - Used by `build_tensorrt_engine` for engine serialization.
  - Must be writable by the current user.

#### `--export_tensorrt_engine`

- **Purpose:** Toggle building the TensorRT engine from the ONNX model.
- **Type:** Boolean flag (no argument); presence enables it.
- **Default:** Disabled (only ONNX export will occur if not set).
- **Behavior:**
  - When present, after ONNX export the pipeline will:
    - Create a default `TensorRTConfig` (INT8 enabled, 4 GB workspace).
    - Call `build_tensorrt_engine(onnx_path, engine_path, config)`.

#### `--run_trt_inference`

- **Purpose:** Run a quick TensorRT inference sanity check after building the engine.
- **Type:** Boolean flag.
- **Default:** Disabled.
- **Behavior:**
  - When present, the pipeline:
    - Creates a `TensorRTInferenceSession` from `engine_path`.
    - Loads one calibration `.npz` volume, slices channels to `stem_in_ch` if needed.
    - Forms a batch of shape `(1, C, D, H, W)` and calls `infer`.
    - Logs the output shape.
  - This does **not** perform full evaluation or metrics; it is for shape / sanity verification.

---

## 3. Module-by-Module Overview

### 3.1. `quant_utils.py`

**Role:** Central collection of quantization math, configs, and calibration helpers.

Key components:

- **Supported dtypes:**
  - `SUPPORTED_QUANT_DTYPES = ("int8", "int6", "int4")`.
  - `get_quant_range(quant_dtype)` returns `(qmin, qmax)` for each.
- **Calibration methods:**
  - `SUPPORTED_CALIBRATION_METHODS = ("minmax", "percentile", "kl", "mse", "omse", "aciq")`.
  - `normalize_calibration_method(method)` validates and normalizes strings.
- **Configs:**
  - `PatchSettings` / `default_patch_settings()` – for patch-based inference (used indirectly).
  - `TensorRTConfig` / `default_tensorrt_config()` – toggles INT8/FP16/strict types and workspace size.
  - `OnnxExportConfig` / `default_onnx_export_config()` – standard ONNX export options.
- **Quantization math:**
  - `compute_scale(min_val, max_val, qmax)` – symmetric scale.
  - `compute_zero_point(min_val, max_val, scale, qmin, qmax, symmetric=True)` – zero-point.
  - `quantize_tensor(tensor, scale, qmin, qmax, zero_point=0, dtype=...)`.
  - `dequantize_tensor(q_tensor, scale, zero_point=0)`.
  - `compute_channel_min_max(weight, channel_axis=0)` + `compute_channel_scales(...)` for per-channel scales.
- **Clipping & errors:**
  - `apply_percentile_clipping(values, percentile)`.
  - `mse_reconstruction_error`, `omse_reconstruction_error`.
- **Advanced thresholds:**
  - `compute_kl_threshold(values, num_bins, num_quant_bins)` for KL-based calibration.
  - `compute_aciq_threshold(values, num_bits)` for ACIQ-based calibration.

### 3.2. `bitpacking.py`

**Role:** Pack low-bit integer codes into `int8` containers for storage or transport.

- **INT6:**
  - `pack_int6(tensor)` – clamp to `[-32, 31]`, store lower 6 bits in `int8`.
  - `unpack_int6(packed)` – recover logical INT6 values.
- **INT4:**
  - `pack_int4(tensor, dim=-1)` – clamp to `[-8, 7]`, pack two values into one byte along `dim`.
  - `unpack_int4(packed, dim=-1, orig_length=None)` – inverse operation, re-expanding to original length (respecting `orig_length`).

> Note: Bit-packing is primarily used for simulated INT6/INT4 storage. For INT8, the main pipeline leaves weights in standard `int8` tensors.

### 3.3. `activation_observer.py`

**Role:** Collect activation statistics during calibration via forward hooks.

- **`LayerStats` dataclass:**
  - `min_val`, `max_val` – running min/max.
  - `abs_values` – list of absolute-value arrays for percentile and advanced methods.
  - `hist`, `hist_range` – histogram counts and range for |activations|.
- **`ActivationObserver` class:**
  - `register_hooks(model)`:
    - Attaches forward hooks to Conv1/2/3d, Linear, BatchNorm1/2/3d, ReLU, LeakyReLU.
  - Hook behavior:
    - Detach, move to CPU, flatten to NumPy.
    - Update min/max.
    - Append `np.abs(flat)` to `abs_values`.
    - Update or lazily rebuild histograms over absolute activations.
  - `reset()` – clear stats and remove hooks.
  - `get_stats()` – return a dictionary mapping layer names to:
    - `{"values", "histogram", "hist_range", "min", "max"}`.

### 3.4. `model_wrappers.py`

**Role:** Provide standardized wrappers around the base model for different stages.

- **`BaseModelWrapper`**
  - Thin `nn.Module` wrapper; ensures:
    - `model.eval()` is called.
    - Inputs `(C,D,H,W)` or `(N,C,D,H,W)` are normalized to `(N,C,D,H,W)`.
    - Moves tensors to the same device as the model.
- **`CalibrationWrapper`**
  - Inherits from `BaseModelWrapper`.
  - Integrates `ActivationObserver`:
    - `__init__` registers hooks on the underlying model.
    - `reset_observer()`, `get_activation_stats()`, `remove_hooks()` for managing statistics.
- **`QuantizedModelWrapper`**
  - Inherits from `BaseModelWrapper`.
  - On construction:
    - Validates `quant_dtype` against `SUPPORTED_QUANT_DTYPES`.
    - Stores `quant_params` (per-layer quantization parameters).
    - Calls `_apply_weight_quantization()` to quantize–dequantize weights and record `weight_qint` and `weight_scale` buffers.
- **`PatchInferenceWrapper`**
  - Inherits from `BaseModelWrapper`.
  - Implements sliding-window inference for 3D volumes:
    - `generate_patches(volume)` yields `(patch, (z,y,x))` for `(C,D,H,W)` inputs.
    - `forward(x)` averages overlapping patch predictions into a full `(C_out,D,H,W)` output.
  - Useful when performing patch-based calibration or baseline FP32 inference.

### 3.5. `calibrate.py`

**Role:** Convert activation statistics into quantization parameters.

- **Core functions:**
  - `_compute_range_minmax`, `_compute_range_percentile`, `_compute_range_kl`, `_compute_range_aciq` – compute `(min_val,max_val)` from a layer’s stats.
  - `_select_range_for_layer` – choose the appropriate range computation based on `calibration_method`.
  - `compute_quant_params_from_stats(activation_stats, quant_dtype, calibration_method, percentile, symmetric)` – returns:
    - `{layer_name: {"scale", "zero_point", "min_val", "max_val"}}`.
  - `run_calibration(model, calib_loader, quant_dtype, calibration_method, percentile, symmetric, device)`:
    - Wraps `model` in `CalibrationWrapper`.
    - Iterates over `calib_loader`, running forwards to collect stats.
    - Calls `compute_quant_params_from_stats` and returns `quant_params`.

### 3.6. `quantize.py`

**Role:**

1. **Library helper**: `quantize_model(model, quant_params, ...)`.
2. **CLI orchestrator**: `main()` to run the full pipeline.

- **`quantize_model`**
  - Inputs: FP32 `model`, `quant_params`, `quant_dtype`, optional bit-packing options.
  - Behavior:
    - Wraps model in `QuantizedModelWrapper`.
    - If `quant_dtype` is `int6`/`int4` and `pack_weights=True`, packs `weight_qint` using `bitpacking.py`.
    - Optionally saves the quantized model `state_dict`.

- **CLI helpers:**
  - `_get_model_folder()` – find nnUNet training folder for Task017 MedNeXt-S.
  - `_get_calibration_files(max_cases)` – list `.npz` calibration files from Task017 preprocessed stage1.
  - `_build_calibration_loader(calib_files, stem_in_ch, device)` – yield `(C,D,H,W)` volumes as torch tensors.
  - `_load_fp32_model(device)` – load trainer and network in FP32 for fold 0.
  - `_get_stem_in_channels(net)` – optional detection of `net.stem.in_channels`.
  - `_parse_args()` – define CLI interface.
  - `main()` – orchestrate all steps (see section 2).

### 3.7. `onnx_export.py`

**Role:** Export a (quantized) PyTorch model to ONNX with a clean forward graph.

- **`export_quantized_onnx(model, output_path, dummy_input_shape, config)`**
  - Ensures `model` is wrapped in `BaseModelWrapper`.
  - Moves to the appropriate device.
  - Creates a random dummy input of the given shape `(N,C,D,H,W)`.
  - Calls `torch.onnx.export` with:
    - Opset 17.
    - Constant folding (by default).
    - Configurable input/output names and dynamic axes.

### 3.8. `tensorrt_engine.py`

**Role:** Build an optimized TensorRT engine from an ONNX file.

- **`build_tensorrt_engine(onnx_path, engine_path, config)`**
  - Loads the ONNX model from disk.
  - Creates TensorRT builder, network (explicit batch), and ONNX parser.
  - Parses the model and checks for errors.
  - Configures builder flags:
    - `INT8` (enabled by default in `default_tensorrt_config`).
    - Optional `FP16`, `STRICT_TYPES` if set in `TensorRTConfig`.
    - `max_workspace_size_bytes`.
  - Builds the engine and serializes it to `engine_path`.

### 3.9. `trt_inference.py`

**Role:** Run inference with the serialized TensorRT engine.

- **`TensorRTInferenceSession`**
  - Constructor:
    - Loads serialized engine from `engine_path`.
    - Creates TensorRT runtime and execution context.
    - Finds a single input and output binding.
    - Creates a CUDA stream via PyCUDA.
  - `infer(input_array)`:
    - Expects a NumPy array `(N,C,D,H,W)` (float32 preferred).
    - Handles dynamic shapes by setting the binding shape from input.
    - Allocates device buffers, copies input to GPU, executes, and copies output back.
    - Returns predictions reshaped to binding output shape.
  - `sliding_window_inference(volume, patch_size, stride)`:
    - Accepts `(C,D,H,W)` or `(1,C,D,H,W)`.
    - Probes the engine with a central patch to infer `C_out`.
    - Slides a window across the volume, averaging overlapping predictions.
    - Returns `(C_out,D,H,W)` prediction volume.

### 3.10. `slurm_quantize.sh`

**Role:** SLURM script to launch the entire pipeline on a GPU node.

- Configures resources (1 node, 1 GPU, 8 CPUs, 200 GB RAM).
- Activates the project virtualenv.
- Exports `nnUNet_raw_data_base`, `nnUNet_preprocessed`, `RESULTS_FOLDER`.
- Logs GPU info via `nvidia-smi`.
- Calls:

  ```bash
  srun python -m quantization_dynamic.quantize \
      --quant_dtype int8 \
      --calibration_method minmax \
      --num_cases 10 \
      --output_dir dynamic_trt_results \
      --export_tensorrt_engine \
      --run_trt_inference \
      --onnx_path quantization_dynamic/model_int8.onnx \
      --engine_path quantization_dynamic/model_int8.engine
  ```

- Verifies presence of ONNX and engine files at the end.

---

## 4. Assumptions & Requirements

- **Environment:**
  - CUDA-capable GPU (e.g., A40) with TensorRT installed.
  - PyCUDA installed and compatible with your Python version (patched if necessary).
  - PyTorch + nnUNet / MedNeXt dependencies available in the active virtualenv.
- **nnUNet paths:**
  - `RESULTS_FOLDER` must point to a directory containing:
    - `nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1`.
  - `nnUNet_preprocessed` must contain preprocessed Task017 data under:
    - `Task017_AbdominalOrganSegmentation/nnUNetData_plans_v2.1_trgSp_1x1x1_stage1`.
- **Data format:**
  - Calibration and TRT sanity check volumes are taken from `.npz` files with key `"data"`, matching nnUNet conventions.

With these in place, the `quantization_dynamic` package provides a self-contained, reproducible path from a trained FP32 MedNeXt model to an INT8 TensorRT engine and basic GPU inference on 3D medical volumes.
