# AI-MicroBMT: Reproducibility Artifacts

**Companion code for the paper:**  
> *AI-MicroBMT: Microbenchmark-Driven Quantization-Aware Performance Characterization of NPUs*  
> KDD 2026 — Dataset & Benchmark Track

This repository contains the complete pipeline to **reproduce all scores, rankings, and figures** presented in the paper. It is organized in two parts:

1. **UDS Scoring** — computes the Unified Deployment Score (S1–S7) and weighted rankings from raw benchmark CSVs.
2. **Analysis & Visualization** — generates all charts, heatmaps, and LaTeX tables appearing in the paper.

> **Task**: Object Detection — accuracy metric is **mAP50** (replaces Top-1 accuracy used in classification benchmarks).  
> **Models**: 20 YOLO variants across 7 families (YOLO11, YOLO12, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10).

---

## Table of Contents

- [Requirements](#requirements)
- [Evaluation Guide — Raw Data Collection](#evaluation-guide--raw-data-collection)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Part 1 — UDS Scoring Pipeline](#part-1--uds-scoring-pipeline)
  - [Sub-Scores (S1–S7)](#sub-scores-s1s7)
  - [Input Data Format](#input-data-format)
  - [Configuration (Scoring)](#configuration-scoring)
  - [Output Files (Scoring)](#output-files-scoring)
- [Part 2 — Analysis & Visualization](#part-2--analysis--visualization)
  - [Configuration (Analysis)](#configuration-analysis)
  - [Output Files (Analysis)](#output-files-analysis)
- [Customization Guide](#customization-guide)
  - [Adding a New Device](#adding-a-new-device)
  - [Adding a New Model Family](#adding-a-new-model-family)
  - [Defining Weight Profiles](#defining-weight-profiles)
  - [Tuning Feasibility Gate Parameters](#tuning-feasibility-gate-parameters)
  - [Analysis Config Reference](#analysis-config-reference)
- [Example End-to-End Workflow](#example-end-to-end-workflow)
- [License](#license)

---

## Requirements

```
python >= 3.8
pandas
numpy
matplotlib
seaborn
```

```bash
pip install pandas numpy matplotlib seaborn
```

---

## Evaluation Guide — Raw Data Collection

Before running any analysis scripts, you must first **collect raw benchmark data** by evaluating your target devices with the AI-BMT program. Follow the steps below.

### Step 0 — Download the AI-BMT Program

1. Visit **[https://www.ai-bmt.com/](https://www.ai-bmt.com/)** and download the **User Manual**.
   - Alternatively, access the manual directly from [https://github.com/kinsingo/SNU_BMT_DOCX](https://github.com/kinsingo/SNU_BMT_DOCX).
2. Refer to the manual and download the **AI-BMT application** that matches your evaluation environment:
   - **Architecture**: x64 / ARM64
   - **OS**: Windows, Linux, or macOS
3. Install and launch the AI-BMT program.

### Step 1 — Download Models & Datasets

Open the AI-BMT application and navigate to the **Data Download** tab, then download the required models and datasets as shown below:

![Download Models & Datasets](download_model_dataset.png)

> **Tip**: Repeat for other task types (Object Detection, Semantic Segmentation, LLM Tasks) if your evaluation scope extends beyond classification.

### Step 2 — Run Benchmarks on Your Devices

1. Follow the User Manual instructions to configure and execute benchmarks on each target device.
2. Your evaluation data will be **automatically uploaded** to the server once each evaluation is finished. You can then **download the results as CSV** from the database — either through the AI-BMT application or the web interface (refer to the User Manual for details).
3. Organize the downloaded CSVs into the folder structure expected by the analysis pipeline (see [Repository Structure](#repository-structure)).

### Reference: Accelerator Evaluation Scripts

The `accelerator_AIBMT_evalution_scripts/` folder contains **C++ based example evaluation scripts** for the hardware platforms used in the paper:

```
accelelerator_AIBMT_evalution_scripts/
├── Apple M4 (CPU/ANE)/
├── DeepX M1/
├── Hailo-8/
├── Mobilint-ARIES/
├── RTX PRO 6000 Max-Q/
└── Rubic pi 3 (QCS6490)/
```

Use these scripts as reference when integrating a new accelerator into the benchmark pipeline.

---

## Quick Start

```bash
# --- Step 0: Convert raw CSV (if starting from AI-BMT platform export) ---
python convert_and_evaluate.py --input newEvalResults.csv --append

# --- Part 1: UDS Scoring ---
python "1. Create UDS Scores.py"   # Compute sub-scores S1~S7
python "2. UDS cases.py"           # Weighted rankings for 16 use-case profiles

# --- Part 2: Analysis & Visualization ---
python analyze_results_singleStream_offline.py
python generate_cases_analysis.py
```

---

## Repository Structure

```
.
├── convert_and_evaluate.py                         # Step 0: raw CSV → variant folder conversion
├── 1. Create UDS Scores.py                        # Step 1: sub-score calculation (S1–S7)
├── 2. UDS cases.py                                # Step 2: weighted UDS rankings
├── analysis_config.py                             # Centralized config for analysis scripts
├── utils.py                                       # Shared utilities (data loading, normalization)
├── analyze_results_singleStream_offline.py        # Radar charts + mAP50 drop heatmap
├── generate_cases_analysis.py                     # Case classification (1–4) + LaTeX tables
│
├── <yolo_family> variant/                         # Benchmark CSVs per YOLO model family
│   ├── <yolo_family> variant single-stream results.csv
│   └── <yolo_family> variant offline results.csv
│
├── result_table.csv                               # Source: raw AI-BMT evaluation export
│
└── analysis_charts/                               # Output: charts & CSV summaries
    ├── singleStream_vs_offline/
    └── UDS metrics/
```

---

## Part 0 — Raw Data Conversion (`convert_and_evaluate.py`)

If your benchmark results are in the **raw CSV format** exported from the AI-BMT platform (e.g., `newEvalResults.csv`), use `convert_and_evaluate.py` to convert them into the variant-folder CSV format required by the scoring and analysis scripts.

### What it does

1. **Loads** the raw CSV (60+ columns) and extracts the 6–7 essential columns
2. **Normalizes** model names (strips `_trained_opset13`, `_pretrained_opset14`, etc.)
3. **Normalizes** device names (e.g., `NVIDIA Jetson Orin` from verbose accelerator strings)
4. **Classifies** each model into: base variant, activation variant, or input-resolution variant
5. **Splits** by scenario (Single-Stream / Offline) into separate CSV files
6. **Writes** output CSVs into the correct `<family> variant/` folders
7. **Prints** an evaluation summary (latency, throughput, accuracy statistics)
8. **Checks** pipeline compatibility and lists required config changes for new devices

### Usage

```bash
# Basic: convert default newEvalResults.csv (overwrites existing data for same device)
python convert_and_evaluate.py

# Append to existing variant CSVs (preserves data from other devices)
python convert_and_evaluate.py --append

# Use a different input file
python convert_and_evaluate.py --input my_results.csv --append

# Override the device name in the output
python convert_and_evaluate.py --device-name "My Custom Device" --append
```

### Command-Line Options

| Flag | Description |
|------|-------------|
| `--input`, `-i` | Input CSV file path (default: `newEvalResults.csv`) |
| `--device-name`, `-d` | Override `accelerator_type` to this name |
| `--append`, `-a` | Append to existing CSVs instead of overwriting |
| `--output-dir`, `-o` | Base output directory (default: current directory) |
| `--no-summary` | Skip the evaluation summary printout |
| `--no-compat-check` | Skip the pipeline compatibility check |

### Pipeline Compatibility

After conversion, the script automatically checks whether the device(s) in the data are registered in the pipeline configuration. If a new device is detected, it prints the exact config changes you need to make:

- **`analysis_config.py`**: `ALL_ACCELERATORS`, `ACCELERATOR_COLORS`, `CASE_ANALYSIS_NPUS`
- **`1. Create UDS Scores.py`**: `HARDWARE_POWER`, `HARDWARE_PEAK_COMPUTE`, `normalize_device_name()`

> **Note**: For devices already registered (e.g., Hailo-8, Apple M4 CPU), no additional config changes are needed — just run `convert_and_evaluate.py --append` and proceed to scoring.

### Device Name Normalization

The script automatically normalizes common device name patterns:

| Raw CSV Value | Normalized Name |
|---------------|-----------------|
| `CUDA 12.6 + cuDNN ... ONNX Runtime ...` (with "jetson" or "orin") | `NVIDIA Jetson Orin` |
| `hailo-8`, `Hailo 8` | `Hailo-8` |
| `deepx`, `DeepX-M1` | `DeepX M1` |
| `qualcomm`, `QCS6490` | `Qualcomm QCS6490` |

You can add more patterns in `convert_and_evaluate.py` → `DEVICE_NAME_NORMALIZE`.

---

## Part 1 — UDS Scoring Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 0 | `convert_and_evaluate.py` | Converts raw AI-BMT CSV to variant folder format |
| 1 | `1. Create UDS Scores.py` | Computes 7 individual sub-scores (S1–S7) from raw benchmark data |
| 2 | `2. UDS cases.py` | Applies user-defined weight profiles to produce weighted UDS rankings |

### Sub-Scores (S1–S7)

| Score | Name | Description |
|-------|------|-------------|
| S1 | Coverage | Fraction of YOLO models that pass the feasibility gate |
| S2 | Efficiency | Average latency speedup vs. CPU baseline (log-scaled) |
| S3 | Scaling | N/A for Object Detection (always 0; no resolution sweep) |
| S4 | Accuracy Retention | How well mAP50 is preserved after quantization/compilation |
| S5 | Throughput Gain | Offline throughput improvement vs. CPU (log-scaled) |
| S6 | Peak Compute Efficiency | Throughput normalized by vendor-reported peak TOPS (optional) |
| S7 | Power Efficiency | Throughput normalized by device power consumption (optional) |

### Input Data Format

#### Single-Stream CSV (latency measurement)

| Column | Type | Description |
|--------|------|-------------|
| `task` | str | Must be "Object Detection" |
| `scenario` | str | Must be "Single-Stream" |
| `mAP50` | float | mAP@0.5 on the device (%) |
| `sample_latency_average` | float | Average inference latency (ms) |
| `accelerator_type` | str | Device name (e.g., "Hailo-8", "Apple M4 CPU") |
| `benchmark_model` | str | Model variant name (e.g., "yolo11m") |

#### Offline CSV (throughput measurement)

| Column | Type | Description |
|--------|------|-------------|
| `task` | str | Must be "Object Detection" |
| `scenario` | str | Must be "Offline" |
| `mAP50` | float | mAP@0.5 on the device (%) |
| `samples_per_second` | float | Throughput (samples/sec) |
| `accelerator_type` | str | Device name |
| `benchmark_model` | str | Model variant name |

> **Note**: One device can optionally serve as the **CPU baseline** (default: `Apple M4 CPU`). If present, it is used to compute relative mAP50 drops and speedups for S2/S4/S5. If absent, S2/S4/S5 are computed where possible, and the feasibility gate falls back to the absolute mAP50 threshold (`A_MIN`).

### Configuration (Scoring)

Open `1. Create UDS Scores.py` and locate the **USER-CONFIGURABLE PARAMETERS** section near the top.

**Feasibility Gate:**

```python
TAU_ACC = 4.0       # Max tolerable mAP50 drop (%) from CPU baseline
THETA_SPEEDUP = 1.0 # Min required speedup vs CPU (1.0 = at least as fast)
A_MIN = 5.0         # Min absolute mAP50 (%) to filter silent failures
```

**Data Folders:**

```python
DATA_FOLDERS_BASE = [
    'yolo11 variant',
    'yolo12 variant',
    'yolov5 variant',
    'yolov7 variant',
    'yolov8 variant',
    'yolov9 variant',
    'yolov10 variant',
    # ... add new YOLO family folders here
]
```

**Hardware Specs (for S6 & S7):**

```python
HARDWARE_POWER = {
    'Hailo-8': 8.65,
    'Mobilint-ARIES': 25.0,
    'Your-Device': 10.0,       # <-- Add your device
    'Apple M4 CPU': None,       # SoC: not separately measurable
}
HARDWARE_PEAK_COMPUTE = {
    'Hailo-8': 26.0,
    'Mobilint-ARIES': 80.0,
    'Your-Device': 50.0,       # <-- Add your device
    'Apple M4 CPU': None,
}
```

Devices with `None` will have S6/S7 = N/A (excluded from extended UDS profiles).

### Output Files (Scoring)

| File | Description |
|------|-------------|
| `analysis_charts/UDS metrics/UDS_scores_summary.csv` | Per-device sub-scores S1–S7 with metadata (input for Step 2) |
| `analysis_charts/UDS metrics/UDS_scores_detailed_base.csv` | Per-task metrics for all YOLO models |
| `UDS_cases_results.csv` | Full device rankings for every weight profile |
| `UDS_cases_winners.csv` | Winner (rank 1) device per weight profile |
| `UDS_weight_profiles.csv` | Weight definitions for all profiles |

---

## Part 2 — Analysis & Visualization

All visualization scripts read settings from a single centralized file:

| File | Description |
|------|-------------|
| `analysis_config.py` | **All user-configurable settings** — edit this file only |
| `utils.py` | Shared utilities (data loading, model name normalization, color palette) |
| `analyze_results_singleStream_offline.py` | Radar charts (3 scalings) + mAP50 drop heatmap |
| `generate_cases_analysis.py` | Case classification (1–4) + LaTeX table generation |

> **Note**: `analyze_activation_sweep.py` and `analyze_results_input_resolution_*.py` are not applicable for the Object Detection benchmark and have been removed.

### Configuration (Analysis)

All settings live in `analysis_config.py`. Edit only that file when adapting the analysis to new data.

See the [Analysis Config Reference](#analysis-config-reference) section below for details.

### Output Files (Analysis)

```
analysis_charts/
└── singleStream_vs_offline/
    ├── combined_radar_chart_linear.png
    ├── combined_radar_chart_sqrt.png
    ├── combined_radar_chart_log10.png
    ├── mAP50_drop_heatmap.png
    └── *.csv  (throughput, heatmap data)
```

---

## Customization Guide

### Adding a New Device

1. In `1. Create UDS Scores.py`:
   - Add to `HARDWARE_POWER` and `HARDWARE_PEAK_COMPUTE` (set `None` if unknown)
2. In `analysis_config.py`:
   - Add to `ALL_ACCELERATORS` (order = chart display order)
   - Add a color entry to `ACCELERATOR_COLORS`

### Adding a New Model Family

1. Create a folder (e.g., `efficientnet variant/`) containing:
   - `efficientnet variant single-stream results.csv`
   - `efficientnet variant offline results.csv`
2. In `1. Create UDS Scores.py`: add `'efficientnet variant'` to `DATA_FOLDERS_BASE`
3. In `analysis_config.py`: add an entry to `DATA_FOLDERS`

### Defining Weight Profiles

In `2. UDS cases.py`:

```python
# UDS Basic profiles (S1~S5 only, weights must sum to 1.0)
UDS_profiles = {
    "UDS_AccuracyStrict": {"S1":0.15, "S2":0.20, "S3":0.15, "S4":0.30, "S5":0.20},
    "UDS_LatencyFirst":   {"S1":0.15, "S2":0.30, "S3":0.15, "S4":0.20, "S5":0.20},
    "My_Custom_Profile":  {"S1":0.10, "S2":0.10, "S3":0.10, "S4":0.40, "S5":0.30},
}

# UDS Extended profiles (S1~S7, weights must sum to 1.0)
EXT_profiles = {
    "EXT_AccStrict_Both": {"S1":0.13,"S2":0.13,"S3":0.13,"S4":0.18,"S5":0.13,"S6":0.15,"S7":0.15},
}
```

> **Rule**: All weights within a profile must sum to **1.0**.

Profiles with `_with_fixed_inputRes` suffix set `S3 = 0` (not applicable for Object Detection; S3 is always 0 in this benchmark).

### Tuning Feasibility Gate Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `TAU_ACC` | 4.0 | Max tolerable mAP50 drop (%) from CPU baseline. |
| `THETA_SPEEDUP` | 1.0 | Min required speedup vs CPU. Set >1 for stricter latency requirements. |
| `A_MIN` | 5.0 | Min absolute mAP50 (%). Filters out trivially bad compilation results. |

### Analysis Config Reference

`analysis_config.py` is organized in 6 sections:

| # | Section | Key Variables |
|---|---------|---------------|
| 1 | Devices | `BASELINE_DEVICE`, `ALL_ACCELERATORS`, `ACCELERATOR_COLORS` |
| 2 | Data Folders | `DATA_FOLDERS` (7 YOLO variant folders) |
| 3 | Output Dirs | `BASE_OUTPUT_DIR`, `OUTPUT_SUBDIR_SINGLESTREAM_VS_OFFLINE` |
| 4 | Model Normalization | `MODEL_NAME_STRIP_SUFFIXES` |
| 5 | Case Classification | `CASE_ANALYSIS_NPUS`, `CASE_TAU_ACCURACY_DROP_PCT`, `CASE_MIN_ABSOLUTE_ACCURACY`, `CASE_TOTAL_MODELS` |
| 6 | Plot Defaults | `MPL_STYLE`, `SNS_PALETTE` |

**To add a new accelerator:** add to `ALL_ACCELERATORS` + `ACCELERATOR_COLORS`.  
**To add a new model family:** add to `DATA_FOLDERS` (CSV file names must follow `<folder_name> single-stream results.csv` / `<folder_name> offline results.csv`).  
**To onboard a new accelerator toolchain:** add its model-name suffix to `MODEL_NAME_STRIP_SUFFIXES`.

---

## Example End-to-End Workflow

1. **Run benchmarks** on your devices using the AI-BMT framework
2. **Download CSV** from the AI-BMT platform (e.g., `newEvalResults.csv`)
3. **Convert data**: `python convert_and_evaluate.py --input newEvalResults.csv --append`
4. **Review compatibility output**: the script will list any required config changes
5. **(If new device)** Edit `1. Create UDS Scores.py`: add your device to `HARDWARE_POWER` / `HARDWARE_PEAK_COMPUTE` / `normalize_device_name()`
6. **(If new device)** Edit `analysis_config.py`: add your device to `ALL_ACCELERATORS`, `ACCELERATOR_COLORS`, and optionally `CASE_ANALYSIS_NPUS`
7. **Run scoring**: `python "1. Create UDS Scores.py"` then `python "2. UDS cases.py"`
8. **Run analysis**: execute the 2 analysis scripts listed in [Quick Start](#quick-start)
9. **Review** output CSVs and charts in `analysis_charts/`

---

## License

This repository is part of the AI-MicroBMT evaluation framework.
