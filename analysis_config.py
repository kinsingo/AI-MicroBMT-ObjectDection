"""
=============================================================================
  Analysis Configuration  (analysis_config.py)
  ---------------------------------------------------------------------------
  All user-configurable settings for the analysis / visualization scripts.
  Edit ONLY this file when adapting the pipeline to a new benchmark run.
=============================================================================
"""

from pathlib import Path

# ============================================================================
# 1. HARDWARE / ACCELERATOR SETTINGS
# ============================================================================

# CPU baseline device name  (used as the reference for speedup & accuracy-drop)
BASELINE_DEVICE = 'Apple M4 CPU'

# Ordered list of ALL accelerators (CPU + NPUs)
ALL_ACCELERATORS = [
    'Apple M4 CPU',
    'Apple M4 ANE',
    'Mobilint-ARIES',
    'DeepX M1',
    'Hailo-8',
    #'Qualcomm QCS6490',
    'RTX PRO 6000 Max-Q',
]

# NPU-only accelerators (automatically excludes BASELINE_DEVICE)
NPU_ACCELERATORS = [acc for acc in ALL_ACCELERATORS if acc != BASELINE_DEVICE]

# Colour palette – consistent across every chart
ACCELERATOR_COLORS = {
    'Hailo-8':              '#1f77b4',   # Blue
    'DeepX M1':             '#ff7f0e',   # Orange
    'Mobilint-ARIES':       '#2ca02c',   # Green
    'Apple M4 ANE':         '#d62728',   # Red
    'Apple M4 CPU':         '#9467bd',   # Purple
    #'Qualcomm QCS6490':     '#8c564b',   # Brown
    'RTX PRO 6000 Max-Q':  '#e377c2',   # Pink
}

# ============================================================================
# 2. DATA FOLDER PATHS  (relative to the project root)
# ============================================================================

# YOLO Object Detection model variant folders  (single-stream & offline CSVs)
# key → folder name;  CSV filenames are derived as
#   "<folder>/<folder> single-stream results.csv"
#   "<folder>/<folder> offline results.csv"
DATA_FOLDERS = {
    'yolo11':   'yolo11 variant',
    'yolo12':   'yolo12 variant',
    'yolov5':   'yolov5 variant',
    'yolov7':   'yolov7 variant',
    'yolov8':   'yolov8 variant',
    'yolov9':   'yolov9 variant',
    'yolov10':  'yolov10 variant',
}

# ============================================================================
# 3. OUTPUT DIRECTORIES
# ============================================================================

BASE_OUTPUT_DIR = Path('analysis_charts')

# Sub-folder names (created automatically by each script)
OUTPUT_SUBDIR_SINGLESTREAM_VS_OFFLINE  = 'singleStream_vs_offline'

# ============================================================================
# 4. MODEL NAME NORMALISATION
# ============================================================================

# Suffixes stripped from raw benchmark_model names to create a canonical key.
# Add / remove entries here when on-boarding a new accelerator toolchain.
MODEL_NAME_STRIP_SUFFIXES = [
    '_bgr2rgb_normalized_quantized_model_compiled',
    '_trained_opset13',
    '_pretrained_opset13',
    '_pretrained_opset14',
    '_opset12',
    '_opset13',
    '_opset14',
    '_dynamic_batch',
]

# Width-multiplier tokens that can appear in a model name
WIDTH_MULTIPLIER_MAP = {
    'w0_25': 0.25,
    'w0_5':  0.5,
    'w0_75': 0.75,
    'w1_0':  1.0,
    'w1_5':  1.5,
    'w2_0':  2.0,
}

# ============================================================================
# 5. CASE CLASSIFICATION (generate_cases_analysis.py)
# ============================================================================

# NPUs used for case-classification analysis
CASE_ANALYSIS_NPUS = [
    'DeepX M1',
    'Mobilint-ARIES',
    'Hailo-8',
    #'Qualcomm QCS6490',
    'Apple M4 ANE',
    'RTX PRO 6000 Max-Q',
]

# Thresholds
CASE_TAU_ACCURACY_DROP_PCT   = 4.0    # below this → "accurate" (mAP50 drop %)
CASE_MIN_ABSOLUTE_ACCURACY   = 5.0    # below this → Case 4 (failure, mAP50 threshold)
CASE_TOTAL_MODELS            = 20     # total YOLO OD models in benchmark suite

# ============================================================================
# 8. MATPLOTLIB / SEABORN DEFAULTS
# ============================================================================

MPL_STYLE          = 'seaborn-v0_8-paper'
MPL_STYLE_FALLBACK = 'seaborn-paper'
SNS_PALETTE        = 'husl'
