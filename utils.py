"""
Utility functions for MLPerf benchmark data analysis.
Common data loading and preprocessing routines.

All user-configurable settings (colours, folder paths, device names, etc.)
live in ``analysis_config.py``.  Edit that file — not this one — when
adapting the pipeline to a new benchmark run.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import ALL settings from the central config
from analysis_config import (
    BASELINE_DEVICE,
    ALL_ACCELERATORS,
    NPU_ACCELERATORS,
    ACCELERATOR_COLORS,
    DATA_FOLDERS,
    MODEL_NAME_STRIP_SUFFIXES,
    WIDTH_MULTIPLIER_MAP,
    BASE_OUTPUT_DIR,
    MPL_STYLE,
    MPL_STYLE_FALLBACK,
    SNS_PALETTE,
)

# Set style for publication-quality figures
try:
    plt.style.use(MPL_STYLE)
except OSError:
    plt.style.use(MPL_STYLE_FALLBACK)
sns.set_palette(SNS_PALETTE)

# Base output directory (subfolders created by each script)
base_output_dir = BASE_OUTPUT_DIR
base_output_dir.mkdir(exist_ok=True)

# ============================================================================
# Model Name Processing Functions
# ============================================================================
def normalize_model_name(model_name):
    """Normalize model name to match across different accelerators.

    Strips the suffixes listed in ``analysis_config.MODEL_NAME_STRIP_SUFFIXES``.
    """
    normalized = model_name
    for suffix in MODEL_NAME_STRIP_SUFFIXES:
        normalized = normalized.replace(suffix, '')
    return normalized

def extract_width_multiplier(model_name):
    """Extract width multiplier (e.g., w0_25 → 0.25) from *model_name*.

    The recognised tokens are defined in ``analysis_config.WIDTH_MULTIPLIER_MAP``.
    """
    for token, value in WIDTH_MULTIPLIER_MAP.items():
        if token in model_name:
            return value
    return None

def extract_model_family(model_name):
    """Extract YOLO model family from model name."""
    name = model_name.lower()
    if 'yolo11' in name:
        if 'yolo11n' in name:
            return 'YOLO11-N'
        elif 'yolo11s' in name:
            return 'YOLO11-S'
        elif 'yolo11m' in name:
            return 'YOLO11-M'
        elif 'yolo11l' in name:
            return 'YOLO11-L'
        elif 'yolo11x' in name:
            return 'YOLO11-X'
        return 'YOLO11'
    elif 'yolo12' in name:
        if 'yolo12n' in name:
            return 'YOLO12-N'
        elif 'yolo12s' in name:
            return 'YOLO12-S'
        elif 'yolo12m' in name:
            return 'YOLO12-M'
        elif 'yolo12l' in name:
            return 'YOLO12-L'
        elif 'yolo12x' in name:
            return 'YOLO12-X'
        return 'YOLO12'
    elif 'yolov10' in name:
        if 'yolov10n' in name:
            return 'YOLOv10-N'
        elif 'yolov10s' in name:
            return 'YOLOv10-S'
        elif 'yolov10m' in name:
            return 'YOLOv10-M'
        elif 'yolov10l' in name:
            return 'YOLOv10-L'
        elif 'yolov10x' in name:
            return 'YOLOv10-X'
        return 'YOLOv10'
    elif 'yolov9' in name:
        if 'yolov9t' in name:
            return 'YOLOv9-T'
        elif 'yolov9s' in name:
            return 'YOLOv9-S'
        elif 'yolov9m' in name:
            return 'YOLOv9-M'
        elif 'yolov9c' in name:
            return 'YOLOv9-C'
        elif 'yolov9e' in name:
            return 'YOLOv9-E'
        return 'YOLOv9'
    elif 'yolov8' in name:
        if 'yolov8n' in name:
            return 'YOLOv8-N'
        elif 'yolov8s' in name:
            return 'YOLOv8-S'
        elif 'yolov8m' in name:
            return 'YOLOv8-M'
        elif 'yolov8l' in name:
            return 'YOLOv8-L'
        elif 'yolov8x' in name:
            return 'YOLOv8-X'
        return 'YOLOv8'
    elif 'yolov7' in name:
        if 'yolov7x' in name:
            return 'YOLOv7-X'
        elif 'yolov7w6' in name:
            return 'YOLOv7-W6'
        return 'YOLOv7'
    elif 'yolov5' in name:
        if 'yolov5nu' in name:
            return 'YOLOv5-NU'
        elif 'yolov5su' in name:
            return 'YOLOv5-SU'
        elif 'yolov5mu' in name:
            return 'YOLOv5-MU'
        elif 'yolov5lu' in name:
            return 'YOLOv5-LU'
        elif 'yolov5xu' in name:
            return 'YOLOv5-XU'
        return 'YOLOv5'
    return 'Unknown'

# ============================================================================
# Data Loading Functions
# ============================================================================
def load_single_stream_data():
    """Load and preprocess single-stream latency data for all model variants.

    Folder paths come from ``analysis_config.DATA_FOLDERS``.
    Accuracy metric: mAP50 (Object Detection).
    """
    dfs = []
    for folder in DATA_FOLDERS.values():
        csv_path = f'{folder}/{folder} single-stream results.csv'
        try:
            dfs.append(pd.read_csv(csv_path))
        except FileNotFoundError:
            print(f"  [WARNING] File not found: {csv_path}")
    if not dfs:
        raise FileNotFoundError("No single-stream data found. Run convert_and_evaluate.py first.")
    df = pd.concat(dfs, ignore_index=True)
    
    # Add preprocessing
    df['normalized_model'] = df['benchmark_model'].apply(normalize_model_name)
    df['width_multiplier'] = df['benchmark_model'].apply(extract_width_multiplier)
    df['model_family'] = df['benchmark_model'].apply(extract_model_family)
    
    # Get baseline mAP50 from the CPU baseline device (if available)
    baseline_df = df[df['accelerator_type'] == BASELINE_DEVICE][['normalized_model', 'mAP50']].copy()
    baseline_df.columns = ['normalized_model', 'baseline_mAP50']
    
    # Merge baseline mAP50 (left join — NaN if no CPU baseline available)
    df = df.merge(baseline_df, on='normalized_model', how='left')
    
    # Calculate mAP50 drop (only where CPU baseline exists)
    df['mAP50_drop'] = df['baseline_mAP50'] - df['mAP50']
    df['mAP50_drop_percent'] = (df['mAP50_drop'] / df['baseline_mAP50']) * 100
    
    # Convert latency
    df['latency_ms'] = df['sample_latency_average']
    
    print("[OK] Single-stream data loaded and preprocessed")
    print(f"Accelerator types: {df['accelerator_type'].unique()}")
    print(f"Model families: {df['model_family'].unique()}")
    print(f"Total data points: {len(df)}\n")
    
    return df

def load_offline_data():
    """Load and preprocess offline throughput data for all model variants.

    Folder paths come from ``analysis_config.DATA_FOLDERS``.
    """
    dfs = []
    for folder in DATA_FOLDERS.values():
        csv_path = f'{folder}/{folder} offline results.csv'
        try:
            dfs.append(pd.read_csv(csv_path))
        except FileNotFoundError:
            print(f"  [WARNING] File not found: {csv_path}")
    if not dfs:
        raise FileNotFoundError("No offline data found. Run convert_and_evaluate.py first.")
    df = pd.concat(dfs, ignore_index=True)
    
    # Add preprocessing
    df['normalized_model'] = df['benchmark_model'].apply(normalize_model_name)
    df['width_multiplier'] = df['benchmark_model'].apply(extract_width_multiplier)
    df['model_family'] = df['benchmark_model'].apply(extract_model_family)
    
    print("[OK] Offline data loaded and preprocessed")
    print(f"Accelerator types: {df['accelerator_type'].unique()}")
    print(f"Model families: {df['model_family'].unique()}")
    print(f"Total data points: {len(df)}\n")
    
    return df
