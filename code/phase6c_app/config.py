"""
Phase 6C: Desktop Reference-BP Validation & Prediction Application Configuration
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only
"""

import sys
from pathlib import Path

# Base Paths
APP_DIR = Path(__file__).resolve().parent
CODE_DIR = APP_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

# Ensure project code is on sys.path
for p in [CODE_DIR, CODE_DIR / "scripts", CODE_DIR / "phase4a"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Frozen Checkpoints and Artifacts
PHASE4A_CKPT = CODE_DIR / "outputs" / "phase4a_single_model" / "checkpoints" / "best_model_ppg_vpg_apg.pt"
PHASE4B_CKPT = CODE_DIR / "outputs" / "phase4b_temporal_gru" / "checkpoints" / "best_temporal_gru.pt"
PHASE5C_DIR = CODE_DIR / "outputs" / "phase5c_extreme_aware"
ISOTONIC_SBP_PKL = PHASE5C_DIR / "mappings" / "isotonic_sbp.pkl"
ISOTONIC_DBP_PKL = PHASE5C_DIR / "mappings" / "isotonic_dbp.pkl"
CONFORMAL_QUANTILES_JSON = PHASE5C_DIR / "calibration" / "conformal_quantiles_by_bin.json"

# Default Output Paths
DEFAULT_OUTPUT_DIR = CODE_DIR / "outputs" / "phase6c_reference_validation"

# Pipeline Timing and Sampling Parameters
INPUT_NOMINAL_FS = 100.0  # Hardware acquisition nominal rate (~100 Hz)
OUTPUT_FS = 125.0         # Research model sampling rate (125 Hz)
RESAMPLE_UP = 5
RESAMPLE_DOWN = 4
WINDOW_SECONDS = 10.0
WINDOW_SAMPLES = int(OUTPUT_FS * WINDOW_SECONDS)  # 1250 samples
SEQUENCE_WINDOWS = 6
SEQUENCE_SECONDS = 60.0
CHUNK_SIZE = 10           # 10 samples per chunk (~100 ms)

# Hardware and ADC Bounds
ADC_MAX = 262143.0        # MAX30102 18-bit ADC limit
ADC_MIN = 0.0

# Synchronization & Pairing Parameters
DEFAULT_PAIRING_TOLERANCE_S = 15.0  # Default maximum tolerance between BP event and prediction timestamp

# Model Invariants (Frozen Checkpoints)
EXPECTED_CNN_PARAMS = 146978
EXPECTED_GRU_PARAMS = 27106
EXPECTED_TOTAL_PARAMS = 174084

# Scientific Safeguards and Disclaimers
DISCLAIMER_TEXT = (
    "RESEARCH APPLICATION ONLY — NOT A MEDICAL DEVICE. "
    "This software executes offline validation of frozen neural research models on physical PPG recordings. "
    "No neural network parameters are trained, fine-tuned, or updated. "
    "Predictions do not establish medical accuracy or clinical validity and must not be used for diagnosis or treatment."
)

DEMO_WATERMARK_TEXT = (
    "DEMO / NOT REAL HARDWARE DATA / NOT RESEARCH RESULT"
)
