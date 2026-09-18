"""
Configuration parameters for Phase 1: PPG-Only Blood Pressure Dataset Cleaning,
Validation, and Visualization.
"""

from pathlib import Path

# ==============================================================================
# Directory Paths
# ==============================================================================
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent

DATASET_DIR = PROJECT_ROOT / "BloodPressureDataset"
OUTPUT_DIR = CODE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
STATISTICS_DIR = OUTPUT_DIR / "statistics"
CLEANED_DIR = OUTPUT_DIR / "cleaned"
LOGS_DIR = CODE_DIR / "logs"

WINDOWS_DIR = OUTPUT_DIR / "windows"
SPLITS_DIR = OUTPUT_DIR / "splits"

# Ensure output directories exist
for d in [OUTPUT_DIR, FIGURES_DIR, STATISTICS_DIR, CLEANED_DIR, LOGS_DIR, WINDOWS_DIR, SPLITS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Output Manifest and Report Filenames
QC_MANIFEST_FILENAME = "dataset_qc_manifest.csv"
QC_REPORT_FILENAME = "DATASET_QUALITY_REPORT.md"
QC_SUMMARY_JSON = "dataset_summary_statistics.json"

# Phase 2 Output Filenames
WINDOW_MANIFEST_FILENAME = "window_manifest.csv"
WINDOW_SUMMARY_JSON = "window_summary_statistics.json"
RECORD_SPLIT_FILENAME = "record_split.csv"
PHASE2_REPORT_FILENAME = "PHASE2_WINDOW_REPORT.md"

# ==============================================================================
# Signal & Acquisition Parameters
# ==============================================================================
SAMPLING_RATE = 125  # Hz, standard PhysioNet MIMIC-II acquisition rate
TIME_STEP = 1.0 / SAMPLING_RATE  # seconds per sample (0.008 s = 8 ms)

# Channel indices in MATLAB cells p[0, i] and CSV files
PPG_CHANNEL_IDX = 0
ABP_CHANNEL_IDX = 1
ECG_CHANNEL_IDX = 2  # NOTE: ECG is explicitly NOT USED for BP estimation!

# ==============================================================================
# Quality Control & Physiological Validation Thresholds
# ==============================================================================
# Minimum signal length: require at least 8 seconds (1000 samples) to ensure
# several full cardiac cycles can be observed
MIN_SIGNAL_LENGTH_SAMPLES = 1000  # 8 seconds at 125 Hz
MIN_SIGNAL_DURATION_SEC = MIN_SIGNAL_LENGTH_SAMPLES / SAMPLING_RATE  # 8.0 s

# PPG Validity Constraints (Decoupled from ABP)
PPG_MIN_STD = 1e-5          # Minimum standard deviation to reject flatlines / zero variance
PPG_MAX_CLIPPING_RATIO = 0.05  # Maximum fraction of samples clamped at peak or trough
PPG_MAX_JUMP_STD_FACTOR = 30.0 # Maximum single-sample jump relative to signal std (discontinuity)
# Note: PPG amplitude (ptp) is treated as a DIAGNOSTIC indicator, not an absolute rejection rule

# ABP Reference Validity Constraints (Decoupled from PPG)
ABP_MIN_STD = 2.0           # Minimum standard deviation (reject zero/flat catheter lines)
ABP_ABSOLUTE_MIN = 25.0     # mmHg, below which arterial catheter is decoupled or clamped
ABP_ABSOLUTE_MAX = 260.0    # mmHg, above which catheter flushing artifact occurs
SBP_PLAUSIBLE_RANGE = (50.0, 240.0)    # mmHg, physiological systolic blood pressure limits
DBP_PLAUSIBLE_RANGE = (30.0, 140.0)    # mmHg, physiological diastolic blood pressure limits
MIN_PULSE_PRESSURE = 10.0              # SBP - DBP must be at least 10 mmHg for a valid beat

# Peak Detection on ABP for Beat-by-Beat Ground Truth Diagnostics
ABP_PEAK_MIN_DISTANCE = int(0.35 * SAMPLING_RATE)  # ~44 samples (max HR ~170 bpm)
ABP_PEAK_PROMINENCE = 10.0                         # mmHg prominence threshold


# ==============================================================================
# Signal Preprocessing Settings
# ==============================================================================
# Offline Research Zero-Phase Filter
# IMPORTANT: Clearly designated as an OFFLINE filter (filtfilt). The future
# ESP32 embedded deployment will require a streaming causal IIR/FIR filter.
FILTER_TYPE = "OFFLINE_RESEARCH_FILTER"
FILTER_METHOD = "butterworth_bandpass_zero_phase"
PPG_BANDPASS_LOWCUT = 0.5   # Hz (30 bpm minimum heart rate, cuts baseline drift)
PPG_BANDPASS_HIGHCUT = 8.0  # Hz (captures systolic wave, dicrotic notch, and harmonics)
PPG_FILTER_ORDER = 3        # 3rd-order Butterworth (effective 6th-order after filtfilt)

# Visualization settings
REPRESENTATIVE_RECORD_IDS = [
    "part_01_record_000001",
    "part_01_record_000002",
    "part_01_record_000005",
    "part_02_record_001050",
    "part_03_record_002100",
    "part_04_record_003200",
]
VIS_WINDOW_SECONDS = [5.0, 10.0, 20.0]  # Window lengths for multi-scale plotting

# ==============================================================================
# Phase 2: Windowing Parameters
# ==============================================================================
WINDOW_SECONDS = 10.0       # Primary modeling window length in seconds
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_RATE)  # 1250 samples at 125 Hz
PRIMARY_OVERLAP = 0.0       # 0% overlap for independent baseline dataset

# ==============================================================================
# Phase 2: Window-Level PPG Quality Control Parameters
# ==============================================================================
PPG_WINDOW_MIN_STD = 1e-5             # Reject flatlines with near-zero variance
PPG_WINDOW_MAX_CLIPPING_RATIO = 0.05  # Reject if > 5% samples clipped at extremum
PPG_WINDOW_WARN_CLIPPING_RATIO = 0.02 # Warn if between 2% and 5%
PPG_WINDOW_MAX_JUMP_FACTOR = 30.0     # Discontinuity step check relative to std
PPG_HR_PLAUSIBLE_RANGE = (40.0, 220.0)# Physiological cardiac frequency in bpm
PPG_PULSE_MIN_DISTANCE = int(0.27 * SAMPLING_RATE) # ~34 samples (~220 bpm)
PPG_PULSE_PROMINENCE_RATIO = 0.15     # Prominence relative to signal std

# ==============================================================================
# Phase 2: Window-Level ABP Target Generation Parameters
# ==============================================================================
ABP_MIN_VALID_BEATS = 3               # Require at least 3 valid cardiac cycles per 10s
ABP_MIN_VALID_BEAT_RATIO = 0.60       # At least 60% candidate beats must be valid
ABP_WARN_VALID_BEAT_RATIO = 0.80      # Warn if between 60% and 80%
SBP_WINDOW_PLAUSIBLE_RANGE = (50.0, 240.0) # Physiological SBP limits (mmHg)
DBP_WINDOW_PLAUSIBLE_RANGE = (30.0, 140.0) # Physiological DBP limits (mmHg)
ABP_WINDOW_MIN_PULSE_PRESSURE = 10.0       # SBP - DBP >= 10 mmHg

# ==============================================================================
# Phase 2: Record-Level Partitioning Parameters (Zero Leakage)
# ==============================================================================
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
SPLIT_RANDOM_SEED = 42
LEAKAGE_CONTROL_NOTE = (
    "Record-level separation is used as the strongest available leakage-control mechanism; "
    "explicit subject identity is not available in the dataset."
)

