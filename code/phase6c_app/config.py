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
for p in [PROJECT_ROOT, CODE_DIR, CODE_DIR / "phase7_reliability", CODE_DIR / "scripts", CODE_DIR / "phase4a", APP_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Cross-platform module aliases for deserialization compatibility
try:
    import phase7_reliability
    import phase7_reliability.reliability_engine
    sys.modules["phase7_reliability"] = phase7_reliability
    sys.modules["phase7_reliability.reliability_engine"] = phase7_reliability.reliability_engine
except Exception:
    pass

try:
    import sklearn._loss._loss as _sklearn_loss
    sys.modules["_loss"] = _sklearn_loss
except Exception:
    pass

try:
    import numpy._core as _np_core
    sys.modules["numpy.core"] = _np_core
    sys.modules["numpy.core.multiarray"] = _np_core.multiarray
except Exception:
    try:
        import numpy.core as _np_core
        sys.modules["numpy._core"] = _np_core
        sys.modules["numpy._core.multiarray"] = _np_core.multiarray
    except Exception:
        pass

import pickle

class RobustUnpickler(pickle.Unpickler):
    """
    Robust Unpickler supporting cross-platform/cross-version deserialization:
    - Resolves compiled '_loss' Cython module alias to 'sklearn._loss._loss'
    - Resolves 'phase7_reliability' classes without requiring root-level pip installation
    - Resolves NumPy 1.x vs 2.x namespace differences ('numpy.core' <-> 'numpy._core')
    """
    def find_class(self, module, name):
        if module == "_loss":
            try:
                import sklearn._loss._loss as _loss_mod
                return getattr(_loss_mod, name)
            except Exception:
                pass
        if module.startswith("phase7_reliability"):
            try:
                import phase7_reliability.reliability_engine as _p7_mod
                if hasattr(_p7_mod, name):
                    return getattr(_p7_mod, name)
            except Exception:
                pass
        if "numpy" in module:
            if "core" in module:
                try:
                    import numpy._core.multiarray as _np_multi
                    if hasattr(_np_multi, name):
                        return getattr(_np_multi, name)
                except Exception:
                    pass
                try:
                    import numpy.core.multiarray as _np_multi
                    if hasattr(_np_multi, name):
                        return getattr(_np_multi, name)
                except Exception:
                    pass
        return super().find_class(module, name)


def safe_load_pickle(file_or_path):
    """Safely load pickle files across different environments and package versions."""
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "rb") as f:
            return RobustUnpickler(f).load()
    return RobustUnpickler(file_or_path).load()

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
