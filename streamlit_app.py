"""
Streamlit Community Cloud Main Entry Point
Project: Calibration-Free Cuffless Blood Pressure Estimation Using PPG Alone

This file serves as the root deployment entry point for Streamlit Community Cloud.
It configures python paths for the project modules and executes code/phase6c_app/app.py.
"""

import sys
import runpy
from pathlib import Path

# Resolve absolute paths relative to repository root
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
APP_DIR = CODE_DIR / "phase6c_app"

# Register required module paths on sys.path
for p in [PROJECT_ROOT, CODE_DIR, CODE_DIR / "phase7_reliability", APP_DIR, CODE_DIR / "scripts", CODE_DIR / "phase4a"]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Pre-register module aliases for deserialization across environments
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

# Execute the primary Streamlit application
runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
