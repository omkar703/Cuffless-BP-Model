#!/usr/bin/env python3
"""
Automated regression test verifying that every representative Phase 2 scenario figure
matches its exact manifest quality fields.
"""

import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config.config import WINDOWS_DIR, WINDOW_MANIFEST_FILENAME
from visualization.window_plots import (
    AUDITED_REPRESENTATIVE_SCENARIOS,
    assert_representative_scenario_matches_manifest,
)

def run_tests():
    manifest_path = WINDOWS_DIR / WINDOW_MANIFEST_FILENAME
    assert manifest_path.exists(), f"Manifest not found at {manifest_path}"
    df = pd.read_csv(manifest_path).set_index("window_id")

    print(f"Auditing all {len(AUDITED_REPRESENTATIVE_SCENARIOS)} representative scenarios against manifest...")
    for scen_key, spec in AUDITED_REPRESENTATIVE_SCENARIOS.items():
        wid = spec["window_id"]
        assert wid in df.index, f"Window ID {wid} not found in manifest!"
        row = df.loc[wid]
        row_dict = row.to_dict()
        row_dict["window_id"] = wid
        row_series = pd.Series(row_dict)

        assert_representative_scenario_matches_manifest(scen_key, row_series, spec)
        print(f"  ✓ {scen_key:25s} [{wid}] -> MATCHES MANIFEST EXACTLY")

    print("\nALL REPRESENTATIVE FIGURE MANIFEST ASSERTIONS PASSED WITH ZERO ERRORS!")

if __name__ == "__main__":
    run_tests()
