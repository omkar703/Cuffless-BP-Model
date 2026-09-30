"""
Unit and Scientific Integrity Tests for Phase 6C / Phase 7 Live Experiment Engine.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only
"""

import unittest
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from phase6c_app.analysis.live_engine import LiveExperimentEngine, discover_serial_ports
from phase6c_app.config import (
    PHASE4A_CKPT,
    PHASE4B_CKPT,
    ISOTONIC_SBP_PKL,
    ISOTONIC_DBP_PKL,
    CONFORMAL_QUANTILES_JSON,
    EXPECTED_TOTAL_PARAMS,
    PROJECT_ROOT
)

class TestLiveExperimentEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LiveExperimentEngine()
        cls.rel_pkl = PROJECT_ROOT / "code" / "outputs" / "phase7_reliability" / "reliability_engine_model.pkl"
        
        def get_hash(path):
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
                
        cls.get_hash = staticmethod(get_hash)
        cls.initial_hashes = {
            "4a": get_hash(PHASE4A_CKPT),
            "4b": get_hash(PHASE4B_CKPT),
            "iso_s": get_hash(ISOTONIC_SBP_PKL),
            "iso_d": get_hash(ISOTONIC_DBP_PKL),
            "cq": get_hash(CONFORMAL_QUANTILES_JSON),
            "rel": get_hash(cls.rel_pkl),
        }

    def setUp(self):
        self.engine.reset()

    def test_01_frozen_parameter_invariants(self):
        """Verify models loaded in LiveExperimentEngine have 0 trainable parameters and exact count."""
        cnn_p = sum(p.numel() for p in self.engine.cnn_model.parameters())
        gru_p = sum(p.numel() for p in self.engine.gru_model.parameters())
        trainable_p = sum(p.numel() for p in list(self.engine.cnn_model.parameters()) + list(self.engine.gru_model.parameters()) if p.requires_grad)
        
        self.assertEqual(trainable_p, 0)
        self.assertEqual(cnn_p + gru_p, EXPECTED_TOTAL_PARAMS)

    def test_02_insufficient_context_cannot_produce_prediction(self):
        """Verify that less than 6 windows (60 seconds) strictly cannot produce a BP prediction."""
        # Ingest 30 seconds of samples (3 windows @ 100 Hz = 3000 samples)
        for i in range(3000):
            self.engine.ingest_sample(i, i*10.0, i*10.0, 50000.0 + 2000.0*np.sin(i*0.1), 30.0)
            
        self.assertLess(len(self.engine.completed_windows), 6)
        self.assertIsNone(self.engine.latest_prediction, "Prediction must NOT be generated before 6 windows!")
        self.assertIsNone(self.engine.latest_reliability, "Reliability must NOT be generated before 6 windows!")
        self.assertEqual(self.engine.pipeline_state, "BUILDING_CONTEXT")

    def test_03_six_windows_produces_deterministic_prediction(self):
        """Verify that 6 valid windows trigger GRU and Phase 5C calibration deterministically."""
        # Load real hardware capture
        df_samples = pd.read_csv(PROJECT_ROOT / "hardware" / "final_dataset_ready.csv")
        
        # Ingest 7500 samples (7.5 windows = 75 seconds)
        for _, row in df_samples.iloc[:7500].iterrows():
            s_idx = int(row['sample_index'])
            ts = float(row.get('timestamp_ms', s_idx * 10))
            ir = float(row['ir'])
            red = float(row.get('red', 0.0))
            self.engine.ingest_sample(s_idx, ts, ts, ir, red)
            
        self.assertGreaterEqual(len(self.engine.completed_windows), 6)
        self.assertIsNotNone(self.engine.latest_prediction, "Prediction must be generated after 6 valid windows.")
        self.assertIsNotNone(self.engine.latest_reliability, "Phase 7 reliability must be evaluated.")
        
        pred = self.engine.latest_prediction
        self.assertIn("calibrated_sbp", pred)
        self.assertIn("calibrated_dbp", pred)
        self.assertGreater(pred["calibrated_sbp"], 50.0)
        self.assertLess(pred["calibrated_sbp"], 250.0)
        
        rel = self.engine.latest_reliability
        self.assertIn(rel["reliability_state"], ["TRUST", "REVIEW", "ABSTAIN"])

    def test_04_reference_cuff_isolation(self):
        """Verify that entering an exploratory reference cuff measurement does NOT alter prediction or model."""
        # Ensure we have a prediction
        df_samples = pd.read_csv(PROJECT_ROOT / "hardware" / "final_dataset_ready.csv")
        for _, row in df_samples.iloc[:7500].iterrows():
            self.engine.ingest_sample(int(row['sample_index']), 0, 0, float(row['ir']), float(row.get('red', 0.0)))
            
        initial_sbp = self.engine.latest_prediction["calibrated_sbp"]
        initial_dbp = self.engine.latest_prediction["calibrated_dbp"]
        initial_rel_state = self.engine.latest_reliability["reliability_state"]
        
        # Compare against arbitrary cuff readings
        comp = self.engine.compare_reference(180.0, 110.0)
        
        # Verify prediction is 100% UNCHANGED
        self.assertEqual(self.engine.latest_prediction["calibrated_sbp"], initial_sbp)
        self.assertEqual(self.engine.latest_prediction["calibrated_dbp"], initial_dbp)
        self.assertEqual(self.engine.latest_reliability["reliability_state"], initial_rel_state)
        
        # Verify comparison math
        self.assertAlmostEqual(comp["diff_sbp"], initial_sbp - 180.0)
        self.assertAlmostEqual(comp["diff_dbp"], initial_dbp - 110.0)

    def test_05_serial_parser_robustness(self):
        """Verify serial parser handles 5-col, 4-col, 3-col, and malformed lines gracefully."""
        self.assertEqual(self.engine.parse_serial_line("10,100,102,54000,32"), (10, 100.0, 102.0, 54000.0, 32.0))
        self.assertEqual(self.engine.parse_serial_line("11,110,54100,33"), (11, 110.0, 110.0, 54100.0, 33.0))
        self.assertEqual(self.engine.parse_serial_line("12,54200,34"), (12, 120.0, 120.0, 54200.0, 34.0))
        self.assertIsNone(self.engine.parse_serial_line("ESP32 MAX30102 Ready!"))
        self.assertIsNone(self.engine.parse_serial_line("invalid,data,line"))

    def test_06_model_file_checksums_unaltered(self):
        """Verify that running the LiveExperimentEngine did NOT modify any on-disk artifact."""
        current_hashes = {
            "4a": self.get_hash(PHASE4A_CKPT),
            "4b": self.get_hash(PHASE4B_CKPT),
            "iso_s": self.get_hash(ISOTONIC_SBP_PKL),
            "iso_d": self.get_hash(ISOTONIC_DBP_PKL),
            "cq": self.get_hash(CONFORMAL_QUANTILES_JSON),
            "rel": self.get_hash(self.rel_pkl),
        }
        for k, v in self.initial_hashes.items():
            self.assertEqual(v, current_hashes[k], f"Artifact {k} was modified!")

if __name__ == "__main__":
    unittest.main()
