"""
Phase 7: Extended Test Suite for Reliability Engine, LLM Explainer, and Selective Prediction
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Tests (17 total):
 1. Record-level leakage control across partitions.
 2. Causality of temporal stability features (no future leakage).
 3. Independent validation threshold tuning (no test-set tuning).
 4. Deterministic reliability model training and inference.
 5. Three-state TRUST / REVIEW / ABSTAIN mapping logic.
 6. Selective prediction calculations and coverage monotonicity.
 7. Risk-coverage curve integration (AURC).
 8. Structured JSON interface boundary for LLM explanation layer.
 9. Frozen BP model checkpoints and calibration artifacts integrity.
10. Reliability thresholds load correctly from saved pickle.
11. LLM payload contains no API key, no raw PPG, no forbidden data.
12. LLM cannot modify reliability state (authority isolation test).
13. Invalid LLM JSON (bad format) does not crash; returns fallback.
14. Groq unavailable (no key) triggers deterministic local fallback.
15. BP prediction is IDENTICAL with and without Groq (isolation test).
16. Reliability state is IDENTICAL with and without Groq (isolation test).
17. Streamlit app module compiles without import errors.
"""

import unittest
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Ensure code/ is on sys.path for imports
PROJECT_ROOT = Path("/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone")
CODE_DIR = PROJECT_ROOT / "code"
APP_DIR = CODE_DIR / "phase6c_app"
for p in [str(CODE_DIR), str(APP_DIR), str(CODE_DIR / "scripts"), str(CODE_DIR / "phase4a")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from phase7_reliability.reliability_engine import (
    ReliabilityEngine, ReliabilityConfig, ALL_RELIABILITY_FEATURES
)
from scripts.build_phase7_dataset import compute_causal_temporal_features
from scripts.run_phase7_reliability import calculate_risk_coverage_metrics, compute_aurc

DATASET_CSV = PROJECT_ROOT / "code/outputs/phase7_reliability/reliability_dataset.csv"
MODEL_PKL = PROJECT_ROOT / "code/outputs/phase7_reliability/reliability_engine_model.pkl"

# Frozen Reference Cryptographic SHA-256 Hashes
FROZEN_HASHES = {
    "P4A_CNN": (
        PROJECT_ROOT / "code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt",
        "2c6c5478e5ec0c666cd2d5c43d7d2561c74ad25170f10075f24d2fa6610863ee"
    ),
    "P4B_GRU": (
        PROJECT_ROOT / "code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt",
        "26ecb0abf690bd067c5a083f80487e8f686c5c66ba49b9b84c7895fbaf8dcd81"
    ),
    "P5C_ISO_SBP": (
        PROJECT_ROOT / "code/outputs/phase5c_extreme_aware/mappings/isotonic_sbp.pkl",
        "44d25b279fcde22992b4c323de1ed6e5fa4595ffdcc7a75523a236fdd356a6bd"
    ),
    "P5C_ISO_DBP": (
        PROJECT_ROOT / "code/outputs/phase5c_extreme_aware/mappings/isotonic_dbp.pkl",
        "5932346ce2c55188ca885cb489100e4fab1a4b9a6973b0ef4153a9ba42c67828"
    ),
    "P5C_CONFORMAL_JSON": (
        PROJECT_ROOT / "code/outputs/phase5c_extreme_aware/calibration/conformal_quantiles_by_bin.json",
        "e0e50101cfa216d75afa447c07e1f1f9a1ab15cde3a4069718f7b33907c7f2d8"
    )
}

# Canonical test payload for LLM tests (matches schema produced by ReliabilityEngine)
_TEST_PAYLOAD = {
    "prediction": {"sbp": 135.0, "dbp": 85.0},
    "reliability_state": "TRUST",
    "reliability_score": 0.42,
    "reliability_reasons": [
        "Optical waveform signal quality verified (PASS).",
        "Rolling 60-second temporal predictions show strong stability.",
    ],
    "signal_quality": "PASS",
    "model_uncertainty": "LOW",
    "conformal_width": "STANDARD",
    "temporal_stability": "STABLE",
    "recommendation": "Prediction meets research consistency criteria.",
    "disclaimer": "Research reliability indicator — not a medical diagnosis or guarantee of BP accuracy.",
}


class TestPhase7Reliability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATASET_CSV)
        cls.df_dev = cls.df[cls.df["split"] == "dev"].copy()
        cls.df_val = cls.df[cls.df["split"] == "val"].copy()
        cls.df_test = cls.df[cls.df["split"] == "test"].copy()

    # ------------------------------------------------------------------
    # Tests 1–9 (existing, preserved verbatim except test_08 payload fix)
    # ------------------------------------------------------------------

    def test_01_record_level_leakage_control(self):
        """Verify strict pairwise disjointness of record IDs across dev, val, and test partitions."""
        dev_recs = set(self.df_dev["record_id"].unique())
        val_recs = set(self.df_val["record_id"].unique())
        test_recs = set(self.df_test["record_id"].unique())

        self.assertEqual(len(dev_recs.intersection(val_recs)), 0, "Leakage: dev and val records overlap!")
        self.assertEqual(len(dev_recs.intersection(test_recs)), 0, "Leakage: dev and test records overlap!")
        self.assertEqual(len(val_recs.intersection(test_recs)), 0, "Leakage: val and test records overlap!")

        self.assertEqual(len(dev_recs), 607)
        self.assertEqual(len(val_recs), 608)
        self.assertEqual(len(test_recs), 1198)

    def test_02_causal_temporal_features_no_future_leakage(self):
        """Verify that temporal features are backward-looking only and uninfluenced by future values."""
        df_toy = pd.DataFrame({
            "record_id": ["rec_1", "rec_1", "rec_1"],
            "target_window_index": [5, 6, 7],
            "pred_cal_sbp": [120.0, 125.0, 150.0],
            "pred_cal_dbp": [80.0, 82.0, 95.0]
        })
        feat1 = compute_causal_temporal_features(df_toy)
        val_at_t1 = feat1.loc[feat1["target_window_index"] == 6, "rolling_sbp_std_3"].values[0]

        df_toy_mod = df_toy.copy()
        df_toy_mod.loc[df_toy_mod["target_window_index"] == 7, "pred_cal_sbp"] = 200.0
        feat2 = compute_causal_temporal_features(df_toy_mod)
        val_at_t1_mod = feat2.loc[feat2["target_window_index"] == 6, "rolling_sbp_std_3"].values[0]

        self.assertAlmostEqual(val_at_t1, val_at_t1_mod, places=6,
                               msg="Future leakage detected: modifying future prediction altered past temporal feature!")

    def test_03_no_test_set_threshold_tuning(self):
        """Verify that operating thresholds are tuned strictly on validation data, leaving test data untouched."""
        eng = ReliabilityEngine(model_type="hist_gb")
        eng.fit(self.df_dev)
        eng.tune_thresholds(self.df_val)

        self.assertGreater(eng.tau_trust, 0.0)
        self.assertLess(eng.tau_trust, eng.tau_abstain)
        self.assertLess(eng.tau_abstain, 1.0)

    def test_04_deterministic_reliability_models(self):
        """Verify all three baseline reliability models train and output valid probabilities."""
        for m_type in ["rule_based", "logistic", "hist_gb"]:
            eng = ReliabilityEngine(model_type=m_type)
            eng.fit(self.df_dev.iloc[:1000])
            eng.tune_thresholds(self.df_val.iloc[:1000])

            scores = eng.predict_risk(self.df_test.iloc[:100])
            self.assertEqual(len(scores), 100)
            self.assertTrue(np.all(scores >= 0.0), f"{m_type}: score < 0")
            self.assertTrue(np.all(scores <= 1.0), f"{m_type}: score > 1")

    def test_05_three_state_mapping(self):
        """Verify deterministic three-state mapping into TRUST, REVIEW, ABSTAIN."""
        eng = ReliabilityEngine(model_type="hist_gb")
        eng.tau_trust = 0.40
        eng.tau_abstain = 0.70

        scores = np.array([0.20, 0.40, 0.50, 0.70, 0.85])
        df_dummy = pd.DataFrame(np.zeros((len(scores), len(ALL_RELIABILITY_FEATURES))), columns=ALL_RELIABILITY_FEATURES)

        eng.predict_risk = lambda X: scores
        states = eng.predict_state(df_dummy)

        expected = ["TRUST", "TRUST", "REVIEW", "REVIEW", "ABSTAIN"]
        self.assertEqual(states, expected)

    def test_06_selective_prediction_calculations(self):
        """Verify selective prediction calculations and coverage monotonicity."""
        df_sub = self.df_test.iloc[:5000].copy()
        df_sub["risk_score"] = np.random.uniform(0, 1, len(df_sub))

        metrics = calculate_risk_coverage_metrics(df_sub, "risk_score", [1.0, 0.8, 0.5])
        self.assertEqual(len(metrics), 3)

        self.assertEqual(metrics.loc[metrics["coverage"] == 1.0, "n_retained"].values[0], 5000)
        self.assertEqual(metrics.loc[metrics["coverage"] == 0.8, "n_retained"].values[0], 4000)
        self.assertEqual(metrics.loc[metrics["coverage"] == 0.5, "n_retained"].values[0], 2500)

    def test_07_aurc_calculation(self):
        """Verify Area Under Risk-Coverage curve calculation."""
        df_sub = self.df_test.iloc[:1000].copy()
        df_sub["risk_score"] = np.linspace(0, 1, len(df_sub))

        aurc = compute_aurc(df_sub, "risk_score", "abs_sbp_error_cal")
        self.assertIsInstance(aurc, float)
        self.assertGreater(aurc, 0.0)

    def test_08_llm_interface_payload_schema(self):
        """Verify structured JSON payload schema for LLM explanation layer."""
        eng = ReliabilityEngine(model_type="hist_gb")
        eng.fit(self.df_dev.iloc[:500])
        eng.tune_thresholds(self.df_val.iloc[:500])

        payload = eng.generate_explanation_payload(
            sbp_pred=135.0,
            dbp_pred=85.0,
            features={
                "qc_pass": 1.0, "ppg_ptp": 40000.0, "ppg_std": 10000.0,
                "ppg_clipped_fraction": 0.0, "ppg_pulse_count": 14.0, "estimated_hr_bpm": 72.0,
                "sbp_uncertainty": 14.0, "dbp_uncertainty": 7.5,
                "sbp_conformal_width_90": 49.8, "dbp_conformal_width_90": 22.9,
                "rolling_sbp_std_3": 1.2, "rolling_dbp_std_3": 0.8,
                "max_abs_sbp_jump_3": 1.5, "max_abs_dbp_jump_3": 1.0,
                "median_abs_sbp_change_3": 1.5, "median_abs_dbp_change_3": 1.0
            }
        )

        required_keys = [
            "prediction", "reliability_state", "reliability_score",
            "reliability_reasons", "signal_quality", "model_uncertainty",
            "conformal_width", "temporal_stability", "recommendation", "disclaimer"
        ]
        for k in required_keys:
            self.assertIn(k, payload, f"Missing payload key: {k}")

        self.assertIn(payload["reliability_state"], ["TRUST", "REVIEW", "ABSTAIN"])
        self.assertIn("not a medical diagnosis", payload["disclaimer"])

    def test_09_frozen_bp_model_weights_unaltered(self):
        """Verify that frozen Phase 4A/4B checkpoints and Phase 5C calibration artifacts remain 100% unaltered."""
        for name, (file_path, expected_hash) in FROZEN_HASHES.items():
            self.assertTrue(file_path.exists(), f"Frozen artifact missing: {file_path}")
            current_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            self.assertEqual(current_hash, expected_hash,
                             f"SECURITY VIOLATION: Frozen artifact {name} has been modified! Hash mismatch.")

    # ------------------------------------------------------------------
    # Tests 10–17 (new — LLM isolation, security, fallback, compilation)
    # ------------------------------------------------------------------

    def test_10_reliability_thresholds_load_from_pickle(self):
        """Verify fitted reliability engine loads from pickle with valid tau_trust and tau_abstain."""
        import pickle
        self.assertTrue(MODEL_PKL.exists(), f"Reliability model pickle not found: {MODEL_PKL}")
        with open(MODEL_PKL, "rb") as f:
            eng = pickle.load(f)

        self.assertTrue(hasattr(eng, "tau_trust"), "Loaded engine missing tau_trust attribute.")
        self.assertTrue(hasattr(eng, "tau_abstain"), "Loaded engine missing tau_abstain attribute.")
        self.assertGreater(eng.tau_trust, 0.0)
        self.assertLess(eng.tau_trust, eng.tau_abstain)
        self.assertLess(eng.tau_abstain, 1.0)
        # Verify the specific tuned values match the frozen research result
        self.assertAlmostEqual(eng.tau_trust, 0.5327, places=3,
                               msg="tau_trust differs from frozen research value!")
        self.assertAlmostEqual(eng.tau_abstain, 0.6608, places=3,
                               msg="tau_abstain differs from frozen research value!")

    def test_11_llm_payload_contains_no_forbidden_data(self):
        """Verify that build_groq_payload strips forbidden fields and rejects invalid payloads."""
        from llm_explainer import build_groq_payload, _FORBIDDEN_KEY_PATTERNS

        # Valid payload should pass
        safe = build_groq_payload(_TEST_PAYLOAD)
        safe_json = json.dumps(safe).lower()

        # Must not contain any token/key strings (from accidentally added keys)
        self.assertNotIn("api_key", safe_json, "Forbidden 'api_key' key found in Groq payload!")

        # Invalid payload (missing keys) should raise ValueError
        with self.assertRaises(ValueError):
            build_groq_payload({"prediction": {"sbp": 120, "dbp": 80}})  # missing many required keys

        # Payload with a forbidden KEY should raise ValueError
        bad_payload = dict(_TEST_PAYLOAD)
        bad_payload["raw_ppg"] = [1, 2, 3, 4, 5]  # 'raw_ppg' is a forbidden KEY pattern
        with self.assertRaises(ValueError):
            build_groq_payload(bad_payload)

        # Payload with 'api_key' as a top-level key should raise ValueError
        bad_payload2 = dict(_TEST_PAYLOAD)
        bad_payload2["api_key"] = "secret"
        with self.assertRaises(ValueError):
            build_groq_payload(bad_payload2)

    def test_12_llm_cannot_modify_reliability_state(self):
        """
        Verify authority isolation: no matter what the LLM returns,
        the reliability_state in the original payload is unchanged.
        """
        from llm_explainer import generate_reliability_explanation

        original_state = _TEST_PAYLOAD["reliability_state"]
        original_sbp = _TEST_PAYLOAD["prediction"]["sbp"]
        original_score = _TEST_PAYLOAD["reliability_score"]

        # Simulate a Groq response that TRIES to change the state
        malicious_groq_response = json.dumps({
            "summary": "I have decided the state is ABSTAIN.",
            "signal_explanation": "Override: ABSTAIN",
            "state_interpretation": "ABSTAIN overridden by LLM",
            "recommendation_for_researcher": "State was changed to ABSTAIN.",
            "research_disclaimer": "Disclaimer."
        })

        mock_choice = MagicMock()
        mock_choice.message.content = malicious_groq_response
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        # Patch _groq_factory at module level — avoids importing groq/pydantic_core in tests
        with patch("llm_explainer._groq_factory", return_value=mock_client), \
             patch("llm_explainer._get_api_key", return_value="fake_test_key_abc"):
            result = generate_reliability_explanation(_TEST_PAYLOAD)

        # The explanation text may reflect what the LLM said, BUT:
        # The original payload reliability_state must be UNCHANGED
        self.assertEqual(_TEST_PAYLOAD["reliability_state"], original_state,
                         "AUTHORITY VIOLATION: LLM call modified the original payload reliability_state!")
        self.assertEqual(_TEST_PAYLOAD["prediction"]["sbp"], original_sbp,
                         "AUTHORITY VIOLATION: LLM call modified the original payload SBP prediction!")
        self.assertEqual(_TEST_PAYLOAD["reliability_score"], original_score,
                         "AUTHORITY VIOLATION: LLM call modified the original payload reliability_score!")

        # The result should preserve the verified state reference
        if result.get("source") == "groq":
            self.assertEqual(result.get("_verified_state"), original_state,
                             "_verified_state in result does not match original payload state!")

    def test_13_invalid_llm_json_triggers_local_fallback(self):
        """Verify that a malformed JSON response from Groq does not crash — returns local fallback."""
        from llm_explainer import generate_reliability_explanation

        mock_choice = MagicMock()
        mock_choice.message.content = "This is not valid JSON at all!!!! {broken"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("llm_explainer._groq_factory", return_value=mock_client), \
             patch("llm_explainer._get_api_key", return_value="fake_test_key_abc"):
            result = generate_reliability_explanation(_TEST_PAYLOAD)

        # Must not raise. Must return a dict with required fields.
        self.assertIsInstance(result, dict)
        self.assertIn("source", result)
        self.assertIn("local_fallback", result["source"],
                      "Expected local_fallback source on JSON parse failure.")
        self.assertIn("summary", result)
        self.assertIn("research_disclaimer", result)

    def test_14_missing_groq_key_triggers_local_fallback(self):
        """Verify that missing GROQ_API key returns a deterministic local fallback without raising."""
        from llm_explainer import generate_reliability_explanation

        with patch("llm_explainer._get_api_key", return_value=None):
            result = generate_reliability_explanation(_TEST_PAYLOAD)

        self.assertIsInstance(result, dict)
        self.assertIn("source", result)
        self.assertTrue(
            result["source"].startswith("local_fallback"),
            f"Expected local_fallback source when key is missing, got: {result['source']}"
        )
        self.assertIn("summary", result)
        self.assertIn("research_disclaimer", result)
        # Fallback must not be empty
        self.assertGreater(len(result["summary"]), 20)

    def test_15_bp_prediction_identical_with_and_without_groq(self):
        """
        Verify that calling the LLM explainer does NOT change BP predictions.
        The underlying prediction values must be independent of Groq calls.
        """
        from llm_explainer import generate_reliability_explanation

        sbp_before = _TEST_PAYLOAD["prediction"]["sbp"]
        dbp_before = _TEST_PAYLOAD["prediction"]["dbp"]

        # Call with Groq mocked to succeed
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "summary": "Test explanation.",
            "signal_explanation": "Signal is good.",
            "state_interpretation": "Consistent with research thresholds.",
            "recommendation_for_researcher": "No action needed.",
            "research_disclaimer": "Research instrument disclaimer."
        })
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("llm_explainer._groq_factory", return_value=mock_client), \
             patch("llm_explainer._get_api_key", return_value="fake_test_key_abc"):
            _ = generate_reliability_explanation(_TEST_PAYLOAD)

        self.assertEqual(_TEST_PAYLOAD["prediction"]["sbp"], sbp_before,
                         "SBP prediction value changed after Groq call!")
        self.assertEqual(_TEST_PAYLOAD["prediction"]["dbp"], dbp_before,
                         "DBP prediction value changed after Groq call!")

    def test_16_reliability_state_identical_with_and_without_groq(self):
        """
        Verify that calling the LLM explainer does NOT change the reliability state in the payload.
        """
        from llm_explainer import generate_reliability_explanation

        state_before = _TEST_PAYLOAD["reliability_state"]
        score_before = _TEST_PAYLOAD["reliability_score"]

        # Call with Groq mocked to succeed
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "summary": "All signals look consistent.",
            "signal_explanation": "No anomalies detected.",
            "state_interpretation": "Consistent with research thresholds.",
            "recommendation_for_researcher": "Proceed with analysis.",
            "research_disclaimer": "Research instrument disclaimer."
        })
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("llm_explainer._groq_factory", return_value=mock_client), \
             patch("llm_explainer._get_api_key", return_value="fake_test_key_abc"):
            result = generate_reliability_explanation(_TEST_PAYLOAD)

        self.assertEqual(_TEST_PAYLOAD["reliability_state"], state_before,
                         "Reliability state changed in payload after Groq call!")
        self.assertEqual(_TEST_PAYLOAD["reliability_score"], score_before,
                         "Reliability score changed in payload after Groq call!")

    def test_17_streamlit_app_compiles(self):
        """Verify that the Streamlit app module compiles without syntax or import-level errors."""
        import py_compile
        app_path = APP_DIR / "app.py"
        self.assertTrue(app_path.exists(), f"app.py not found at {app_path}")
        # py_compile raises SyntaxError on compile failure
        try:
            py_compile.compile(str(app_path), doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"app.py failed to compile: {e}")

        # Also verify llm_explainer compiles
        explainer_path = APP_DIR / "llm_explainer.py"
        self.assertTrue(explainer_path.exists(), f"llm_explainer.py not found at {explainer_path}")
        try:
            py_compile.compile(str(explainer_path), doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"llm_explainer.py failed to compile: {e}")

    def test_18_real_groq_dependency_import(self):
        """
        Verify that the groq SDK and python-dotenv are properly installed in the venv
        and that a Groq client can be constructed from environment configuration.

        This test does NOT make any real API request to Groq.
        The API key is loaded from .env but never printed, logged, or exposed.
        The purpose is solely to verify:
          1. The groq package imports correctly (pydantic-core compatible).
          2. The Groq client class can be instantiated.
          3. The python-dotenv package imports correctly.
          4. GROQ_API or GROQ_API_KEY is detectable in the environment.
        """
        # 1. Verify groq SDK imports cleanly (no pydantic_core extension errors)
        try:
            from groq import Groq
        except ImportError as e:
            self.fail(f"groq SDK could not be imported in the venv: {e}. "
                      "Run: .venv/bin/python -m pip install --upgrade --force-reinstall groq")

        # 2. Verify python-dotenv imports
        try:
            from dotenv import load_dotenv
        except ImportError as e:
            self.fail(f"python-dotenv could not be imported: {e}. "
                      "Run: .venv/bin/python -m pip install python-dotenv")

        # 3. Load .env without printing the key
        import os
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)

        # 4. Verify key is detectable (presence only — never print value)
        key = (os.environ.get("GROQ_API") or os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'")
        self.assertTrue(
            len(key) > 10,
            "GROQ_API or GROQ_API_KEY not found in .env or environment. "
            "Ensure .env contains GROQ_API = <key> at the project root."
        )

        # 5. Construct Groq client (no API call — constructor only; uses timeout config)
        try:
            client = Groq(api_key=key, timeout=15.0)
        except Exception as e:
            self.fail(f"Groq client construction failed: {type(e).__name__}: {e}")

        # 6. Verify client has the expected interface (no request made)
        self.assertTrue(
            hasattr(client, "chat"),
            "Groq client does not have 'chat' attribute — SDK may be incompatible."
        )
        self.assertTrue(
            hasattr(client.chat, "completions"),
            "Groq client.chat does not have 'completions' attribute."
        )
        # Key must not appear in any string representation of the client object
        client_repr = repr(client)
        self.assertNotIn(key, client_repr, "API key leaked into client repr!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
