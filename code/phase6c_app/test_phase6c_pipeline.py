"""
Phase 6C Pipeline Comprehensive Automated Test.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Tests the full end-to-end flow of the Phase 6C desktop validation tool:
1. Synthetic session creation & ZIP export/loading
2. Hardware data integrity audit
3. Frozen model parameter count assertion (174,084 total, 0 trainable)
4. Streaming replay simulation & causal feature extraction
5. Deterministic BP pairing rule verification
6. Metric calculations (MAE, RMSE, Bias, SD, correlations, AAMI/BHS)
7. Diagnostic plot generation
8. Markdown & JSON validation report synthesis
9. Export packaging & ZIP archive verification
"""

import sys
import tempfile
from pathlib import Path

# Add project root and app dir to path
APP_DIR = Path(__file__).resolve().parent
CODE_DIR = APP_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

for p in [APP_DIR, CODE_DIR, CODE_DIR / "scripts", CODE_DIR / "phase4a"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import torch

from phase6c_app.config import (
    PHASE4A_CKPT,
    PHASE4B_CKPT,
    EXPECTED_CNN_PARAMS,
    EXPECTED_GRU_PARAMS,
    EXPECTED_TOTAL_PARAMS,
    DEFAULT_PAIRING_TOLERANCE_S,
)
from phase6c_app.data_io.demo_generator import generate_demo_session, create_demo_zip_bytes
from phase6c_app.data_io.session_loader import load_session_from_zip
from phase6c_app.data_io.exporter import export_full_results
from phase6c_app.analysis.hardware_audit import audit_hardware_data
from phase6c_app.analysis.replay_engine import FrozenPipelineRunner
from phase6c_app.analysis.metrics import compute_validation_summary
from phase6c_app.pairing.bp_pairing import pair_predictions_with_reference


def run_comprehensive_test():
    print("=" * 80)
    print("  PHASE 6C PIPELINE AUTOMATED VERIFICATION TEST")
    print("=" * 80)

    # 1. Test Demo Data Generation & ZIP Loading
    print("\n[TEST 1] Testing Demo Data Generation and ZIP Serialization...")
    demo_session = generate_demo_session()
    assert demo_session.is_demo is True, "Demo session must be flagged as demo"
    assert len(demo_session.df_ppg_raw) > 8000, f"Expected >8000 raw samples, got {len(demo_session.df_ppg_raw)}"
    assert len(demo_session.df_bp_events) >= 2, f"Expected >=2 BP events, got {len(demo_session.df_bp_events)}"

    zip_bytes = create_demo_zip_bytes()
    assert len(zip_bytes) > 1000, "ZIP bytes must not be empty"

    import io
    loaded_from_zip = load_session_from_zip(io.BytesIO(zip_bytes))
    assert len(loaded_from_zip.df_ppg_raw) == len(demo_session.df_ppg_raw), "ZIP round-trip row count mismatch"
    print("  ✓ Demo session generation and ZIP round-trip passed.")

    # 2. Test Hardware Data Audit
    print("\n[TEST 2] Testing Hardware Acquisition Audit...")
    audit_res = audit_hardware_data(demo_session.df_ppg_replay, "synthetic_test.csv")
    assert audit_res["row_count"] == len(demo_session.df_ppg_replay)
    assert audit_res["sample_index"]["discontinuities_vs_modal_step"] == 0
    assert audit_res["host_timing"]["effective_rate_hz"] > 95.0
    print(f"  ✓ Hardware audit completed: Verdict = {audit_res['audit_verdict']}, Rate = {audit_res['host_timing']['effective_rate_hz']:.2f} Hz.")

    # 3. Test Frozen Model Architecture and Zero Retraining Invariant
    print("\n[TEST 3] Testing Frozen Model Architecture & Parameter Verification...")
    runner = FrozenPipelineRunner()
    meta = runner.load_models()
    assert meta["trainable_params"] == 0, f"Trainable params must be 0! Got {meta['trainable_params']}"
    assert meta["total_params"] == EXPECTED_TOTAL_PARAMS, f"Params mismatch: {meta['total_params']} != {EXPECTED_TOTAL_PARAMS}"
    assert meta["cnn_params"] == EXPECTED_CNN_PARAMS
    assert meta["gru_params"] == EXPECTED_GRU_PARAMS
    print(f"  ✓ Model integrity confirmed: Total = {meta['total_params']:,} (Trainable: {meta['trainable_params']})")

    # 4. Test Replay Execution
    print("\n[TEST 4] Running Streaming Replay Simulation on Demo PPG...")
    replay_res = runner.run_replay(demo_session.df_ppg_replay)
    assert len(replay_res.df_windows) >= 8, f"Expected >=8 windows, got {len(replay_res.df_windows)}"
    assert len(replay_res.df_predictions) >= 3, f"Expected >=3 predictions, got {len(replay_res.df_predictions)}"
    assert replay_res.summary["mean_chunk_latency_ms"] < 10.0, "Mean latency should be well under budget"
    print(f"  ✓ Replay simulation completed: {len(replay_res.df_windows)} windows, {len(replay_res.df_predictions)} predictions.")
    print(f"    Host Latency: Mean = {replay_res.summary['mean_chunk_latency_ms']:.4f} ms, Headroom = {replay_res.summary['host_budget_headroom_x']:.1f}x")

    # 5. Test Deterministic BP Pairing
    print("\n[TEST 5] Testing Predefined Deterministic BP Pairing Rule...")
    df_pairs, pairing_audit = pair_predictions_with_reference(
        df_predictions=replay_res.df_predictions,
        df_bp_events=demo_session.df_bp_events,
        session_id=demo_session.session_id,
        tolerance_s=DEFAULT_PAIRING_TOLERANCE_S,
        exclude_warn_windows=False
    )
    assert len(df_pairs) >= 2, f"Expected >=2 pairs, got {len(df_pairs)}"
    assert pairing_audit["matched_pairs_count"] >= 2, f"Expected matched pairs, got {pairing_audit['matched_pairs_count']}"
    assert "pairing_delta_s" in df_pairs.columns
    print(f"  ✓ BP pairing completed: {pairing_audit['matched_pairs_count']} matched pairs within ±{DEFAULT_PAIRING_TOLERANCE_S}s tolerance.")

    # 6. Test Statistical Metrics Computation
    print("\n[TEST 6] Testing Validation Metrics & Stratification...")
    val_metrics = compute_validation_summary(df_pairs)
    assert val_metrics["n_valid_matched_pairs"] >= 2, f"Expected valid pairs >=2, got {val_metrics['n_valid_matched_pairs']}"
    cal_s = val_metrics["calibrated"]["sbp"]
    cal_d = val_metrics["calibrated"]["dbp"]
    assert not np.isnan(cal_s["mae"]), "SBP MAE must not be NaN"
    assert not np.isnan(cal_d["mae"]), "DBP MAE must not be NaN"
    print(f"  ✓ Metrics computed on N={val_metrics['n_valid_matched_pairs']} pairs:")
    print(f"    Calibrated SBP: MAE = {cal_s['mae']:.2f} mmHg, Bias = {cal_s['bias_mean_error']:+.2f} mmHg, SD = {cal_s['std_error']:.2f}")
    print(f"    Calibrated DBP: MAE = {cal_d['mae']:.2f} mmHg, Bias = {cal_d['bias_mean_error']:+.2f} mmHg, SD = {cal_d['std_error']:.2f}")

    # 7. Test Exporting Deliverables and Reports
    print("\n[TEST 7] Testing Full Package Export & Report Synthesis...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        saved_files, zip_out_bytes = export_full_results(
            output_dir=tmp_path,
            session_data=demo_session,
            audit_results=audit_res,
            df_windows=replay_res.df_windows,
            df_predictions=replay_res.df_predictions,
            streaming_traces=replay_res.streaming_traces,
            replay_summary=replay_res.summary,
            df_pairs=df_pairs,
            pairing_audit=pairing_audit,
            validation_metrics=val_metrics,
        )
        assert (tmp_path / "phase6c_validation_report.md").exists(), "Markdown report missing"
        assert (tmp_path / "phase6c_validation_report.json").exists(), "JSON report missing"
        assert (tmp_path / "session_audit.json").exists(), "Audit JSON missing"
        assert (tmp_path / "phase6c_prediction_reference_pairs.csv").exists(), "Pairs CSV missing"
        assert (tmp_path / "figures" / "01_raw_ir_bp_timeline.png").exists(), "Figure 01 missing"
        assert (tmp_path / "figures" / "06_bland_altman_agreement.png").exists(), "Figure 06 missing"
        assert len(zip_out_bytes) > 5000, "ZIP bundle empty"
        print(f"  ✓ Export successfully created {len(saved_files)} files in temporary output directory.")

    print("\n" + "=" * 80)
    print("  ALL PHASE 6C AUTOMATED TESTS PASSED SUCCESSFULLY! (100% GREEN)")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_test()
