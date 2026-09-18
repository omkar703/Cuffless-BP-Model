#!/usr/bin/env python3
"""
CLI Runner Script for Phase 3A: PPG-Only Classical Baseline, Evaluation & Error Analysis.

Usage:
    python code/scripts/run_classical_baselines.py --sample-run
    python code/scripts/run_classical_baselines.py --full-run
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

# Ensure project code root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config.config import (
    DATASET_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    WINDOWS_DIR,
    SPLITS_DIR,
    WINDOW_MANIFEST_FILENAME,
    RECORD_SPLIT_FILENAME,
    SAMPLING_RATE,
    WINDOW_SAMPLES,
)
from utils.logging_utils import setup_logger
from modeling.data_loader import load_eligible_manifest, stream_window_ppg
from modeling.features import (
    extract_features_from_ppg_window,
    BRANCH_A_FEATURES,
    BRANCH_B_FEATURES,
    BRANCH_C_FEATURES,
    FEATURE_GROUPS,
    assert_feature_integrity,
)
from modeling.preprocessing_pipeline import (
    FeatureCacheManager,
    LeakageSafePreprocessor,
    CANONICAL_H5_CACHE_PATH,
)
from modeling.classical_models import train_model, get_model_instance
from modeling.evaluation import (
    compute_regression_metrics,
    compute_dummy_improvements,
    compute_record_weighted_metrics,
    compute_clustered_correlation,
)
from modeling.error_analysis import (
    audit_bp_range_coverage,
    compute_bp_range_errors,
    compute_quality_stratified_errors,
    compute_feature_group_importance,
    run_feature_group_ablation,
)
import visualization.model_plots as mplots

logger = setup_logger("run_classical_baselines")

FEATURES_DIR = OUTPUT_DIR / "features"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [FEATURES_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Helper: Extract Features with Parallel Worker Pool
# ==============================================================================
def extract_dataset_features(
    df_manifest: pd.DataFrame,
    n_jobs: int = 8,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Streams windows from MAT files and extracts handcrafted features in parallel."""
    logger.info(f"Extracting features for {len(df_manifest)} windows using {n_jobs} workers...")
    t0 = time.time()

    # Pre-allocate records by part for streaming
    grouped = df_manifest.groupby("part_id", sort=True)

    part_cache_dir = FEATURES_DIR / "part_caches"
    part_cache_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    processed_count = 0
    total_windows = len(df_manifest)

    for part_id, group in grouped:
        part_cache_file = part_cache_dir / f"{part_id}_features.h5"
        if part_cache_file.exists():
            logger.info(f"Part {part_id} cache exists at {part_cache_file.name}. Loading...")
            part_df = FeatureCacheManager.load_cache(part_cache_file)
            all_dfs.append(part_df)
            processed_count += len(part_df)
            logger.info(f"Loaded {part_id} ({len(part_df)} windows, {processed_count}/{total_windows} total).")
            continue

        part_t0 = time.time()
        logger.info(f"Processing {part_id} ({len(group)} windows)...")

        window_stream = list(stream_window_ppg(group))
        if not window_stream:
            continue

        feature_dicts = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(extract_features_from_ppg_window)(w["ppg"]) for w in window_stream
        )

        part_rows = []
        for w, f_dict in zip(window_stream, feature_dicts):
            row = {
                "record_id": w["record_id"],
                "window_id": w["window_id"],
                "split": w["split"],
                "sbp": w["sbp"],
                "dbp": w["dbp"],
                "map": w["map"],
                "ppg_quality_status": w["ppg_quality_status"],
                "window_quality_status": w["window_quality_status"],
                "ppg_clipped_fraction": w["ppg_clipped_fraction"],
            }
            row.update(f_dict)
            part_rows.append(row)

        part_df = pd.DataFrame(part_rows)
        # Save part cache
        FeatureCacheManager.save_cache(part_df, part_cache_file)
        all_dfs.append(part_df)

        processed_count += len(window_stream)
        part_dt = time.time() - part_t0
        logger.info(
            f"Completed & cached {part_id}: {len(window_stream)} windows in {part_dt:.1f} s "
            f"({processed_count}/{total_windows} total, {1000 * part_dt / len(window_stream):.2f} ms/win)."
        )

        del window_stream, feature_dicts, part_rows, part_df
        gc.collect()

    df_feats = pd.concat(all_dfs, axis=0).reset_index(drop=True)
    total_dt = time.time() - t0
    logger.info(
        f"Feature extraction complete! {len(df_feats)} windows processed in {total_dt/60:.2f} min."
    )
    return df_feats


# ==============================================================================
# Main Pipeline Runner
# ==============================================================================
def run_pipeline(
    is_sample_run: bool = False,
    force_recompute: bool = False,
    n_jobs: int = 8,
) -> None:
    pipeline_t0 = time.time()
    mode_str = "SAMPLE-RUN (1,000 windows)" if is_sample_run else "FULL-RUN (261,339 windows)"
    logger.info("=" * 80)
    logger.info(f"STARTING PHASE 3A CLASSICAL BASELINES PIPELINE: {mode_str}")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Phase 2 Manifest & Record Split Verification
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 1: Phase 2 Manifest & Record Split Verification ---")
    manifest_file = WINDOWS_DIR / WINDOW_MANIFEST_FILENAME
    split_file = SPLITS_DIR / RECORD_SPLIT_FILENAME

    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest missing: {manifest_file}")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file missing: {split_file}")

    df_manifest_raw = pd.read_csv(manifest_file)
    df_splits = pd.read_csv(split_file)

    train_recs = set(df_splits[df_splits["split"] == "train"]["record_id"])
    val_recs = set(df_splits[df_splits["split"] == "val"]["record_id"])
    test_recs = set(df_splits[df_splits["split"] == "test"]["record_id"])

    logger.info(f"Split Records: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")
    assert len(train_recs.intersection(val_recs)) == 0, "DATA LEAKAGE: Train & Val overlap!"
    assert len(train_recs.intersection(test_recs)) == 0, "DATA LEAKAGE: Train & Test overlap!"
    assert len(val_recs.intersection(test_recs)) == 0, "DATA LEAKAGE: Val & Test overlap!"
    logger.info("ZERO RECORD LEAKAGE VERIFIED: Pairwise record intersections are all 0.")

    df_eligible = df_manifest_raw[df_manifest_raw["modeling_eligible"] == True].copy()
    logger.info(f"Total modeling-eligible windows: {len(df_eligible)}")

    # -------------------------------------------------------------------------
    # STEP 2: BP Range Coverage Audit Prior to Training
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 2: BP Range Coverage Audit Prior to Modeling ---")
    df_bp_cov, bp_summary_stats = audit_bp_range_coverage(
        df_eligible, save_path=METRICS_DIR / "bp_range_coverage.csv"
    )
    logger.info("BP Range Coverage Summary:")
    for _, r in df_bp_cov.iterrows():
        logger.info(
            f"  {r['target']} [{r['range']:<8}]: Train={r['train_count']:>6} ({r['train_percentage']:>4.1f}%) | "
            f"Val={r['validation_count']:>5} ({r['validation_percentage']:>4.1f}%) | "
            f"Test={r['test_count']:>5} ({r['test_percentage']:>4.1f}%)"
        )

    # -------------------------------------------------------------------------
    # STEP 3: Feature Extraction / Loading from Canonical Cache
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 3: Feature Extraction / Cache Check ---")
    h5_cache_path = CANONICAL_H5_CACHE_PATH if not is_sample_run else (FEATURES_DIR / "sample_features_cache.h5")

    if is_sample_run:
        # Deterministic sample run on 1,000 windows across 2 parts
        sample_train = df_eligible[df_eligible["split"] == "train"].head(700)
        sample_val = df_eligible[df_eligible["split"] == "val"].head(150)
        sample_test = df_eligible[df_eligible["split"] == "test"].head(150)
        df_target_manifest = pd.concat([sample_train, sample_val, sample_test], axis=0).reset_index(drop=True)
        logger.info(f"Sample run subset created: {len(df_target_manifest)} windows.")
    else:
        df_target_manifest = df_eligible

    if h5_cache_path.exists() and not force_recompute:
        logger.info(f"Found existing feature cache at {h5_cache_path}. Loading...")
        df_features = FeatureCacheManager.load_cache(h5_cache_path)
        if len(df_features) != len(df_target_manifest):
            logger.warning(
                f"Cache length {len(df_features)} != manifest length {len(df_target_manifest)}. Recomputing..."
            )
            df_features = extract_dataset_features(df_target_manifest, n_jobs=n_jobs)
            FeatureCacheManager.save_cache(df_features, h5_cache_path)
    else:
        df_features = extract_dataset_features(df_target_manifest, n_jobs=n_jobs)
        FeatureCacheManager.save_cache(df_features, h5_cache_path)

    # -------------------------------------------------------------------------
    # STEP 4: Feature Integrity Audit
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 4: Feature Integrity Audit ---")
    meta_cols = [
        "record_id",
        "window_id",
        "split",
        "sbp",
        "dbp",
        "map",
        "ppg_quality_status",
        "window_quality_status",
        "ppg_clipped_fraction",
    ]
    extracted_feature_cols = [c for c in df_features.columns if c not in meta_cols]
    assert_feature_integrity(extracted_feature_cols)
    logger.info(
        f"Feature integrity verified: 0 forbidden terms found across {len(extracted_feature_cols)} features."
    )

    # Partition features
    df_train = df_features[df_features["split"] == "train"].copy()
    df_val = df_features[df_features["split"] == "val"].copy()
    df_test = df_features[df_features["split"] == "test"].copy()

    logger.info(
        f"Partition Sizes: Train={len(df_train)} windows, Val={len(df_val)} windows, Test={len(df_test)} windows."
    )

    # -------------------------------------------------------------------------
    # STEP 5: Cumulative Feature Group Ablation
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 5: Cumulative Feature Group Ablation ---")
    df_ablation = run_feature_group_ablation(
        df_train,
        df_val,
        model_name="hist_gradient_boosting",
        save_path=METRICS_DIR / "feature_ablation_results.csv",
    )
    logger.info("Ablation Results Summary:")
    for _, r in df_ablation.iterrows():
        logger.info(
            f"  {r['ablation']:<30} | {r['target']} | Feats={r['feature_count']:>2} | "
            f"MAE={r['MAE']:>5.2f} mmHg | RMSE={r['RMSE']:>5.2f} mmHg | R²={r['R2']:>6.3f}"
        )
    mplots.plot_feature_group_ablation(
        df_ablation, save_path=FIGURES_DIR / "model_18_feature_group_ablation.png"
    )

    # -------------------------------------------------------------------------
    # STEP 6: Representation Branches & Baseline Model Comparison (on Validation)
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 6: Representation Branches & Baseline Model Evaluation on VALIDATION ---")

    branches = {
        "Branch_A_Amplitude_Preserving": BRANCH_A_FEATURES,
        "Branch_B_Normalized_Morphology": BRANCH_B_FEATURES,
        "Branch_C_Combined": BRANCH_C_FEATURES,
    }

    models_to_test = [
        "dummy",
        "linear",
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    ]

    all_val_results = []
    trained_val_models = {}  # (branch, model_name, target) -> fitted model
    val_preprocessors = {}   # branch -> preprocessor

    # Pre-fit preprocessors for all 3 branches
    for b_name, b_feats in branches.items():
        preproc = LeakageSafePreprocessor(b_feats)
        preproc.fit_train(df_train)
        val_preprocessors[b_name] = preproc

    # Compute Dummy baseline once
    dummy_metrics_dict = {}
    for tgt in ["SBP", "DBP"]:
        tgt_lower = tgt.lower()
        y_tr = df_train[tgt_lower].to_numpy()
        y_v = df_val[tgt_lower].to_numpy()
        dummy_pred = np.full_like(y_v, fill_value=np.mean(y_tr))
        d_metrics = compute_regression_metrics(y_v, dummy_pred, target_name=tgt)
        dummy_metrics_dict[tgt] = d_metrics

    for b_name, b_feats in branches.items():
        preproc = val_preprocessors[b_name]
        X_tr = preproc.transform(df_train)
        X_v = preproc.transform(df_val)

        for m_name in models_to_test:
            for tgt in ["SBP", "DBP"]:
                tgt_lower = tgt.lower()
                y_tr = df_train[tgt_lower].to_numpy()
                y_v = df_val[tgt_lower].to_numpy()

                if m_name == "dummy":
                    m_metrics = dummy_metrics_dict[tgt]
                    m_imp = compute_dummy_improvements(m_metrics, dummy_metrics_dict[tgt])
                else:
                    fitted_m, meta = train_model(
                        m_name,
                        X_tr,
                        y_tr,
                        X_val=X_v,
                        y_val=y_v,
                        target_name=tgt,
                    )
                    trained_val_models[(b_name, m_name, tgt)] = fitted_m
                    preds_v = fitted_m.predict(X_v)
                    m_metrics = compute_regression_metrics(y_v, preds_v, target_name=tgt)
                    m_imp = compute_dummy_improvements(m_metrics, dummy_metrics_dict[tgt])

                res_row = {
                    "branch": b_name,
                    "model": m_name,
                    "target": tgt,
                    "n_features": len(b_feats),
                    "split": "validation",
                }
                res_row.update(m_metrics)
                res_row.update(m_imp)
                all_val_results.append(res_row)

    df_val_results = pd.DataFrame(all_val_results)
    df_val_results.to_csv(METRICS_DIR / "classical_baseline_results.csv", index=False)
    logger.info(f"Saved complete baseline validation results to {METRICS_DIR / 'classical_baseline_results.csv'}")

    # Model learning value summary table
    df_learning_val = df_val_results[
        [
            "branch",
            "model",
            "target",
            "dummy_mae",
            "mae",
            "delta_MAE_vs_dummy",
            "percentage_MAE_improvement_vs_dummy",
            "dummy_rmse",
            "rmse",
            "delta_RMSE_vs_dummy",
            "r2",
        ]
    ].rename(
        columns={
            "mae": "model_mae",
            "rmse": "model_rmse",
            "delta_MAE_vs_dummy": "mae_improvement",
            "delta_RMSE_vs_dummy": "rmse_improvement",
        }
    )
    df_learning_val.to_csv(METRICS_DIR / "model_learning_value.csv", index=False)
    logger.info(f"Saved model learning value summary to {METRICS_DIR / 'model_learning_value.csv'}")

    # Plot Model Comparison Bar Chart (Model 12) for Branch C
    branch_c_res = df_val_results[df_val_results["branch"] == "Branch_C_Combined"]
    mplots.plot_model_comparison(
        branch_c_res, save_path=FIGURES_DIR / "model_12_model_comparison_bar.png"
    )

    # -------------------------------------------------------------------------
    # STEP 7: Select Best Configuration on Validation & Freeze
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 7: Validation Configuration Selection & FREEZE ---")
    # Identify non-dummy models and rank by combined SBP+DBP MAE
    df_non_dummy = df_val_results[df_val_results["model"] != "dummy"].copy()
    grouped_cfg = (
        df_non_dummy.groupby(["branch", "model"])
        .agg(mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"))
        .reset_index()
        .sort_values(by="mean_mae")
    )

    best_branch = grouped_cfg.iloc[0]["branch"]
    best_model_name = grouped_cfg.iloc[0]["model"]
    best_val_mean_mae = grouped_cfg.iloc[0]["mean_mae"]

    logger.info(f"BEST VALIDATION CONFIGURATION: Branch={best_branch} | Model={best_model_name} (Val Combined MAE: {best_val_mean_mae:.2f} mmHg)")
    logger.info("Ranking of all evaluated configurations on VALIDATION:")
    for rank_idx, r in grouped_cfg.iterrows():
        logger.info(f"  Rank: Branch={r['branch']:<32} | Model={r['model']:<25} | Mean MAE={r['mean_mae']:.2f} mmHg")

    frozen_config = {
        "model": best_model_name,
        "representation_branch": best_branch,
        "features": branches[best_branch],
        "feature_count": len(branches[best_branch]),
        "hyperparameters": {
            "max_iter": 150 if best_model_name == "hist_gradient_boosting" else None,
            "max_depth": 8 if best_model_name == "hist_gradient_boosting" else (12 if best_model_name == "random_forest" else None),
            "random_state": 42,
        },
        "imputation": "SimpleImputer(strategy='median') fit on TRAIN",
        "scaling": "StandardScaler() fit on TRAIN",
        "val_combined_mae": float(best_val_mean_mae),
    }

    with open(MODELS_DIR / "frozen_configuration.json", "w") as f:
        json.dump(frozen_config, f, indent=2)
    logger.info("Configuration FROZEN and saved to models/frozen_configuration.json. NO FURTHER TUNING.")

    # -------------------------------------------------------------------------
    # STEP 8: Final TEST Set Evaluation (Executed ONCE)
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 8: Final TEST Set Evaluation (Evaluated ONCE) ---")
    frozen_preproc = val_preprocessors[best_branch]
    X_test = frozen_preproc.transform(df_test)

    best_sbp_model = trained_val_models[(best_branch, best_model_name, "SBP")]
    best_dbp_model = trained_val_models[(best_branch, best_model_name, "DBP")]

    # Save trained model artifacts
    joblib.dump(best_sbp_model, MODELS_DIR / "best_sbp_model.joblib")
    joblib.dump(best_dbp_model, MODELS_DIR / "best_dbp_model.joblib")
    logger.info("Saved best model weights to models/best_sbp_model.joblib and best_dbp_model.joblib.")

    sbp_pred_test = best_sbp_model.predict(X_test)
    dbp_pred_test = best_dbp_model.predict(X_test)

    # Window-weighted test metrics
    sbp_test_metrics = compute_regression_metrics(
        df_test["sbp"].to_numpy(), sbp_pred_test, target_name="SBP"
    )
    dbp_test_metrics = compute_regression_metrics(
        df_test["dbp"].to_numpy(), dbp_pred_test, target_name="DBP"
    )

    # Dummy test metrics for relative improvement
    dummy_sbp_test_pred = np.full_like(df_test["sbp"].to_numpy(), fill_value=np.mean(df_train["sbp"].to_numpy()))
    dummy_dbp_test_pred = np.full_like(df_test["dbp"].to_numpy(), fill_value=np.mean(df_train["dbp"].to_numpy()))
    dummy_sbp_test_m = compute_regression_metrics(df_test["sbp"].to_numpy(), dummy_sbp_test_pred, target_name="SBP")
    dummy_dbp_test_m = compute_regression_metrics(df_test["dbp"].to_numpy(), dummy_dbp_test_pred, target_name="DBP")

    sbp_imp_test = compute_dummy_improvements(sbp_test_metrics, dummy_sbp_test_m)
    dbp_imp_test = compute_dummy_improvements(dbp_test_metrics, dummy_dbp_test_m)

    logger.info("FINAL TEST RESULTS (Window-Weighted):")
    logger.info(
        f"  SBP: MAE={sbp_test_metrics['mae']:.2f} mmHg | RMSE={sbp_test_metrics['rmse']:.2f} mmHg | "
        f"R²={sbp_test_metrics['r2']:.3f} | Bias={sbp_test_metrics['bias']:.2f} | SD={sbp_test_metrics['error_sd']:.2f} | "
        f"<=5mmHg: {sbp_test_metrics['pct_le_5']:.1f}% | <=10mmHg: {sbp_test_metrics['pct_le_10']:.1f}% | "
        f"Dummy Imp: {sbp_imp_test['percentage_MAE_improvement_vs_dummy']:.1f}%"
    )
    logger.info(
        f"  DBP: MAE={dbp_test_metrics['mae']:.2f} mmHg | RMSE={dbp_test_metrics['rmse']:.2f} mmHg | "
        f"R²={dbp_test_metrics['r2']:.3f} | Bias={dbp_test_metrics['bias']:.2f} | SD={dbp_test_metrics['error_sd']:.2f} | "
        f"<=5mmHg: {dbp_test_metrics['pct_le_5']:.1f}% | <=10mmHg: {dbp_test_metrics['pct_le_10']:.1f}% | "
        f"Dummy Imp: {dbp_imp_test['percentage_MAE_improvement_vs_dummy']:.1f}%"
    )

    # Assemble prediction DataFrame
    df_preds = df_test[
        [
            "record_id",
            "window_id",
            "sbp",
            "dbp",
            "map",
            "ppg_quality_status",
            "window_quality_status",
            "ppg_clipped_fraction",
            "hr_bpm",
            "pulse_amp_cv_a",
        ]
    ].copy()
    df_preds["sbp_pred"] = sbp_pred_test
    df_preds["dbp_pred"] = dbp_pred_test
    df_preds["sbp_error"] = sbp_pred_test - df_test["sbp"].to_numpy()
    df_preds["dbp_error"] = dbp_pred_test - df_test["dbp"].to_numpy()
    df_preds["sbp_abs_error"] = np.abs(df_preds["sbp_error"])
    df_preds["dbp_abs_error"] = np.abs(df_preds["dbp_error"])

    pred_save_path = PREDICTIONS_DIR / "baseline_test_predictions.csv"
    df_preds.to_csv(pred_save_path, index=False)
    logger.info(f"Saved test predictions to {pred_save_path}")

    # -------------------------------------------------------------------------
    # STEP 9: Dual Evaluation (Record-Weighted vs Window-Weighted)
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 9: Record-Weighted vs Window-Weighted Analysis ---")
    rec_sum_sbp, rec_df_sbp = compute_record_weighted_metrics(
        df_preds, "sbp", "sbp_pred", target_name="SBP"
    )
    rec_sum_dbp, rec_df_dbp = compute_record_weighted_metrics(
        df_preds, "dbp", "dbp_pred", target_name="DBP"
    )

    logger.info("RECORD-WEIGHTED TEST RESULTS:")
    logger.info(
        f"  SBP: Mean Record MAE={rec_sum_sbp['mean_record_mae']:.2f} mmHg | "
        f"Median={rec_sum_sbp['median_record_mae']:.2f} | SD={rec_sum_sbp['std_record_mae']:.2f} | "
        f"IQR={rec_sum_sbp['iqr_record_mae']:.2f} (across {rec_sum_sbp['n_records']} records)"
    )
    logger.info(
        f"  DBP: Mean Record MAE={rec_sum_dbp['mean_record_mae']:.2f} mmHg | "
        f"Median={rec_sum_dbp['median_record_mae']:.2f} | SD={rec_sum_dbp['std_record_mae']:.2f} | "
        f"IQR={rec_sum_dbp['iqr_record_mae']:.2f} (across {rec_sum_dbp['n_records']} records)"
    )

    # Plot Model 11 & Model 16
    mplots.plot_record_level_distribution(
        rec_df_sbp, rec_df_dbp, save_path=FIGURES_DIR / "model_11_record_level_mae_distribution.png"
    )
    mplots.plot_record_vs_window_weighted(
        win_mae_sbp=sbp_test_metrics["mae"],
        rec_mae_mean_sbp=rec_sum_sbp["mean_record_mae"],
        rec_mae_med_sbp=rec_sum_sbp["median_record_mae"],
        win_mae_dbp=dbp_test_metrics["mae"],
        rec_mae_mean_dbp=rec_sum_dbp["mean_record_mae"],
        rec_mae_med_dbp=rec_sum_dbp["median_record_mae"],
        save_path=FIGURES_DIR / "model_16_record_weighted_vs_window_weighted.png",
    )

    # -------------------------------------------------------------------------
    # STEP 10: Granular Error Analysis (BP Range & Quality Strata)
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 10: BP Range & Signal Quality Error Analysis ---")
    df_range_sbp = compute_bp_range_errors(
        df_preds,
        "sbp",
        "sbp_pred",
        target_name="SBP",
        save_path=METRICS_DIR / "bp_range_error_analysis_sbp.csv",
    )
    df_range_dbp = compute_bp_range_errors(
        df_preds,
        "dbp",
        "dbp_pred",
        target_name="DBP",
        save_path=METRICS_DIR / "bp_range_error_analysis_dbp.csv",
    )
    df_range_both = pd.concat([df_range_sbp, df_range_dbp], axis=0)
    df_range_both.to_csv(METRICS_DIR / "bp_range_error_analysis.csv", index=False)

    mplots.plot_mae_by_bp_range(
        df_range_sbp, "SBP", save_path=FIGURES_DIR / "model_09_sbp_mae_by_bp_range.png"
    )
    mplots.plot_mae_by_bp_range(
        df_range_dbp, "DBP", save_path=FIGURES_DIR / "model_10_dbp_mae_by_bp_range.png"
    )

    df_qual_sbp = compute_quality_stratified_errors(df_preds, "sbp", "sbp_pred", target_name="SBP")
    df_qual_dbp = compute_quality_stratified_errors(df_preds, "dbp", "dbp_pred", target_name="DBP")
    mplots.plot_error_vs_quality(
        df_qual_sbp, df_qual_dbp, save_path=FIGURES_DIR / "model_14_error_vs_ppg_quality.png"
    )

    # Scatter, Residuals, Bland-Altman, Error vs Ref
    mplots.plot_true_vs_pred(
        df_preds["sbp"].to_numpy(),
        sbp_pred_test,
        "SBP",
        save_path=FIGURES_DIR / "model_01_sbp_true_vs_pred.png",
        r2_val=sbp_test_metrics["r2"],
        mae_val=sbp_test_metrics["mae"],
    )
    mplots.plot_true_vs_pred(
        df_preds["dbp"].to_numpy(),
        dbp_pred_test,
        "DBP",
        save_path=FIGURES_DIR / "model_02_dbp_true_vs_pred.png",
        r2_val=dbp_test_metrics["r2"],
        mae_val=dbp_test_metrics["mae"],
    )
    mplots.plot_residuals(
        df_preds["sbp"].to_numpy(),
        sbp_pred_test,
        "SBP",
        save_path=FIGURES_DIR / "model_03_sbp_residuals.png",
    )
    mplots.plot_residuals(
        df_preds["dbp"].to_numpy(),
        dbp_pred_test,
        "DBP",
        save_path=FIGURES_DIR / "model_04_dbp_residuals.png",
    )
    mplots.plot_bland_altman(
        df_preds["sbp"].to_numpy(),
        sbp_pred_test,
        "SBP",
        save_path=FIGURES_DIR / "model_05_sbp_bland_altman.png",
    )
    mplots.plot_bland_altman(
        df_preds["dbp"].to_numpy(),
        dbp_pred_test,
        "DBP",
        save_path=FIGURES_DIR / "model_06_dbp_bland_altman.png",
    )
    mplots.plot_error_vs_ref(
        df_preds["sbp"].to_numpy(),
        sbp_pred_test,
        "SBP",
        save_path=FIGURES_DIR / "model_07_sbp_error_vs_ref.png",
    )
    mplots.plot_error_vs_ref(
        df_preds["dbp"].to_numpy(),
        dbp_pred_test,
        "DBP",
        save_path=FIGURES_DIR / "model_08_dbp_error_vs_ref.png",
    )
    mplots.plot_error_vs_hr(
        df_preds, save_path=FIGURES_DIR / "model_13_error_vs_heart_rate.png"
    )
    mplots.plot_error_vs_amplitude_variation(
        df_preds, save_path=FIGURES_DIR / "model_15_error_vs_pulse_amplitude_variation.png"
    )

    # -------------------------------------------------------------------------
    # STEP 11: Feature Group Importance
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 11: Feature Group Importance ---")
    X_val_frozen = frozen_preproc.transform(df_val)
    df_grp_imp = compute_feature_group_importance(
        best_sbp_model,
        branches[best_branch],
        X_val=X_val_frozen,
        y_val=df_val["sbp"].to_numpy(),
        save_path=METRICS_DIR / "feature_group_importance.csv",
        target_name="SBP",
    )
    mplots.plot_feature_group_importance(
        df_grp_imp, save_path=FIGURES_DIR / "model_17_feature_group_importance.png"
    )
    logger.info("Feature Group Importance Summary (SBP):")
    for _, r in df_grp_imp.iterrows():
        logger.info(
            f"  Group: {r['group']:<15} | Count={r['feature_count']:>2} | Total={r['total_importance']:>7.2f} | Share={r['importance_percentage']:>5.1f}%"
        )

    # -------------------------------------------------------------------------
    # STEP 12: Record-Clustered Error Correlation Analysis
    # -------------------------------------------------------------------------
    logger.info("\n--- STEP 12: Record-Clustered Error Correlation Analysis ---")
    corr_features = [
        "hr_bpm",
        "pulse_amp_median_a",
        "pulse_amp_cv_a",
        "std_a",
        "ptp_a",
        "ibi_std",
        "sbp",
    ]
    corr_rows = []
    for f in corr_features:
        if f in df_preds.columns:
            res_c = compute_clustered_correlation(
                df_preds, feature_col=f, error_col="sbp_abs_error", n_bootstraps=500
            )
            res_c["target"] = "SBP"
            corr_rows.append(res_c)

    df_corrs = pd.DataFrame(corr_rows)
    df_corrs.to_csv(METRICS_DIR / "error_correlations.csv", index=False)
    logger.info(f"Saved clustered correlation analysis to {METRICS_DIR / 'error_correlations.csv'}")

    # Summary table of final test metrics
    df_final_test = pd.DataFrame(
        [
            {
                "target": "SBP",
                "model": best_model_name,
                "branch": best_branch,
                "MAE": sbp_test_metrics["mae"],
                "RMSE": sbp_test_metrics["rmse"],
                "R2": sbp_test_metrics["r2"],
                "bias": sbp_test_metrics["bias"],
                "error_sd": sbp_test_metrics["error_sd"],
                "pct_le_5": sbp_test_metrics["pct_le_5"],
                "pct_le_10": sbp_test_metrics["pct_le_10"],
                "pct_le_15": sbp_test_metrics["pct_le_15"],
                "dummy_mae": dummy_sbp_test_m["mae"],
                "mae_improvement": sbp_imp_test["delta_MAE_vs_dummy"],
                "pct_mae_improvement": sbp_imp_test["percentage_MAE_improvement_vs_dummy"],
                "mean_record_mae": rec_sum_sbp["mean_record_mae"],
                "median_record_mae": rec_sum_sbp["median_record_mae"],
            },
            {
                "target": "DBP",
                "model": best_model_name,
                "branch": best_branch,
                "MAE": dbp_test_metrics["mae"],
                "RMSE": dbp_test_metrics["rmse"],
                "R2": dbp_test_metrics["r2"],
                "bias": dbp_test_metrics["bias"],
                "error_sd": dbp_test_metrics["error_sd"],
                "pct_le_5": dbp_test_metrics["pct_le_5"],
                "pct_le_10": dbp_test_metrics["pct_le_10"],
                "pct_le_15": dbp_test_metrics["pct_le_15"],
                "dummy_mae": dummy_dbp_test_m["mae"],
                "mae_improvement": dbp_imp_test["delta_MAE_vs_dummy"],
                "pct_mae_improvement": dbp_imp_test["percentage_MAE_improvement_vs_dummy"],
                "mean_record_mae": rec_sum_dbp["mean_record_mae"],
                "median_record_mae": rec_sum_dbp["median_record_mae"],
            },
        ]
    )
    df_final_test.to_csv(METRICS_DIR / "final_test_summary.csv", index=False)
    logger.info(f"Saved final test summary to {METRICS_DIR / 'final_test_summary.csv'}")

    total_pipeline_time = time.time() - pipeline_t0
    logger.info("=" * 80)
    logger.info(f"PHASE 3A PIPELINE EXECUTION FINISHED in {total_pipeline_time/60:.2f} minutes.")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3A Classical Baselines Runner")
    parser.add_argument("--sample-run", action="store_true", help="Fast sample run on 1,000 windows")
    parser.add_argument("--full-run", action="store_true", help="Complete run across all 261,339 eligible windows")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation of feature cache")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of parallel worker processes (default: 8)")

    args = parser.parse_args()

    if not args.sample_run and not args.full_run:
        logger.error("Must specify either --sample-run or --full-run. Exiting.")
        sys.exit(1)

    run_pipeline(
        is_sample_run=args.sample_run,
        force_recompute=args.force_recompute,
        n_jobs=args.n_jobs,
    )
