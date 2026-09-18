"""
Error Analysis, BP Coverage Audit, and Feature Ablation Engine for Phase 3A.

Implements:
1. BP Range Coverage Audit (bp_range_coverage.csv)
2. BP Range Error Analysis with sample density (bp_range_error_analysis.csv)
3. Quality-Stratified Analysis (PASS vs WARN, clipping, variability)
4. Record-Level Generalization & Outliers (best/worst records)
5. Feature Importance Aggregation (feature_group_importance.csv)
6. Feature-Group Cumulative Ablation (feature_ablation_results.csv)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.config import OUTPUT_DIR
from modeling.features import FEATURE_GROUPS, assert_feature_integrity
from modeling.classical_models import train_model
from modeling.preprocessing_pipeline import LeakageSafePreprocessor
from utils.logging_utils import setup_logger

logger = setup_logger("error_analysis")

METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. BP Range Coverage Audit
# ==============================================================================
def audit_bp_range_coverage(
    df_manifest: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculates target distribution coverage across train, val, and test partitions.
    Saves to code/outputs/metrics/bp_range_coverage.csv.
    """
    logger.info("Computing BP range coverage audit across partitions...")
    out_path = save_path or (METRICS_DIR / "bp_range_coverage.csv")

    sbp_bins = [-np.inf, 90, 120, 140, 160, np.inf]
    sbp_labels = ["<90", "90-119", "120-139", "140-159", ">=160"]

    dbp_bins = [-np.inf, 60, 80, 90, 100, np.inf]
    dbp_labels = ["<60", "60-79", "80-89", "90-99", ">=100"]

    df = df_manifest.copy()
    df["sbp_range"] = pd.cut(df["sbp"], bins=sbp_bins, labels=sbp_labels, right=False)
    df["dbp_range"] = pd.cut(df["dbp"], bins=dbp_bins, labels=dbp_labels, right=False)

    rows = []

    for target, col_range, labels in [
        ("SBP", "sbp_range", sbp_labels),
        ("DBP", "dbp_range", dbp_labels),
    ]:
        train_sub = df[df["split"] == "train"]
        val_sub = df[df["split"] == "val"]
        test_sub = df[df["split"] == "test"]

        n_train_tot = len(train_sub)
        n_val_tot = len(val_sub)
        n_test_tot = len(test_sub)

        for lbl in labels:
            c_train = int((train_sub[col_range] == lbl).sum())
            c_val = int((val_sub[col_range] == lbl).sum())
            c_test = int((test_sub[col_range] == lbl).sum())

            rows.append(
                {
                    "target": target,
                    "range": lbl,
                    "train_count": c_train,
                    "validation_count": c_val,
                    "test_count": c_test,
                    "train_percentage": float(100.0 * c_train / max(1, n_train_tot)),
                    "validation_percentage": float(100.0 * c_val / max(1, n_val_tot)),
                    "test_percentage": float(100.0 * c_test / max(1, n_test_tot)),
                }
            )

    df_coverage = pd.DataFrame(rows)
    df_coverage.to_csv(out_path, index=False)
    logger.info(f"Saved BP range coverage audit to {out_path}.")

    # Summary continuous statistics
    summary_stats = {}
    for tgt, tgt_col in [("SBP", "sbp"), ("DBP", "dbp")]:
        summary_stats[tgt] = {}
        for s in ["train", "val", "test"]:
            sub = df[df["split"] == s][tgt_col].dropna()
            summary_stats[tgt][s] = {
                "min": float(sub.min()),
                "max": float(sub.max()),
                "mean": float(sub.mean()),
                "median": float(sub.median()),
                "std": float(sub.std()),
            }

    return df_coverage, summary_stats


# ==============================================================================
# 2. BP Range Error Analysis
# ==============================================================================
def compute_bp_range_errors(
    df_eval: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    target_name: str = "SBP",
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Evaluates model error across physiological blood pressure ranges along with sample counts.
    Saves to code/outputs/metrics/bp_range_error_analysis.csv.
    """
    df = df_eval.copy()
    errors = df[y_pred_col] - df[y_true_col]
    df["error"] = errors
    df["abs_error"] = np.abs(errors)

    if target_name.upper() == "SBP":
        bins = [-np.inf, 90, 120, 140, 160, np.inf]
        labels = ["<90", "90-119", "120-139", "140-159", ">=160"]
    else:
        bins = [-np.inf, 60, 80, 90, 100, np.inf]
        labels = ["<60", "60-79", "80-89", "90-99", ">=100"]

    df["bp_range"] = pd.cut(df[y_true_col], bins=bins, labels=labels, right=False)

    rows = []
    for lbl in labels:
        sub = df[df["bp_range"] == lbl]
        cnt = len(sub)
        if cnt > 0:
            mae = float(sub["abs_error"].mean())
            rmse = float(np.sqrt((sub["error"] ** 2).mean()))
            bias = float(sub["error"].mean())
            err_sd = float(sub["error"].std())
        else:
            mae, rmse, bias, err_sd = np.nan, np.nan, np.nan, np.nan

        rows.append(
            {
                "target": target_name,
                "range": lbl,
                "sample_count": cnt,
                "MAE": mae,
                "RMSE": rmse,
                "bias": bias,
                "error_std": err_sd,
            }
        )

    df_out = pd.DataFrame(rows)
    if save_path:
        df_out.to_csv(save_path, index=False)
        logger.info(f"Saved BP range error analysis to {save_path}.")
    return df_out


# ==============================================================================
# 3. Quality-Stratified Error Analysis
# ==============================================================================
def compute_quality_stratified_errors(
    df_eval: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    target_name: str = "SBP",
) -> pd.DataFrame:
    """Evaluates error on quality strata: PPG PASS vs PPG WARN, clipping, etc."""
    df = df_eval.copy()
    errors = df[y_pred_col] - df[y_true_col]
    df["error"] = errors
    df["abs_error"] = np.abs(errors)

    strata = {
        "ALL_ELIGIBLE": df,
        "PPG_PASS": df[df["ppg_quality_status"] == "PASS"],
        "PPG_WARN": df[df["ppg_quality_status"] == "WARN"],
        "CLIPPED_>0": df[df.get("ppg_clipped_fraction", 0.0) > 0.0],
        "UNCLIPPED": df[df.get("ppg_clipped_fraction", 0.0) == 0.0],
    }

    rows = []
    for name, sub in strata.items():
        cnt = len(sub)
        if cnt > 0:
            mae = float(sub["abs_error"].mean())
            rmse = float(np.sqrt((sub["error"] ** 2).mean()))
            bias = float(sub["error"].mean())
            err_sd = float(sub["error"].std())
        else:
            mae, rmse, bias, err_sd = np.nan, np.nan, np.nan, np.nan

        rows.append(
            {
                "target": target_name,
                "stratum": name,
                "sample_count": cnt,
                "MAE": mae,
                "RMSE": rmse,
                "bias": bias,
                "error_std": err_sd,
            }
        )

    return pd.DataFrame(rows)


# ==============================================================================
# 4. Feature Group Importance Aggregation
# ==============================================================================
def compute_feature_group_importance(
    model: Any,
    feature_names: List[str],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    target_name: str = "BP",
) -> pd.DataFrame:
    """
    Computes individual feature importances and aggregates total and percentage
    importance by feature group (A_basic, B_timing, C_morphology, D_vpg, E_apg, F_spectral).
    """
    out_path = save_path or (METRICS_DIR / "feature_group_importance.csv")

    # Extract raw feature importances
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For linear models use absolute normalized coefficients
        raw_importances = np.abs(model.coef_)
    else:
        from sklearn.inspection import permutation_importance
        if X_val is not None and y_val is not None:
            n_sub = min(2500, len(X_val))
            perm = permutation_importance(
                model, X_val[:n_sub], y_val[:n_sub], n_repeats=5, random_state=42
            )
            raw_importances = np.maximum(0, perm.importances_mean)
        else:
            raw_importances = np.ones(len(feature_names))

    tot_imp = float(np.sum(raw_importances)) + 1e-12
    norm_importances = raw_importances / tot_imp

    df_feats = pd.DataFrame(
        {
            "feature": feature_names,
            "raw_importance": raw_importances,
            "normalized_importance": norm_importances,
        }
    )

    # Assign group to each feature
    feat_to_group = {}
    for grp, f_list in FEATURE_GROUPS.items():
        for f in f_list:
            feat_to_group[f] = grp

    df_feats["group"] = df_feats["feature"].map(lambda f: feat_to_group.get(f, "other"))

    # Group-level aggregation
    grp_df = (
        df_feats.groupby("group")
        .agg(
            feature_count=("feature", "count"),
            total_importance=("raw_importance", "sum"),
            importance_percentage=("normalized_importance", lambda s: float(np.sum(s) * 100.0)),
        )
        .reset_index()
    )

    grp_df["target"] = target_name
    grp_df = grp_df.sort_values(by="importance_percentage", ascending=False)

    if save_path:
        grp_df.to_csv(out_path, index=False)
        logger.info(f"Saved feature group importance to {out_path}.")

    return grp_df


# ==============================================================================
# 5. Feature-Group Cumulative Ablation
# ==============================================================================
def run_feature_group_ablation(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str = "hist_gradient_boosting",
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Runs cumulative feature ablation experiments (Ablation 0 through 6)
    using the exact same frozen Phase 2 train/val splits.
    
    Ablation 0: Dummy baseline
    Ablation 1: Group A (Basic Waveform)
    Ablation 2: Groups A + B (+ Timing/HR)
    Ablation 3: Groups A + B + C (+ Pulse Morphology)
    Ablation 4: Groups A + B + C + D (+ VPG)
    Ablation 5: Groups A + B + C + D + E (+ APG)
    Ablation 6: Groups A + B + C + D + E + F (+ Spectral)
    
    Saves to code/outputs/metrics/feature_ablation_results.csv.
    """
    logger.info("Starting Cumulative Feature Group Ablation experiments...")
    out_path = save_path or (METRICS_DIR / "feature_ablation_results.csv")

    cumulative_configs = [
        ("Ablation_0_Dummy", []),
        ("Ablation_1_Basic", FEATURE_GROUPS["A_basic"]),
        ("Ablation_2_Basic_Timing", FEATURE_GROUPS["A_basic"] + FEATURE_GROUPS["B_timing"]),
        (
            "Ablation_3_Basic_Timing_Morph",
            FEATURE_GROUPS["A_basic"] + FEATURE_GROUPS["B_timing"] + FEATURE_GROUPS["C_morphology"],
        ),
        (
            "Ablation_4_Plus_VPG",
            FEATURE_GROUPS["A_basic"]
            + FEATURE_GROUPS["B_timing"]
            + FEATURE_GROUPS["C_morphology"]
            + FEATURE_GROUPS["D_vpg"],
        ),
        (
            "Ablation_5_Plus_APG",
            FEATURE_GROUPS["A_basic"]
            + FEATURE_GROUPS["B_timing"]
            + FEATURE_GROUPS["C_morphology"]
            + FEATURE_GROUPS["D_vpg"]
            + FEATURE_GROUPS["E_apg"],
        ),
        (
            "Ablation_6_All_Features",
            FEATURE_GROUPS["A_basic"]
            + FEATURE_GROUPS["B_timing"]
            + FEATURE_GROUPS["C_morphology"]
            + FEATURE_GROUPS["D_vpg"]
            + FEATURE_GROUPS["E_apg"]
            + FEATURE_GROUPS["F_spectral"],
        ),
    ]

    results = []

    for cfg_name, feat_list in cumulative_configs:
        n_feats = len(feat_list)
        logger.info(f"Running {cfg_name} with {n_feats} features...")

        if cfg_name == "Ablation_0_Dummy":
            # Dummy evaluation
            for tgt in ["sbp", "dbp"]:
                y_tr = df_train[tgt].to_numpy()
                y_v = df_val[tgt].to_numpy()

                dummy_pred = np.full_like(y_v, fill_value=np.mean(y_tr))
                errs = dummy_pred - y_v

                results.append(
                    {
                        "ablation": cfg_name,
                        "feature_count": 0,
                        "target": tgt.upper(),
                        "MAE": float(np.mean(np.abs(errs))),
                        "RMSE": float(np.sqrt(np.mean(errs**2))),
                        "R2": 0.0,
                        "bias": float(np.mean(errs)),
                        "error_sd": float(np.std(errs)),
                    }
                )
            continue

        # Fit leakage-safe preprocessor on train strictly
        preproc = LeakageSafePreprocessor(feat_list)
        X_tr = preproc.fit_transform_train(df_train)
        X_v = preproc.transform(df_val)

        for tgt in ["sbp", "dbp"]:
            y_tr = df_train[tgt].to_numpy()
            y_v = df_val[tgt].to_numpy()

            m, _ = train_model(
                model_name,
                X_tr,
                y_tr,
                target_name=tgt.upper(),
            )
            preds_v = m.predict(X_v)
            errs = preds_v - y_v

            mae = float(np.mean(np.abs(errs)))
            rmse = float(np.sqrt(np.mean(errs**2)))
            r2 = float(1.0 - np.sum(errs**2) / np.sum((y_v - np.mean(y_v)) ** 2))
            bias = float(np.mean(errs))
            error_sd = float(np.std(errs))

            results.append(
                {
                    "ablation": cfg_name,
                    "feature_count": n_feats,
                    "target": tgt.upper(),
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "bias": bias,
                    "error_sd": error_sd,
                }
            )

    df_abl = pd.DataFrame(results)
    df_abl.to_csv(out_path, index=False)
    logger.info(f"Saved feature group ablation results to {out_path}.")
    return df_abl
