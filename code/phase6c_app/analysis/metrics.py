"""
Phase 6C Validation Metrics & Stratified Subgroup Analysis.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Calculates research-standard agreement metrics (MAE, RMSE, Bias, SD, R2, Pearson r,
Spearman rho, error quantiles, AAMI/BHS indicators) strictly on valid matched pairs.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats


def compute_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_name: str = "SBP") -> Dict[str, Any]:
    """
    Compute rigorous statistical validation metrics between reference BP and predicted BP.
    
    Args:
        y_true: 1D array of ground truth reference measurements.
        y_pred: 1D array of model predictions.
        target_name: "SBP" or "DBP".
        
    Returns:
        Dictionary of computed metrics or empty indicators if N < 1.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask]
    yp = y_pred[mask]
    n = len(yt)

    if n == 0:
        return {
            "target": target_name,
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias_mean_error": np.nan,
            "std_error": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "abs_error_quantiles": {},
            "bhs_grades": {},
            "aami_compliant": False,
        }

    errors = yp - yt  # Prediction error: y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    std_err = float(np.std(errors, ddof=1)) if n > 1 else 0.0

    # R^2
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    ss_res = float(np.sum(errors ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else np.nan

    # Pearson & Spearman
    if n >= 2 and np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
        pr_r, pr_p = stats.pearsonr(yt, yp)
        sp_r, sp_p = stats.spearmanr(yt, yp)
        pearson_r, pearson_p = float(pr_r), float(pr_p)
        spearman_rho, spearman_p = float(sp_r), float(sp_p)
    else:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_rho, spearman_p = np.nan, np.nan

    # Quantiles of absolute error
    quantiles = {
        "p25": float(np.percentile(abs_errors, 25)),
        "median_p50": float(np.percentile(abs_errors, 50)),
        "p75": float(np.percentile(abs_errors, 75)),
        "p90": float(np.percentile(abs_errors, 90)),
        "p95": float(np.percentile(abs_errors, 95)),
        "max": float(np.max(abs_errors)),
    }

    # BHS (British Hypertension Society) cumulative error percentages
    pct_leq_5 = float(np.mean(abs_errors <= 5.0) * 100.0)
    pct_leq_10 = float(np.mean(abs_errors <= 10.0) * 100.0)
    pct_leq_15 = float(np.mean(abs_errors <= 15.0) * 100.0)

    # AAMI criteria: Bias <= 5 mmHg and SD <= 8 mmHg
    aami_pass = bool((abs(bias) <= 5.0) and (std_err <= 8.0) and (n >= 2))

    return {
        "target": target_name,
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "bias_mean_error": bias,
        "std_error": std_err,
        "r2": r2,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "abs_error_quantiles": quantiles,
        "bhs_grades": {
            "percent_le_5mmHg": pct_leq_5,
            "percent_le_10mmHg": pct_leq_10,
            "percent_le_15mmHg": pct_leq_15,
        },
        "aami_compliant": aami_pass,
    }


def compute_validation_summary(df_pairs: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive metrics on valid matched pairs, reporting RAW and CALIBRATED separately.
    
    Args:
        df_pairs: Output from bp_pairing module.
        
    Returns:
        Structured dictionary containing SBP/DBP raw & calibrated metrics, plus stratification.
    """
    valid_pairs = df_pairs[df_pairs["included_in_metrics"] == True] if len(df_pairs) > 0 else pd.DataFrame()
    n_valid = len(valid_pairs)

    if n_valid == 0:
        return {
            "n_valid_matched_pairs": 0,
            "raw": {
                "sbp": compute_target_metrics(np.array([]), np.array([]), "SBP"),
                "dbp": compute_target_metrics(np.array([]), np.array([]), "DBP"),
            },
            "calibrated": {
                "sbp": compute_target_metrics(np.array([]), np.array([]), "SBP"),
                "dbp": compute_target_metrics(np.array([]), np.array([]), "DBP"),
            },
            "stratified": {},
        }

    y_ref_sbp = valid_pairs["reference_sbp"].to_numpy(dtype=float)
    y_pred_sbp_raw = valid_pairs["predicted_sbp_raw"].to_numpy(dtype=float)
    y_pred_sbp_cal = valid_pairs["predicted_sbp_calibrated"].to_numpy(dtype=float)

    y_ref_dbp = valid_pairs["reference_dbp"].to_numpy(dtype=float)
    y_pred_dbp_raw = valid_pairs["predicted_dbp_raw"].to_numpy(dtype=float)
    y_pred_dbp_cal = valid_pairs["predicted_dbp_calibrated"].to_numpy(dtype=float)

    sbp_raw_metrics = compute_target_metrics(y_ref_sbp, y_pred_sbp_raw, "SBP_RAW")
    sbp_cal_metrics = compute_target_metrics(y_ref_sbp, y_pred_sbp_cal, "SBP_CALIBRATED")

    dbp_raw_metrics = compute_target_metrics(y_ref_dbp, y_pred_dbp_raw, "DBP_RAW")
    dbp_cal_metrics = compute_target_metrics(y_ref_dbp, y_pred_dbp_cal, "DBP_CALIBRATED")

    # Stratified Subgroup Analysis (Only when subgroup has n >= 2)
    stratified_results = {}

    # SBP Subgroups: <120 (Normotensive), 120-139 (Elevated/Pre), >=140 (Hypertensive)
    sbp_bins = [
        ("<120 mmHg (Normotensive)", y_ref_sbp < 120.0),
        ("120-139 mmHg (Elevated/Stage 1)", (y_ref_sbp >= 120.0) & (y_ref_sbp < 140.0)),
        (">=140 mmHg (Stage 2/Hypertensive)", y_ref_sbp >= 140.0),
    ]
    strat_sbp = {}
    for label, mask in sbp_bins:
        n_sub = int(np.sum(mask))
        if n_sub >= 2:
            strat_sbp[label] = {
                "n": n_sub,
                "raw_mae": float(np.mean(np.abs(y_pred_sbp_raw[mask] - y_ref_sbp[mask]))),
                "cal_mae": float(np.mean(np.abs(y_pred_sbp_cal[mask] - y_ref_sbp[mask]))),
                "cal_bias": float(np.mean(y_pred_sbp_cal[mask] - y_ref_sbp[mask])),
            }
        else:
            strat_sbp[label] = {"n": n_sub, "status": "Insufficient samples (N < 2)"}
    stratified_results["sbp_subgroups"] = strat_sbp

    # DBP Subgroups: <60, 60-79, >=80
    dbp_bins = [
        ("<60 mmHg (Low)", y_ref_dbp < 60.0),
        ("60-79 mmHg (Normal)", (y_ref_dbp >= 60.0) & (y_ref_dbp < 80.0)),
        (">=80 mmHg (Elevated/Stage 1+)", y_ref_dbp >= 80.0),
    ]
    strat_dbp = {}
    for label, mask in dbp_bins:
        n_sub = int(np.sum(mask))
        if n_sub >= 2:
            strat_dbp[label] = {
                "n": n_sub,
                "raw_mae": float(np.mean(np.abs(y_pred_dbp_raw[mask] - y_ref_dbp[mask]))),
                "cal_mae": float(np.mean(np.abs(y_pred_dbp_cal[mask] - y_ref_dbp[mask]))),
                "cal_bias": float(np.mean(y_pred_dbp_cal[mask] - y_ref_dbp[mask])),
            }
        else:
            strat_dbp[label] = {"n": n_sub, "status": "Insufficient samples (N < 2)"}
    stratified_results["dbp_subgroups"] = strat_dbp

    return {
        "n_valid_matched_pairs": n_valid,
        "raw": {
            "sbp": sbp_raw_metrics,
            "dbp": dbp_raw_metrics,
        },
        "calibrated": {
            "sbp": sbp_cal_metrics,
            "dbp": dbp_cal_metrics,
        },
        "stratified": stratified_results,
    }
