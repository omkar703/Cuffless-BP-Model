"""
Evaluation Engine for Phase 3A Baseline Modeling.

Implements:
1. Primary & supplementary regression metrics:
   - MAE, RMSE, R2, Bias, Error SD, Pearson r, Spearman rho, MAPE.
2. Clinical descriptive accuracy bands:
   - % <= 5 mmHg, % <= 10 mmHg, % <= 15 mmHg.
3. Dummy improvement metrics:
   - delta_MAE_vs_dummy, percentage_MAE_improvement_vs_dummy, delta_RMSE_vs_dummy, delta_R2_vs_dummy.
4. Record-weighted vs Window-weighted evaluation:
   - Computes per-record MAE and reports mean, median, std, and IQR across records.
5. Record-level clustered correlation analysis:
   - Pearson r, Spearman rho, and record-clustered bootstrap 95% confidence intervals.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.logging_utils import setup_logger

logger = setup_logger("evaluation_engine")


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str = "BP",
) -> Dict[str, float]:
    """Computes standard window-weighted regression metrics and clinical error bands."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    r2 = float(r2_score(y_true, y_pred)) if np.var(y_true) > 1e-9 else 0.0
    bias = float(np.mean(errors))
    error_sd = float(np.std(errors))

    # Pearson & Spearman correlations
    if np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9:
        pr, _ = pearsonr(y_true, y_pred)
        sr, _ = spearmanr(y_true, y_pred)
        pearson_r = float(pr)
        spearman_r = float(sr)
    else:
        pearson_r = 0.0
        spearman_r = 0.0

    mape = float(np.mean(abs_errors / (y_true + 1e-6)) * 100.0)

    # Clinical error bands (<= 5, 10, 15 mmHg)
    pct_le_5 = float(np.mean(abs_errors <= 5.0) * 100.0)
    pct_le_10 = float(np.mean(abs_errors <= 10.0) * 100.0)
    pct_le_15 = float(np.mean(abs_errors <= 15.0) * 100.0)

    return {
        "target": target_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "error_sd": error_sd,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "mape": mape,
        "pct_le_5": pct_le_5,
        "pct_le_10": pct_le_10,
        "pct_le_15": pct_le_15,
        "sample_count": len(y_true),
    }


def compute_dummy_improvements(
    model_metrics: Dict[str, float],
    dummy_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Computes learning value relative to the Dummy population-mean baseline."""
    dummy_mae = dummy_metrics["mae"]
    dummy_rmse = dummy_metrics["rmse"]
    dummy_r2 = dummy_metrics["r2"]

    model_mae = model_metrics["mae"]
    model_rmse = model_metrics["rmse"]
    model_r2 = model_metrics["r2"]

    delta_mae = float(dummy_mae - model_mae)
    pct_mae_imp = float(100.0 * delta_mae / (dummy_mae + 1e-9))
    delta_rmse = float(dummy_rmse - model_rmse)
    delta_r2 = float(model_r2 - dummy_r2)

    return {
        "delta_MAE_vs_dummy": delta_mae,
        "percentage_MAE_improvement_vs_dummy": pct_mae_imp,
        "delta_RMSE_vs_dummy": delta_rmse,
        "delta_R2_vs_dummy": delta_r2,
        "dummy_mae": dummy_mae,
        "dummy_rmse": dummy_rmse,
        "dummy_r2": dummy_r2,
    }


def compute_record_weighted_metrics(
    df_eval: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    target_name: str = "BP",
) -> Dict[str, Any]:
    """
    Computes per-record MAE and aggregates across unique records:
    mean, median, std, and IQR of record-level MAE.
    
    Returns:
        summary_dict, df_record_metrics
    """
    df = df_eval.copy()
    df["error"] = df[y_pred_col] - df[y_true_col]
    df["abs_error"] = np.abs(df["error"])

    # Aggregate per record
    rec_grp = df.groupby("record_id")
    rec_df = rec_grp.agg(
        record_mae=("abs_error", "mean"),
        record_rmse=("error", lambda s: np.sqrt(np.mean(s**2))),
        record_bias=("error", "mean"),
        n_windows=("abs_error", "count"),
    ).reset_index()

    rec_maes = rec_df["record_mae"].to_numpy()
    q25, q50, q75 = np.percentile(rec_maes, [25, 50, 75])

    summary = {
        "target": target_name,
        "n_records": len(rec_df),
        "mean_record_mae": float(np.mean(rec_maes)),
        "median_record_mae": float(q50),
        "std_record_mae": float(np.std(rec_maes)),
        "iqr_record_mae": float(q75 - q25),
        "min_record_mae": float(np.min(rec_maes)),
        "max_record_mae": float(np.max(rec_maes)),
    }

    return summary, rec_df


def compute_clustered_correlation(
    df_eval: pd.DataFrame,
    feature_col: str,
    error_col: str = "abs_error",
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes Pearson and Spearman correlation between prediction error and a feature,
    with record-level clustered bootstrap to construct robust 95% confidence intervals.
    """
    sub_df = df_eval[["record_id", feature_col, error_col]].dropna().copy()
    if len(sub_df) < 20:
        return {
            "feature": feature_col,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "pearson_ci_lower": np.nan,
            "pearson_ci_upper": np.nan,
            "spearman_ci_lower": np.nan,
            "spearman_ci_upper": np.nan,
            "sample_size": len(sub_df),
            "record_count": sub_df["record_id"].nunique(),
        }

    x = sub_df[feature_col].to_numpy()
    y = sub_df[error_col].to_numpy()

    p_r, _ = pearsonr(x, y) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else (0.0, 1.0)
    s_r, _ = spearmanr(x, y) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else (0.0, 1.0)

    # Clustered Bootstrap by record_id
    rng = np.random.RandomState(random_state)
    unique_recs = sub_df["record_id"].unique()
    n_recs = len(unique_recs)

    rec_to_indices = sub_df.groupby("record_id").indices

    boot_p_rs = []
    boot_s_rs = []

    for _ in range(n_bootstraps):
        sampled_recs = rng.choice(unique_recs, size=n_recs, replace=True)
        sampled_idx_lists = [rec_to_indices[r] for r in sampled_recs]
        boot_indices = np.concatenate(sampled_idx_lists)

        bx = x[boot_indices]
        by = y[boot_indices]

        if np.std(bx) > 1e-9 and np.std(by) > 1e-9:
            br, _ = pearsonr(bx, by)
            bs, _ = spearmanr(bx, by)
            boot_p_rs.append(br)
            boot_s_rs.append(bs)

    if len(boot_p_rs) >= 100:
        p_ci_low, p_ci_high = np.percentile(boot_p_rs, [2.5, 97.5])
        s_ci_low, s_ci_high = np.percentile(boot_s_rs, [2.5, 97.5])
    else:
        p_ci_low, p_ci_high = np.nan, np.nan
        s_ci_low, s_ci_high = np.nan, np.nan

    return {
        "feature": feature_col,
        "pearson_r": float(p_r),
        "spearman_r": float(s_r),
        "pearson_ci_lower": float(p_ci_low),
        "pearson_ci_upper": float(p_ci_high),
        "spearman_ci_lower": float(s_ci_low),
        "spearman_ci_upper": float(s_ci_high),
        "sample_size": len(sub_df),
        "record_count": n_recs,
    }
