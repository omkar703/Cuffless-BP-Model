"""
Deterministic Reference BP Synchronization & Quality Gating Module.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Implements the predefined deterministic pairing rule matching REFERENCE_BP_RESULT
events to the nearest causal Phase 6B temporal prediction within a fixed time window.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def pair_predictions_with_reference(
    df_predictions: pd.DataFrame,
    df_bp_events: pd.DataFrame,
    session_id: str = "session_01",
    tolerance_s: float = 15.0,
    rule_name: str = "NEAREST_PREDICTION_TIMESTAMP",
    exclude_warn_windows: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Deterministically pair reference BP events with frozen Phase 6B predictions.
    
    Args:
        df_predictions: Output DataFrame from ReplayEngine.
        df_bp_events: DataFrame of standardized BP events.
        session_id: Session identifier string.
        tolerance_s: Configurable maximum tolerance in seconds (default +/- 15s).
        rule_name: Primary pairing rule identifier.
        exclude_warn_windows: If True, exclude pairs with quality_status=='WARN'.
        
    Returns:
        Tuple of (df_pairs, pairing_audit_summary)
    """
    pairs_list = []
    audit_summary = {
        "rule_name": rule_name,
        "tolerance_seconds": tolerance_s,
        "exclude_warn_windows": exclude_warn_windows,
        "total_bp_events_recorded": len(df_bp_events),
        "total_reference_results": 0,
        "matched_pairs_count": 0,
        "included_pairs_count": 0,
        "unmatched_count": 0,
        "excluded_count": 0,
    }

    if df_bp_events is None or len(df_bp_events) == 0:
        return pd.DataFrame(columns=[
            "session_id", "prediction_id", "prediction_timestamp", "context_start", "context_end",
            "reference_event_id", "reference_timestamp", "pairing_delta_s",
            "reference_sbp", "predicted_sbp_raw", "predicted_sbp_calibrated",
            "sbp_error_raw", "sbp_error_calibrated",
            "reference_dbp", "predicted_dbp_raw", "predicted_dbp_calibrated",
            "dbp_error_raw", "dbp_error_calibrated",
            "quality_status", "included_in_metrics", "exclusion_reason"
        ]), audit_summary

    # Filter to REFERENCE_BP_RESULT events (or events that have valid numeric SBP/DBP)
    ref_mask = (df_bp_events["event_type"] == "REFERENCE_BP_RESULT") | (
        df_bp_events["sbp"].notna() & df_bp_events["dbp"].notna()
    )
    df_ref = df_bp_events[ref_mask].copy()
    audit_summary["total_reference_results"] = len(df_ref)

    if len(df_ref) == 0 or len(df_predictions) == 0:
        return pd.DataFrame(columns=[
            "session_id", "prediction_id", "prediction_timestamp", "context_start", "context_end",
            "reference_event_id", "reference_timestamp", "pairing_delta_s",
            "reference_sbp", "predicted_sbp_raw", "predicted_sbp_calibrated",
            "sbp_error_raw", "sbp_error_calibrated",
            "reference_dbp", "predicted_dbp_raw", "predicted_dbp_calibrated",
            "dbp_error_raw", "dbp_error_calibrated",
            "quality_status", "included_in_metrics", "exclusion_reason"
        ]), audit_summary

    pred_ts_ms = df_predictions["prediction_timestamp_ms"].to_numpy(dtype=float)
    seen_event_ids = set()

    for _, ref_row in df_ref.iterrows():
        evt_id = str(ref_row["event_id"])
        ref_ts_ms = float(ref_row["host_timestamp_ms"]) if pd.notna(ref_row["host_timestamp_ms"]) else np.nan
        ref_sbp = float(ref_row["sbp"]) if pd.notna(ref_row["sbp"]) else np.nan
        ref_dbp = float(ref_row["dbp"]) if pd.notna(ref_row["dbp"]) else np.nan

        # Exclusion checks on reference measurement itself
        if np.isnan(ref_sbp) or np.isnan(ref_dbp):
            pairs_list.append({
                "session_id": session_id,
                "prediction_id": "NONE",
                "prediction_timestamp": np.nan,
                "context_start": np.nan,
                "context_end": np.nan,
                "reference_event_id": evt_id,
                "reference_timestamp": ref_ts_ms,
                "pairing_delta_s": np.nan,
                "reference_sbp": ref_sbp,
                "predicted_sbp_raw": np.nan,
                "predicted_sbp_calibrated": np.nan,
                "sbp_error_raw": np.nan,
                "sbp_error_calibrated": np.nan,
                "reference_dbp": ref_dbp,
                "predicted_dbp_raw": np.nan,
                "predicted_dbp_calibrated": np.nan,
                "dbp_error_raw": np.nan,
                "dbp_error_calibrated": np.nan,
                "quality_status": "EXCLUDED",
                "included_in_metrics": False,
                "exclusion_reason": "MISSING_REFERENCE_BP_VALUES",
            })
            audit_summary["excluded_count"] += 1
            continue

        if evt_id in seen_event_ids:
            pairs_list.append({
                "session_id": session_id,
                "prediction_id": "NONE",
                "prediction_timestamp": np.nan,
                "context_start": np.nan,
                "context_end": np.nan,
                "reference_event_id": evt_id,
                "reference_timestamp": ref_ts_ms,
                "pairing_delta_s": np.nan,
                "reference_sbp": ref_sbp,
                "predicted_sbp_raw": np.nan,
                "predicted_sbp_calibrated": np.nan,
                "sbp_error_raw": np.nan,
                "sbp_error_calibrated": np.nan,
                "reference_dbp": ref_dbp,
                "predicted_dbp_raw": np.nan,
                "predicted_dbp_calibrated": np.nan,
                "dbp_error_raw": np.nan,
                "dbp_error_calibrated": np.nan,
                "quality_status": "EXCLUDED",
                "included_in_metrics": False,
                "exclusion_reason": "DUPLICATE_EVENT_ID",
            })
            audit_summary["excluded_count"] += 1
            continue
        seen_event_ids.add(evt_id)

        if np.isnan(ref_ts_ms):
            pairs_list.append({
                "session_id": session_id,
                "prediction_id": "NONE",
                "prediction_timestamp": np.nan,
                "context_start": np.nan,
                "context_end": np.nan,
                "reference_event_id": evt_id,
                "reference_timestamp": np.nan,
                "pairing_delta_s": np.nan,
                "reference_sbp": ref_sbp,
                "predicted_sbp_raw": np.nan,
                "predicted_sbp_calibrated": np.nan,
                "sbp_error_raw": np.nan,
                "sbp_error_calibrated": np.nan,
                "reference_dbp": ref_dbp,
                "predicted_dbp_raw": np.nan,
                "predicted_dbp_calibrated": np.nan,
                "dbp_error_raw": np.nan,
                "dbp_error_calibrated": np.nan,
                "quality_status": "EXCLUDED",
                "included_in_metrics": False,
                "exclusion_reason": "MISSING_EVENT_TIMESTAMP",
            })
            audit_summary["excluded_count"] += 1
            continue

        # Find closest prediction
        deltas_sec = (pred_ts_ms - ref_ts_ms) / 1000.0
        abs_deltas = np.abs(deltas_sec)
        best_idx = int(np.argmin(abs_deltas))
        min_delta_s = float(deltas_sec[best_idx])
        min_abs_delta_s = float(abs_deltas[best_idx])

        best_pred = df_predictions.iloc[best_idx]
        pred_id = str(best_pred["prediction_id"])
        pred_ts = float(best_pred["prediction_timestamp_ms"])
        ctx_start = float(best_pred["context_start_s"])
        ctx_end = float(best_pred["context_end_s"])
        q_status = str(best_pred["quality_status"])

        pred_sbp_raw = float(best_pred["raw_sbp"])
        pred_dbp_raw = float(best_pred["raw_dbp"])
        pred_sbp_cal = float(best_pred["calibrated_sbp"])
        pred_dbp_cal = float(best_pred["calibrated_dbp"])

        # Check tolerance
        if min_abs_delta_s > tolerance_s:
            pairs_list.append({
                "session_id": session_id,
                "prediction_id": pred_id,
                "prediction_timestamp": pred_ts,
                "context_start": ctx_start,
                "context_end": ctx_end,
                "reference_event_id": evt_id,
                "reference_timestamp": ref_ts_ms,
                "pairing_delta_s": min_delta_s,
                "reference_sbp": ref_sbp,
                "predicted_sbp_raw": pred_sbp_raw,
                "predicted_sbp_calibrated": pred_sbp_cal,
                "sbp_error_raw": np.nan,
                "sbp_error_calibrated": np.nan,
                "reference_dbp": ref_dbp,
                "predicted_dbp_raw": pred_dbp_raw,
                "predicted_dbp_calibrated": pred_dbp_cal,
                "dbp_error_raw": np.nan,
                "dbp_error_calibrated": np.nan,
                "quality_status": "UNMATCHED",
                "included_in_metrics": False,
                "exclusion_reason": f"EXCESSIVE_TIMING_DIFF ({abs(min_delta_s):.1f}s > {tolerance_s:.1f}s)",
            })
            audit_summary["unmatched_count"] += 1
            continue

        audit_summary["matched_pairs_count"] += 1

        # Check signal quality
        if q_status == "REJECTED_SIGNAL_QUALITY" or np.isnan(pred_sbp_raw):
            included = False
            ex_reason = "POOR_PPG_QUALITY_REJECTED"
        elif exclude_warn_windows and q_status == "WARN":
            included = False
            ex_reason = "PPG_QUALITY_WARN_FLAGGED"
        else:
            included = True
            ex_reason = "NONE"

        sbp_err_raw = (pred_sbp_raw - ref_sbp) if (included and not np.isnan(pred_sbp_raw)) else np.nan
        sbp_err_cal = (pred_sbp_cal - ref_sbp) if (included and not np.isnan(pred_sbp_cal)) else np.nan
        dbp_err_raw = (pred_dbp_raw - ref_dbp) if (included and not np.isnan(pred_dbp_raw)) else np.nan
        dbp_err_cal = (pred_dbp_cal - ref_dbp) if (included and not np.isnan(pred_dbp_cal)) else np.nan

        if included:
            audit_summary["included_pairs_count"] += 1
        else:
            audit_summary["excluded_count"] += 1

        pairs_list.append({
            "session_id": session_id,
            "prediction_id": pred_id,
            "prediction_timestamp": pred_ts,
            "context_start": ctx_start,
            "context_end": ctx_end,
            "reference_event_id": evt_id,
            "reference_timestamp": ref_ts_ms,
            "pairing_delta_s": min_delta_s,
            "reference_sbp": ref_sbp,
            "predicted_sbp_raw": pred_sbp_raw,
            "predicted_sbp_calibrated": pred_sbp_cal,
            "sbp_error_raw": sbp_err_raw,
            "sbp_error_calibrated": sbp_err_cal,
            "reference_dbp": ref_dbp,
            "predicted_dbp_raw": pred_dbp_raw,
            "predicted_dbp_calibrated": pred_dbp_cal,
            "dbp_error_raw": dbp_err_raw,
            "dbp_error_calibrated": dbp_err_cal,
            "quality_status": q_status,
            "included_in_metrics": included,
            "exclusion_reason": ex_reason,
        })

    df_pairs = pd.DataFrame(pairs_list)
    return df_pairs, audit_summary
