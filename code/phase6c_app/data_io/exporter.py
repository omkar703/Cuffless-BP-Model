"""
Phase 6C Results Exporter & Package Bundler.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Persists all validation artifacts, CSV tables, JSON records, figures,
and markdown reports, and builds a comprehensive downloadable ZIP package.
"""

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd

from phase6c_app.visualization.plots import save_all_figures
from phase6c_app.reports.report_generator import generate_phase6c_reports


def export_full_results(
    output_dir: Path,
    session_data: Any,
    audit_results: Dict[str, Any],
    df_windows: pd.DataFrame,
    df_predictions: pd.DataFrame,
    streaming_traces: Dict[str, Any],
    replay_summary: Dict[str, Any],
    df_pairs: pd.DataFrame,
    pairing_audit: Dict[str, Any],
    validation_metrics: Dict[str, Any]
) -> Tuple[Dict[str, Path], bytes]:
    """
    Save all tables, figures, metadata, and reports to disk, and return in-memory zip bytes.
    
    Args:
        output_dir: Target directory path for file exports.
        ...
        
    Returns:
        Tuple of (dictionary of created file paths, zip_bytes)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    is_demo = getattr(session_data, "is_demo", False)

    # 1. Hardware Audit JSON
    audit_path = output_dir / "session_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    saved_files["session_audit_json"] = audit_path

    # 2. Window Quality CSV
    win_path = output_dir / "window_quality.csv"
    if len(df_windows) > 0:
        df_windows.to_csv(win_path, index=False)
    else:
        pd.DataFrame().to_csv(win_path, index=False)
    saved_files["window_quality_csv"] = win_path

    # 3. Model Predictions CSV
    pred_path = output_dir / "predictions.csv"
    if len(df_predictions) > 0:
        df_predictions.to_csv(pred_path, index=False)
    else:
        pd.DataFrame().to_csv(pred_path, index=False)
    saved_files["predictions_csv"] = pred_path

    # 4. Reference BP Events CSV
    events_path = output_dir / "reference_bp_events.csv"
    if len(session_data.df_bp_events) > 0:
        session_data.df_bp_events.to_csv(events_path, index=False)
    else:
        pd.DataFrame().to_csv(events_path, index=False)
    saved_files["reference_bp_events_csv"] = events_path

    # 5. Prediction-Reference Pairs CSV
    pairs_path = output_dir / "phase6c_prediction_reference_pairs.csv"
    if len(df_pairs) > 0:
        df_pairs.to_csv(pairs_path, index=False)
    else:
        pd.DataFrame().to_csv(pairs_path, index=False)
    saved_files["prediction_reference_pairs_csv"] = pairs_path

    # 6. Figures
    fig_paths = save_all_figures(
        fig_dir=fig_dir,
        df_ppg=session_data.df_ppg_replay,
        df_bp_events=session_data.df_bp_events,
        df_predictions=df_predictions,
        df_pairs=df_pairs,
        traces=streaming_traces,
        is_demo=is_demo,
    )
    for k, v in fig_paths.items():
        saved_files[f"figure_{k}"] = v

    # 7. Reports
    md_path, json_path, meta_path = generate_phase6c_reports(
        output_dir=output_dir,
        session_data=session_data,
        audit_results=audit_results,
        replay_summary=replay_summary,
        df_windows=df_windows,
        df_predictions=df_predictions,
        df_pairs=df_pairs,
        pairing_audit=pairing_audit,
        validation_metrics=validation_metrics,
        figure_paths=fig_paths,
    )
    saved_files["validation_report_md"] = md_path
    saved_files["validation_report_json"] = json_path
    saved_files["run_metadata_json"] = meta_path

    # 8. Create ZIP archive in memory
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f_key, f_path in saved_files.items():
            if f_path.is_file():
                rel_path = f_path.relative_to(output_dir)
                z.write(f_path, arcname=str(rel_path))

    zip_bytes = zip_buf.getvalue()
    zip_export_path = output_dir / f"phase6c_{session_data.session_id}_validation_export.zip"
    with open(zip_export_path, "wb") as f:
        f.write(zip_bytes)
    saved_files["zip_package"] = zip_export_path

    return saved_files, zip_bytes
