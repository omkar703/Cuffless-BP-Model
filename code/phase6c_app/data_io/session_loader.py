"""
Session Loader & Compatibility Adapter for Phase 6C.
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Loads session packages (ZIP, files, directory), validates schemas,
preserves raw data untouched, and produces standardized replay DataFrames.
"""

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class SessionData:
    session_id: str
    source_type: str  # "ZIP", "FILES", "LOCAL_DIR", "DEMO"
    source_name: str
    df_ppg_raw: pd.DataFrame
    df_ppg_replay: pd.DataFrame
    df_bp_events: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    readme_text: str = ""
    is_demo: bool = False
    load_warnings: list = field(default_factory=list)


def _standardize_ppg_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Produce a standardized derived replay DataFrame without modifying df_raw.
    
    Standardized columns:
    - sample_index
    - host_timestamp_ms
    - expected_timestamp_ms
    - elapsed_s
    - ir
    - red
    """
    warnings = []
    df_replay = df_raw.copy()

    # Column mapping & checks
    cols = list(df_replay.columns)
    cols_lower = {c.lower(): c for c in cols}

    # Identify IR channel
    if "ir" in cols_lower:
        ir_col = cols_lower["ir"]
    elif "ir_ppg" in cols_lower:
        ir_col = cols_lower["ir_ppg"]
    else:
        raise ValueError(f"PPG data must contain an 'ir' channel. Found columns: {cols}")

    # Identify Red channel
    if "red" in cols_lower:
        red_col = cols_lower["red"]
    elif "red_ppg" in cols_lower:
        red_col = cols_lower["red_ppg"]
    else:
        red_col = None
        warnings.append("Red channel missing; defaulting red counts to 0.0")

    # Identify sample index
    if "sample_index" in cols_lower:
        idx_col = cols_lower["sample_index"]
    elif "index" in cols_lower:
        idx_col = cols_lower["index"]
    else:
        warnings.append("sample_index column missing; generating contiguous 0-indexed sequence")
        df_replay["sample_index"] = np.arange(len(df_replay)) * 10
        idx_col = "sample_index"

    # Identify host timestamp
    if "host_timestamp_ms" in cols_lower:
        host_ts_col = cols_lower["host_timestamp_ms"]
    elif "timestamp_ms" in cols_lower:
        host_ts_col = cols_lower["timestamp_ms"]
        warnings.append("Mapped 'timestamp_ms' -> 'host_timestamp_ms'")
    elif "timestamp" in cols_lower:
        host_ts_col = cols_lower["timestamp"]
        warnings.append("Mapped 'timestamp' -> 'host_timestamp_ms'")
    else:
        warnings.append("Timestamp missing; synthesizing host timestamps assuming 100 Hz (10 ms steps)")
        df_replay["host_timestamp_ms"] = np.arange(len(df_replay)) * 10.0
        host_ts_col = "host_timestamp_ms"

    # Build standardized derived DataFrame
    s_idx = pd.to_numeric(df_replay[idx_col], errors="coerce").fillna(0).to_numpy()
    first_idx = s_idx[0] if len(s_idx) > 0 else 0
    exp_ts_ms = s_idx - first_idx

    host_ts = pd.to_numeric(df_replay[host_ts_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    t0_host = host_ts[0] if len(host_ts) > 0 else 0.0
    elapsed_s = (host_ts - t0_host) / 1000.0

    ir_data = pd.to_numeric(df_replay[ir_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if red_col is not None:
        red_data = pd.to_numeric(df_replay[red_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        red_data = np.zeros_like(ir_data)

    df_clean = pd.DataFrame({
        "sample_index": s_idx,
        "host_timestamp_ms": host_ts,
        "expected_timestamp_ms": exp_ts_ms,
        "elapsed_s": elapsed_s,
        "ir": ir_data,
        "red": red_data,
    })

    return df_clean, warnings


def _standardize_bp_events(df_raw: Optional[pd.DataFrame], t0_host_ms: float = 0.0) -> Tuple[pd.DataFrame, list]:
    """
    Standardize BP events DataFrame to guaranteed schema:
    event_id, host_timestamp_ms, elapsed_s, event_type, label, sbp, dbp, notes
    """
    warnings = []
    if df_raw is None or len(df_raw) == 0:
        empty_df = pd.DataFrame(columns=[
            "event_id", "host_timestamp_ms", "elapsed_s", "event_type", "label", "sbp", "dbp", "notes"
        ])
        return empty_df, warnings

    df_events = df_raw.copy()
    cols_lower = {c.lower(): c for c in df_events.columns}

    # event_id
    if "event_id" in cols_lower:
        event_id = df_events[cols_lower["event_id"]].astype(str)
    else:
        event_id = [f"evt_{i:03d}" for i in range(len(df_events))]

    # host_timestamp_ms
    if "host_timestamp_ms" in cols_lower:
        host_ts = pd.to_numeric(df_events[cols_lower["host_timestamp_ms"]], errors="coerce")
    elif "timestamp_ms" in cols_lower:
        host_ts = pd.to_numeric(df_events[cols_lower["timestamp_ms"]], errors="coerce")
    elif "timestamp" in cols_lower:
        host_ts = pd.to_numeric(df_events[cols_lower["timestamp"]], errors="coerce")
    else:
        host_ts = pd.Series([np.nan] * len(df_events))

    # elapsed_s
    if "elapsed_s" in cols_lower:
        elapsed = pd.to_numeric(df_events[cols_lower["elapsed_s"]], errors="coerce")
    elif not host_ts.isna().all() and t0_host_ms > 0:
        elapsed = (host_ts - t0_host_ms) / 1000.0
    else:
        elapsed = pd.Series([np.nan] * len(df_events))

    # event_type
    if "event_type" in cols_lower:
        event_type = df_events[cols_lower["event_type"]].astype(str).str.upper()
    elif "type" in cols_lower:
        event_type = df_events[cols_lower["type"]].astype(str).str.upper()
    else:
        event_type = pd.Series(["REFERENCE_BP_RESULT"] * len(df_events))

    # label
    if "label" in cols_lower:
        label = df_events[cols_lower["label"]].astype(str)
    else:
        label = pd.Series(["Reference BP"] * len(df_events))

    # sbp & dbp
    if "sbp" in cols_lower:
        sbp = pd.to_numeric(df_events[cols_lower["sbp"]], errors="coerce")
    elif "sys" in cols_lower:
        sbp = pd.to_numeric(df_events[cols_lower["sys"]], errors="coerce")
    else:
        sbp = pd.Series([np.nan] * len(df_events))

    if "dbp" in cols_lower:
        dbp = pd.to_numeric(df_events[cols_lower["dbp"]], errors="coerce")
    elif "dia" in cols_lower:
        dbp = pd.to_numeric(df_events[cols_lower["dia"]], errors="coerce")
    else:
        dbp = pd.Series([np.nan] * len(df_events))

    # notes
    if "notes" in cols_lower:
        notes = df_events[cols_lower["notes"]].astype(str)
    elif "note" in cols_lower:
        notes = df_events[cols_lower["note"]].astype(str)
    else:
        notes = pd.Series([""] * len(df_events))

    std_df = pd.DataFrame({
        "event_id": event_id,
        "host_timestamp_ms": host_ts,
        "elapsed_s": elapsed,
        "event_type": event_type,
        "label": label,
        "sbp": sbp,
        "dbp": dbp,
        "notes": notes,
    })

    return std_df, warnings


def load_session_from_zip(zip_source: Union[str, Path, io.BytesIO]) -> SessionData:
    """Load a Phase 6C session package from a ZIP file."""
    all_warnings = []
    with zipfile.ZipFile(zip_source, "r") as z:
        namelist = z.namelist()
        
        # Locate ppg_samples.csv
        ppg_cand = [n for n in namelist if n.endswith("ppg_samples.csv") or "ppg" in n.lower() and n.endswith(".csv")]
        if not ppg_cand:
            csv_cand = [n for n in namelist if n.endswith(".csv") and not "bp" in n.lower()]
            if csv_cand:
                ppg_cand = csv_cand
            else:
                raise FileNotFoundError("Could not find ppg_samples.csv inside the ZIP archive.")
        ppg_filename = ppg_cand[0]
        with z.open(ppg_filename) as f:
            df_ppg_raw = pd.read_csv(f)

        # Standardize PPG
        df_ppg_replay, ppg_warns = _standardize_ppg_dataframe(df_ppg_raw)
        all_warnings.extend(ppg_warns)

        t0_host = float(df_ppg_replay["host_timestamp_ms"].iloc[0]) if len(df_ppg_replay) > 0 else 0.0

        # Locate bp_events.csv
        bp_cand = [n for n in namelist if "bp_events.csv" in n or ("bp" in n.lower() and "event" in n.lower() and n.endswith(".csv"))]
        if bp_cand:
            with z.open(bp_cand[0]) as f:
                df_bp_raw = pd.read_csv(f)
        else:
            df_bp_raw = None
            all_warnings.append("bp_events.csv not found in ZIP archive.")

        df_bp_events, bp_warns = _standardize_bp_events(df_bp_raw, t0_host_ms=t0_host)
        all_warnings.extend(bp_warns)

        # Locate session_metadata.json
        meta_cand = [n for n in namelist if n.endswith(".json")]
        metadata = {}
        if meta_cand:
            try:
                with z.open(meta_cand[0]) as f:
                    metadata = json.load(f)
            except Exception as e:
                all_warnings.append(f"Failed to parse metadata JSON: {e}")

        # Locate README.txt
        readme_cand = [n for n in namelist if "readme" in n.lower()]
        readme_text = ""
        if readme_cand:
            try:
                with z.open(readme_cand[0]) as f:
                    readme_text = f.read().decode("utf-8", errors="replace")
            except Exception:
                pass

    session_id = metadata.get("session_id", Path(getattr(zip_source, "name", "session_zip")).stem)
    is_demo = bool(metadata.get("IS_DEMO", False))

    return SessionData(
        session_id=session_id,
        source_type="ZIP",
        source_name=Path(getattr(zip_source, "name", "uploaded.zip")).name,
        df_ppg_raw=df_ppg_raw,
        df_ppg_replay=df_ppg_replay,
        df_bp_events=df_bp_events,
        metadata=metadata,
        readme_text=readme_text,
        is_demo=is_demo,
        load_warnings=all_warnings,
    )


def load_session_from_files(
    ppg_file: Any,
    bp_file: Optional[Any] = None,
    meta_file: Optional[Any] = None,
    session_id_override: Optional[str] = None
) -> SessionData:
    """Load session from individual files or file-like objects (e.g. Streamlit upload)."""
    all_warnings = []
    
    # PPG
    if isinstance(ppg_file, (str, Path)):
        df_ppg_raw = pd.read_csv(ppg_file)
        source_name = Path(ppg_file).name
    else:
        df_ppg_raw = pd.read_csv(ppg_file)
        source_name = getattr(ppg_file, "name", "ppg_samples.csv")

    df_ppg_replay, ppg_warns = _standardize_ppg_dataframe(df_ppg_raw)
    all_warnings.extend(ppg_warns)

    t0_host = float(df_ppg_replay["host_timestamp_ms"].iloc[0]) if len(df_ppg_replay) > 0 else 0.0

    # BP Events
    if bp_file is not None:
        try:
            df_bp_raw = pd.read_csv(bp_file)
        except Exception as e:
            df_bp_raw = None
            all_warnings.append(f"Failed to read BP events file: {e}")
    else:
        df_bp_raw = None

    df_bp_events, bp_warns = _standardize_bp_events(df_bp_raw, t0_host_ms=t0_host)
    all_warnings.extend(bp_warns)

    # Metadata
    metadata = {}
    if meta_file is not None:
        try:
            if isinstance(meta_file, (str, Path)):
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = json.load(meta_file)
        except Exception as e:
            all_warnings.append(f"Failed to read metadata JSON: {e}")

    session_id = session_id_override or metadata.get("session_id", Path(source_name).stem)
    is_demo = bool(metadata.get("IS_DEMO", False))

    return SessionData(
        session_id=session_id,
        source_type="FILES",
        source_name=source_name,
        df_ppg_raw=df_ppg_raw,
        df_ppg_replay=df_ppg_replay,
        df_bp_events=df_bp_events,
        metadata=metadata,
        readme_text="",
        is_demo=is_demo,
        load_warnings=all_warnings,
    )
