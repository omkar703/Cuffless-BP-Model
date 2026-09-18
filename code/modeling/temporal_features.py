"""
Temporal Sequence Construction & Dynamic Feature Aggregation Engine for Phase 3B Discovery 1.

Implements:
1. Causal, leak-safe temporal sequence construction across windows of each record.
2. Rigid integrity assertions:
   - Zero future leakage (strictly past-to-current context W_(t-k) ... W_t).
   - Zero record crossing (all context windows within the same record_id).
   - Strict monotonic chronological order of window_index.
   - Target is strictly the current window (SBP_t, DBP_t).
   - Rejection and tracking of windows with INSUFFICIENT_HISTORY.
3. Feature grouping for dynamic temporal statistics:
   - Timing (5): hr_bpm, ibi_mean, ibi_median, ibi_std, ibi_cv
   - Pulse morphology (9): pulse_amp_median_a, pulse_amp_iqr_a, pulse_amp_cv_a,
     pulse_width_median, rise_time_median, decay_time_median,
     max_upstroke_slope_median, max_downstroke_slope_median, pulse_area_median
   - VPG (5): vpg_max, vpg_min, vpg_std, vpg_rms, vpg_max_upstroke
   - APG (4): apg_max, apg_min, apg_std, apg_b_to_a_ratio
   - Spectral (3): dominant_freq, pulse_band_power, spectral_entropy
4. Vectorized computation of 7 temporal summary statistics per dynamic feature:
   - hist_mean, hist_std, hist_min, hist_max, delta_prev, delta_hist_mean, linear_slope
5. Representation definitions for Experiment C and D:
   - C1: PPG scalar/morphology only (31 features)
   - C2: PPG + VPG (36 features)
   - C3: PPG + VPG + APG (40 features)
   - D1: Current PPG only (Context-0)
   - D2: Current PPG + VPG + APG (Context-0)
   - D3: Temporal PPG only (Temporal without derivatives)
   - D4: Temporal PPG + VPG + APG (Combined full representation)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from modeling.features import (
    BRANCH_A_FEATURES,
    BRANCH_B_FEATURES,
    BRANCH_C_FEATURES,
    FEATURE_GROUPS,
    assert_feature_integrity,
    FORBIDDEN_SUBSTRINGS,
)
from utils.logging_utils import setup_logger

logger = setup_logger("temporal_features")


# ==============================================================================
# Dynamic Feature Group Definitions (Section 10)
# ==============================================================================
DYNAMIC_FEATURE_GROUPS = {
    "timing": [
        "hr_bpm",
        "ibi_mean",
        "ibi_median",
        "ibi_std",
        "ibi_cv",
    ],
    "pulse_morphology": [
        "pulse_amp_median_a",
        "pulse_amp_iqr_a",
        "pulse_amp_cv_a",
        "pulse_width_median",
        "rise_time_median",
        "decay_time_median",
        "max_upstroke_slope_median",
        "max_downstroke_slope_median",
        "pulse_area_median",
    ],
    "vpg": [
        "vpg_max",
        "vpg_min",
        "vpg_std",
        "vpg_rms",
        "vpg_max_upstroke",
    ],
    "apg": [
        "apg_max",
        "apg_min",
        "apg_std",
        "apg_b_to_a_ratio",
    ],
    "spectral": [
        "dominant_freq",
        "pulse_band_power",
        "spectral_entropy",
    ],
}

# 26 dynamic features in total
ALL_DYNAMIC_FEATURES = (
    DYNAMIC_FEATURE_GROUPS["timing"]
    + DYNAMIC_FEATURE_GROUPS["pulse_morphology"]
    + DYNAMIC_FEATURE_GROUPS["vpg"]
    + DYNAMIC_FEATURE_GROUPS["apg"]
    + DYNAMIC_FEATURE_GROUPS["spectral"]
)

# 17 dynamic PPG-only features (excluding VPG and APG)
PPG_DYNAMIC_FEATURES = (
    DYNAMIC_FEATURE_GROUPS["timing"]
    + DYNAMIC_FEATURE_GROUPS["pulse_morphology"]
    + DYNAMIC_FEATURE_GROUPS["spectral"]
)

# Derivative feature sets
VPG_FEATURES = DYNAMIC_FEATURE_GROUPS["vpg"]
APG_FEATURES = DYNAMIC_FEATURE_GROUPS["apg"]

# Static representation branches for Experiment C
# C1: PPG only (31 features)
PPG_ONLY_STATIC_FEATURES = [
    f for f in BRANCH_C_FEATURES if f not in VPG_FEATURES and f not in APG_FEATURES
]
# C2: PPG + VPG (36 features)
PPG_VPG_STATIC_FEATURES = [
    f for f in BRANCH_C_FEATURES if f not in APG_FEATURES
]
# C3: PPG + VPG + APG (40 features = Branch C Combined)
PPG_VPG_APG_STATIC_FEATURES = list(BRANCH_C_FEATURES)

# Verify integrity of all feature lists
assert_feature_integrity(ALL_DYNAMIC_FEATURES)
assert_feature_integrity(PPG_DYNAMIC_FEATURES)
assert_feature_integrity(PPG_ONLY_STATIC_FEATURES)
assert_feature_integrity(PPG_VPG_STATIC_FEATURES)
assert_feature_integrity(PPG_VPG_APG_STATIC_FEATURES)


# ==============================================================================
# Automated Integrity Assertions (Section 7)
# ==============================================================================
def assert_temporal_sequence_integrity(
    sequence_matrix: np.ndarray,
    record_ids: np.ndarray,
    window_indices: np.ndarray,
    splits: np.ndarray,
    context_k: int,
) -> None:
    """
    Validates mathematical and chronological integrity of built sequences.
    
    Args:
        sequence_matrix: (M, K+1) array of integer indices into original dataset.
        record_ids: (N,) array of record IDs.
        window_indices: (N,) array of integer window indices.
        splits: (N,) array of partition strings ('train', 'val', 'test').
        context_k: context window count (K in [0, 1, 2, 5]).
    """
    if context_k == 0:
        return

    M, seq_len = sequence_matrix.shape
    assert seq_len == context_k + 1, f"Expected sequence length {context_k + 1}, got {seq_len}"

    if M == 0:
        return

    # Check a deterministic sample or all if M <= 10000
    sample_indices = np.arange(M) if M <= 10000 else np.linspace(0, M - 1, 10000, dtype=int)

    for idx in sample_indices:
        seq = sequence_matrix[idx]
        seq_recs = record_ids[seq]
        seq_win_idxs = window_indices[seq]
        seq_splits = splits[seq]

        # 1. No sequence contains multiple record IDs
        if not np.all(seq_recs == seq_recs[0]):
            raise AssertionError(
                f"INTEGRITY VIOLATION: Sequence {idx} contains multiple record IDs: {np.unique(seq_recs)}"
            )

        # 2. Sequence indices are strictly ordered and contiguous
        diffs = np.diff(seq_win_idxs)
        if not np.all(diffs == 1):
            raise AssertionError(
                f"INTEGRITY VIOLATION: Sequence {idx} is not strictly contiguous: window_indices={seq_win_idxs}, diffs={diffs}"
            )

        # 3. Current window is the final element, no future window appears
        target_win_idx = seq_win_idxs[-1]
        for past_idx in seq_win_idxs[:-1]:
            if past_idx >= target_win_idx:
                raise AssertionError(
                    f"INTEGRITY VIOLATION: Future leakage! past_idx={past_idx} >= target_win_idx={target_win_idx}"
                )

        # 4. All windows belong to the same split
        if not np.all(seq_splits == seq_splits[0]):
            raise AssertionError(
                f"INTEGRITY VIOLATION: Split boundary crossed in sequence {idx}: {np.unique(seq_splits)}"
            )


# ==============================================================================
# Temporal Sequence Builder
# ==============================================================================
class TemporalSequenceBuilder:
    """
    Builds causal, within-record temporal sequences for a given context length K.
    
    For context K:
    - Target window: W_t
    - Historical context: W_(t-K), ..., W_(t-1)
    - Total sequence length: K + 1
    """

    def __init__(self, context_k: int):
        self.context_k = context_k
        self.context_duration_sec = (context_k + 1) * 10.0

    def build_sequences(
        self,
        df_meta: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Constructs context sequences from metadata DataFrame.
        
        Args:
            df_meta: DataFrame with ['record_id', 'window_id', 'split'] and integer 'win_idx'.
                     Must be sorted by record_id and win_idx.
                     
        Returns:
            eligible_target_indices: (M,) indices in df_meta of eligible target windows.
            sequence_matrix: (M, K+1) indices in df_meta of windows in each sequence.
            audit_stats: dict of retention statistics per split.
        """
        K = self.context_k
        N = len(df_meta)
        record_ids = df_meta["record_id"].to_numpy()
        window_ids = df_meta["window_id"].to_numpy()
        win_indices = df_meta["win_idx"].to_numpy()
        splits = df_meta["split"].to_numpy()

        if K == 0:
            # Context-0: every window is its own target and sequence
            eligible_targets = np.arange(N, dtype=np.int64)
            sequence_matrix = eligible_targets[:, np.newaxis]
            audit_stats = self._compute_audit_stats(df_meta, eligible_targets)
            return eligible_targets, sequence_matrix, audit_stats

        # Fast contiguous sequence search grouped by record
        eligible_targets_list = []
        sequence_list = []

        # Find record group boundaries
        change_points = np.where(record_ids[:-1] != record_ids[1:])[0] + 1
        group_starts = np.concatenate([[0], change_points])
        group_ends = np.concatenate([change_points, [N]])

        for start, end in zip(group_starts, group_ends):
            rec_len = end - start
            if rec_len <= K:
                continue  # Insufficient windows in this record

            rec_win_idxs = win_indices[start:end]
            rec_global_idxs = np.arange(start, end, dtype=np.int64)

            # Check for contiguous runs: rec_win_idxs[i] - rec_win_idxs[i-K] == K
            # Vectorized contiguous test
            contiguous_mask = (rec_win_idxs[K:] - rec_win_idxs[:-K]) == K
            valid_offset_indices = np.where(contiguous_mask)[0] + K

            for v_idx in valid_offset_indices:
                seq = rec_global_idxs[v_idx - K : v_idx + 1]
                eligible_targets_list.append(rec_global_idxs[v_idx])
                sequence_list.append(seq)

        if len(eligible_targets_list) > 0:
            eligible_targets = np.array(eligible_targets_list, dtype=np.int64)
            sequence_matrix = np.vstack(sequence_list)
        else:
            eligible_targets = np.empty(0, dtype=np.int64)
            sequence_matrix = np.empty((0, K + 1), dtype=np.int64)

        # Run automated safety assertions
        assert_temporal_sequence_integrity(
            sequence_matrix=sequence_matrix,
            record_ids=record_ids,
            window_indices=win_indices,
            splits=splits,
            context_k=K,
        )

        audit_stats = self._compute_audit_stats(df_meta, eligible_targets)
        return eligible_targets, sequence_matrix, audit_stats

    def _compute_audit_stats(
        self,
        df_meta: pd.DataFrame,
        eligible_targets: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculates missing history and retention statistics across splits."""
        total_windows = len(df_meta)
        eligible_set = set(eligible_targets)
        is_eligible_mask = np.isin(np.arange(total_windows), eligible_targets)

        stats = {
            "context_k": self.context_k,
            "context_duration_sec": self.context_duration_sec,
            "total_windows": total_windows,
            "eligible_windows": len(eligible_targets),
            "insufficient_history_windows": total_windows - len(eligible_targets),
            "overall_retention_pct": float(100.0 * len(eligible_targets) / max(1, total_windows)),
            "splits": {},
        }

        for s in ["train", "val", "test"]:
            split_mask = (df_meta["split"].to_numpy() == s)
            split_tot = int(np.sum(split_mask))
            split_elig = int(np.sum(split_mask & is_eligible_mask))
            split_recs_tot = int(df_meta[split_mask]["record_id"].nunique())
            split_recs_elig = int(df_meta[split_mask & is_eligible_mask]["record_id"].nunique())

            stats["splits"][s] = {
                "total_windows": split_tot,
                "eligible_windows": split_elig,
                "insufficient_history": split_tot - split_elig,
                "retention_pct": float(100.0 * split_elig / max(1, split_tot)),
                "total_records": split_recs_tot,
                "active_records": split_recs_elig,
                "avg_windows_per_active_record": float(split_elig / max(1, split_recs_elig)),
            }

        return stats


# ==============================================================================
# Vectorized Temporal Feature Aggregator (Section 9)
# ==============================================================================
class TemporalFeatureAggregator:
    """
    Computes 7 temporal summary statistics for dynamic features over historical windows:
    1. hist_mean
    2. hist_std
    3. hist_min
    4. hist_max
    5. delta_prev
    6. delta_hist_mean
    7. linear_slope
    """

    def __init__(
        self,
        context_k: int,
        all_feature_names: List[str],
        dynamic_feature_names: List[str],
        static_feature_names: List[str],
    ):
        self.context_k = context_k
        self.all_feature_names = all_feature_names
        self.dynamic_feature_names = dynamic_feature_names
        self.static_feature_names = static_feature_names

        # Map dynamic feature names to column indices in full feature matrix
        feat2idx = {name: idx for idx, name in enumerate(all_feature_names)}
        self.dynamic_indices = np.array([feat2idx[name] for name in dynamic_feature_names], dtype=np.int64)
        self.static_indices = np.array([feat2idx[name] for name in static_feature_names], dtype=np.int64)

        # Precompute OLS linear slope weights for sequence length K+1
        if context_k > 0:
            T = np.arange(context_k + 1, dtype=np.float64)
            T_dev = T - T.mean()
            self.slope_weights = T_dev / np.sum(T_dev**2)
        else:
            self.slope_weights = None

        # Build feature output names
        self.output_feature_names = self._build_feature_names()
        assert_feature_integrity(self.output_feature_names)

    def _build_feature_names(self) -> List[str]:
        """Constructs output feature names for static and temporal features."""
        names = list(self.static_feature_names)
        if self.context_k == 0:
            return names

        # 7 temporal statistics per dynamic feature
        stat_suffixes = [
            "hist_mean",
            "hist_std",
            "hist_min",
            "hist_max",
            "delta_prev",
            "delta_hist_mean",
            "slope",
        ]
        for suffix in stat_suffixes:
            for feat in self.dynamic_feature_names:
                names.append(f"{feat}_{suffix}")

        return names

    def aggregate(
        self,
        features_mat: np.ndarray,
        sequence_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Transforms features matrix into combined static + temporal feature matrix.
        
        Args:
            features_mat: (N, 40) complete array of window features.
            sequence_matrix: (M, K+1) index matrix into features_mat.
            
        Returns:
            X_agg: (M, D_out) aggregated feature matrix.
        """
        K = self.context_k
        M = len(sequence_matrix)

        if K == 0:
            target_indices = sequence_matrix[:, 0]
            return features_mat[target_indices][:, self.static_indices]

        # Extract static features of the target window (final window in sequence)
        target_indices = sequence_matrix[:, -1]
        X_static = features_mat[target_indices][:, self.static_indices]

        # Extract dynamic features across sequence: shape (M, K+1, D_dyn)
        seq_dyn = features_mat[sequence_matrix][:, :, self.dynamic_indices]

        curr_vals = seq_dyn[:, -1, :]        # (M, D_dyn)
        past_vals = seq_dyn[:, :-1, :]       # (M, K, D_dyn)
        prev_vals = seq_dyn[:, -2, :]        # (M, D_dyn)

        # 1. Historical mean
        hist_mean = np.mean(past_vals, axis=1)

        # 2. Historical standard deviation (ddof=0; 0.0 if K=1)
        if K > 1:
            hist_std = np.std(past_vals, axis=1)
        else:
            hist_std = np.zeros_like(hist_mean)

        # 3. Historical min
        hist_min = np.min(past_vals, axis=1)

        # 4. Historical max
        hist_max = np.max(past_vals, axis=1)

        # 5. Immediate change: current - previous
        delta_prev = curr_vals - prev_vals

        # 6. Change vs historical mean
        delta_hist_mean = curr_vals - hist_mean

        # 7. Linear slope across sequence
        linear_slope = np.tensordot(seq_dyn, self.slope_weights, axes=([1], [0]))

        # Stack into single feature matrix
        X_agg = np.hstack([
            X_static,
            hist_mean,
            hist_std,
            hist_min,
            hist_max,
            delta_prev,
            delta_hist_mean,
            linear_slope,
        ])

        return X_agg
