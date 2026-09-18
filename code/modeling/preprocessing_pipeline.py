"""
Preprocessing Pipeline & Canonical HDF5 Feature Cache for Phase 3A Baseline Modeling.

Implements:
1. Canonical HDF5 Cache Manager (features_cache.h5) for zero-redundancy streaming.
2. Leakage-Safe Imputation: SimpleImputer(strategy='median') fitted STRICTLY on TRAIN.
3. Leakage-Safe Scaling: StandardScaler fitted STRICTLY on TRAIN.
4. Representation Extractor: Selects features for Branch A, Branch B, Branch C,
   or specific Ablation groups.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import h5py
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config.config import OUTPUT_DIR
from modeling.features import (
    BRANCH_A_FEATURES,
    BRANCH_B_FEATURES,
    BRANCH_C_FEATURES,
    FEATURE_GROUPS,
    assert_feature_integrity,
)
from utils.logging_utils import setup_logger

logger = setup_logger("preprocessing_pipeline")

FEATURES_DIR = OUTPUT_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
CANONICAL_H5_CACHE_PATH = FEATURES_DIR / "features_cache.h5"


class FeatureCacheManager:
    """Manages the canonical HDF5 feature cache."""

    @staticmethod
    def save_cache(df: pd.DataFrame, h5_path: Path = CANONICAL_H5_CACHE_PATH) -> None:
        """
        Saves DataFrame containing identifiers, targets, and features into canonical HDF5.
        """
        logger.info(f"Writing canonical feature cache to {h5_path} ({len(df)} rows)...")
        assert_feature_integrity(BRANCH_C_FEATURES)

        # Separate metadata/targets from numerical features
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
        feature_cols = [c for c in df.columns if c not in meta_cols]

        # Verify forbidden terms
        assert_feature_integrity(feature_cols)

        with h5py.File(h5_path, "w") as f:
            # Metadata strings
            f.create_dataset(
                "record_id",
                data=np.array(df["record_id"].astype(str), dtype=h5py.string_dtype(encoding="utf-8")),
            )
            f.create_dataset(
                "window_id",
                data=np.array(df["window_id"].astype(str), dtype=h5py.string_dtype(encoding="utf-8")),
            )
            f.create_dataset(
                "split",
                data=np.array(df["split"].astype(str), dtype=h5py.string_dtype(encoding="utf-8")),
            )
            f.create_dataset(
                "ppg_quality_status",
                data=np.array(
                    df.get("ppg_quality_status", "PASS").astype(str),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                ),
            )
            f.create_dataset(
                "window_quality_status",
                data=np.array(
                    df.get("window_quality_status", "PPG_VALID_ABP_VALID").astype(str),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                ),
            )

            # Targets and continuous meta
            f.create_dataset("sbp", data=df["sbp"].to_numpy(dtype=np.float64))
            f.create_dataset("dbp", data=df["dbp"].to_numpy(dtype=np.float64))
            f.create_dataset("map", data=df["map"].to_numpy(dtype=np.float64))
            if "ppg_clipped_fraction" in df.columns:
                f.create_dataset(
                    "ppg_clipped_fraction",
                    data=df["ppg_clipped_fraction"].to_numpy(dtype=np.float64),
                )

            # Numerical feature matrix
            f_mat = df[feature_cols].to_numpy(dtype=np.float64)
            f.create_dataset("features", data=f_mat, compression="gzip", compression_opts=4)
            f.create_dataset(
                "feature_names",
                data=np.array(feature_cols, dtype=h5py.string_dtype(encoding="utf-8")),
            )

        logger.info(f"Successfully cached {len(df)} windows and {len(feature_cols)} features to {h5_path}.")

    @staticmethod
    def load_cache(h5_path: Path = CANONICAL_H5_CACHE_PATH) -> pd.DataFrame:
        """Loads complete DataFrame from canonical HDF5 cache."""
        if not h5_path.exists():
            raise FileNotFoundError(f"Feature cache does not exist: {h5_path}")

        logger.info(f"Loading canonical feature cache from {h5_path}...")
        with h5py.File(h5_path, "r") as f:
            record_id = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["record_id"][:]]
            window_id = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["window_id"][:]]
            split = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["split"][:]]
            ppg_quality_status = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["ppg_quality_status"][:]
            ]
            window_quality_status = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["window_quality_status"][:]
            ]
            sbp = f["sbp"][:]
            dbp = f["dbp"][:]
            map_bp = f["map"][:]
            ppg_clipped_fraction = (
                f["ppg_clipped_fraction"][:]
                if "ppg_clipped_fraction" in f
                else np.zeros(len(sbp))
            )

            feature_names = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["feature_names"][:]
            ]
            f_mat = f["features"][:]

        df_meta = pd.DataFrame(
            {
                "record_id": record_id,
                "window_id": window_id,
                "split": split,
                "sbp": sbp,
                "dbp": dbp,
                "map": map_bp,
                "ppg_quality_status": ppg_quality_status,
                "window_quality_status": window_quality_status,
                "ppg_clipped_fraction": ppg_clipped_fraction,
            }
        )
        df_feat = pd.DataFrame(f_mat, columns=feature_names)
        df = pd.concat([df_meta, df_feat], axis=1)

        # Integrity check
        assert_feature_integrity(feature_names)
        logger.info(f"Loaded cache: {len(df)} windows, {len(feature_names)} features.")
        return df


class LeakageSafePreprocessor:
    """
    Fits median imputation and standard scaling STRICTLY on the training split,
    then transforms train, val, and test splits independently without leakage.
    """

    def __init__(self, feature_columns: List[str]):
        assert_feature_integrity(feature_columns)
        self.feature_columns = feature_columns
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_train(self, df_train: pd.DataFrame) -> None:
        """Fits imputer and scaler strictly on the training partition."""
        logger.info(f"Fitting preprocessors strictly on TRAIN ({len(df_train)} rows)...")
        X_train_raw = df_train[self.feature_columns].to_numpy(dtype=np.float64)

        # 1. Fit & transform imputer on train
        X_train_imp = self.imputer.fit_transform(X_train_raw)

        # 2. Fit scaler on imputed train
        self.scaler.fit(X_train_imp)
        self.is_fitted = True
        logger.info("Fitted SimpleImputer and StandardScaler on TRAIN.")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms any partition using pre-fitted transformers."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on TRAIN before transforming!")
        X_raw = df[self.feature_columns].to_numpy(dtype=np.float64)
        X_imp = self.imputer.transform(X_raw)
        X_scaled = self.scaler.transform(X_imp)
        return X_scaled

    def fit_transform_train(self, df_train: pd.DataFrame) -> np.ndarray:
        """Convenience method to fit and transform TRAIN in one call."""
        self.fit_train(df_train)
        return self.transform(df_train)
