"""
Classical Baseline Models for Phase 3A PPG-Only Blood Pressure Estimation.

Implements the 5 mandated baselines:
- Model 0: DummyRegressor(strategy="mean")
- Model 1: LinearRegression()
- Model 2: Ridge(alpha) with alpha grid search [0.01, 0.1, 1.0, 10.0, 100.0] tuned on VALIDATION
- Model 3: RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
- Model 4: HistGradientBoostingRegressor(max_iter=150, max_depth=8, random_state=42)

Separate regressors are trained for SBP and DBP.
Strictly CPU-only execution (scikit-learn).
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from utils.logging_utils import setup_logger

logger = setup_logger("classical_models")

RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def get_model_instance(model_name: str, **kwargs) -> Any:
    """Returns an unfitted scikit-learn regressor instance."""
    name = model_name.lower()
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    elif name == "linear":
        return LinearRegression()
    elif name == "ridge":
        alpha = kwargs.get("alpha", 1.0)
        return Ridge(alpha=alpha, random_state=42)
    elif name == "random_forest":
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", 12)
        n_jobs = kwargs.get("n_jobs", -1)
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=n_jobs,
        )
    elif name == "hist_gradient_boosting":
        max_iter = kwargs.get("max_iter", 150)
        max_depth = kwargs.get("max_depth", 8)
        return HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    target_name: str = "BP",
    **kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains specified baseline regressor on (X_train, y_train).
    For Ridge, tunes alpha on (X_val, y_val) validation set if provided.
    
    Returns:
        (fitted_model, metadata_dict)
    """
    name = model_name.lower()
    meta: Dict[str, Any] = {"model_name": model_name, "target": target_name}

    if name == "ridge" and X_val is not None and y_val is not None:
        logger.info(f"Tuning Ridge alpha grid {RIDGE_ALPHA_GRID} on VALIDATION ({target_name})...")
        best_alpha = 1.0
        best_val_mae = float("inf")
        best_model = None

        for alpha in RIDGE_ALPHA_GRID:
            m = Ridge(alpha=alpha, random_state=42)
            m.fit(X_train, y_train)
            preds_val = m.predict(X_val)
            val_mae = mean_absolute_error(y_val, preds_val)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_alpha = alpha
                best_model = m

        logger.info(f"Selected best Ridge alpha={best_alpha} with Val MAE={best_val_mae:.3f} mmHg.")
        meta["best_alpha"] = best_alpha
        meta["best_val_mae"] = float(best_val_mae)
        return best_model, meta

    # Standard fitting
    model = get_model_instance(model_name, **kwargs)
    logger.info(f"Fitting {model_name} on TRAIN (N={len(X_train)}) for {target_name}...")
    model.fit(X_train, y_train)
    return model, meta
