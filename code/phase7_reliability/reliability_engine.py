"""
Phase 7: Deterministic Reliability Engine & Selective Prediction
Project: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

Implements research-grade prediction reliability assessment, abstention thresholding,
and structured multi-domain evidence generation.

Three-State Outputs:
- TRUST (Green): High-confidence prediction meeting reliability criteria.
- REVIEW (Yellow): Moderate risk; usable for exploratory display but with elevated indicators.
- ABSTAIN (Red): High risk of exceeding research error thresholds; recommendation to remeasure.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass
class ReliabilityConfig:
    target_error_col: str = "high_error_composite_10"
    trust_quantile: float = 0.70     # Bottom 70% lowest risk on val set -> TRUST
    abstain_quantile: float = 0.90   # Top 10% highest risk on val set -> ABSTAIN
    random_state: int = 42


FEATURE_GROUPS = {
    "signal": [
        "qc_pass", "ppg_ptp", "ppg_std", "ppg_clipped_fraction", 
        "ppg_pulse_count", "estimated_hr_bpm"
    ],
    "uncertainty": [
        "sbp_uncertainty", "dbp_uncertainty"
    ],
    "conformal": [
        "sbp_conformal_width_90", "dbp_conformal_width_90"
    ],
    "temporal": [
        "rolling_sbp_std_3", "rolling_dbp_std_3", 
        "max_abs_sbp_jump_3", "max_abs_dbp_jump_3", 
        "median_abs_sbp_change_3", "median_abs_dbp_change_3"
    ]
}

ALL_RELIABILITY_FEATURES = (
    FEATURE_GROUPS["signal"] + 
    FEATURE_GROUPS["uncertainty"] + 
    FEATURE_GROUPS["conformal"] + 
    FEATURE_GROUPS["temporal"]
)


class RuleBasedReliabilityScorer:
    """
    Method A: Transparent Heuristic Reliability Scorer.
    Combines normalized indicator penalties across signal QC, uncertainty,
    conformal interval width, and temporal instability.
    """
    def __init__(self):
        self.feature_mins = {}
        self.feature_maxs = {}
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        for col in ALL_RELIABILITY_FEATURES:
            if col in X.columns:
                self.feature_mins[col] = float(X[col].quantile(0.01))
                self.feature_maxs[col] = float(X[col].quantile(0.99))
        self.fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("RuleBasedReliabilityScorer must be fitted first.")
        
        n = len(X)
        scores = np.zeros(n, dtype=np.float64)
        
        # 1. Signal QC penalty (weight = 0.25)
        if "qc_pass" in X.columns:
            qc_penalty = (1.0 - X["qc_pass"].values.astype(float))
            scores += 0.25 * qc_penalty
            
        # 2. Epistemic uncertainty penalty (weight = 0.25)
        if "sbp_uncertainty" in X.columns and "dbp_uncertainty" in X.columns:
            u_s = np.clip((X["sbp_uncertainty"].values - 10.0) / 10.0, 0.0, 1.0)
            u_d = np.clip((X["dbp_uncertainty"].values - 5.0) / 5.0, 0.0, 1.0)
            scores += 0.25 * (0.6 * u_s + 0.4 * u_d)
            
        # 3. Conformal width penalty (weight = 0.25)
        if "sbp_conformal_width_90" in X.columns:
            w_s = np.clip((X["sbp_conformal_width_90"].values - 40.0) / 25.0, 0.0, 1.0)
            scores += 0.25 * w_s
            
        # 4. Temporal instability penalty (weight = 0.25)
        if "max_abs_sbp_jump_3" in X.columns and "rolling_sbp_std_3" in X.columns:
            jump = np.clip(X["max_abs_sbp_jump_3"].values / 15.0, 0.0, 1.0)
            std = np.clip(X["rolling_sbp_std_3"].values / 10.0, 0.0, 1.0)
            scores += 0.25 * (0.5 * jump + 0.5 * std)
            
        scores = np.clip(scores, 0.0, 1.0)
        # Return 2D array [P(reliable), P(unreliable)]
        return np.column_stack([1.0 - scores, scores])


class ReliabilityEngine:
    """
    Central Research-Grade Prediction Reliability Engine.
    Coordinates model training, validation-based threshold selection,
    selective abstention, and structured evidence payloads.
    """
    def __init__(self, model_type: str = "hist_gb", config: Optional[ReliabilityConfig] = None):
        self.model_type = model_type
        self.config = config or ReliabilityConfig()
        self.tau_trust = 0.30
        self.tau_abstain = 0.60
        self.feature_names = list(ALL_RELIABILITY_FEATURES)
        
        if model_type == "rule_based":
            self.model = RuleBasedReliabilityScorer()
        elif model_type == "logistic":
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=self.config.random_state))
            ])
        elif model_type == "hist_gb":
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                min_samples_leaf=50,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, df_dev: pd.DataFrame, feature_subset: Optional[List[str]] = None):
        """Fit model strictly on dev partition."""
        if feature_subset is not None:
            self.feature_names = feature_subset
            
        X = df_dev[self.feature_names].copy()
        y = df_dev[self.config.target_error_col].values
        
        self.model.fit(X, y)
        return self

    def tune_thresholds(self, df_val: pd.DataFrame):
        """Select operating thresholds strictly on validation/audit partition."""
        X_val = df_val[self.feature_names].copy()
        risk_scores = self.predict_risk(X_val)
        
        self.tau_trust = float(np.quantile(risk_scores, self.config.trust_quantile))
        self.tau_abstain = float(np.quantile(risk_scores, self.config.abstain_quantile))
        
        print(f"[{self.model_type.upper()}] Operating Thresholds Selected on Validation Set:")
        print(f"  tau_trust (quantile {self.config.trust_quantile:.2f}):   {self.tau_trust:.4f}")
        print(f"  tau_abstain (quantile {self.config.abstain_quantile:.2f}): {self.tau_abstain:.4f}")
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Returns estimated risk score P(high_error) in [0, 1]."""
        X_sub = X[self.feature_names].copy()
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_sub)
            return probs[:, 1]
        elif hasattr(self.model, "decision_function"):
            raw = self.model.decision_function(X_sub)
            return 1.0 / (1.0 + np.exp(-raw))
        else:
            return np.zeros(len(X_sub))

    def predict_state(self, X: pd.DataFrame) -> List[str]:
        """Classifies each row into TRUST / REVIEW / ABSTAIN."""
        scores = self.predict_risk(X)
        states = []
        for s in scores:
            if s <= self.tau_trust:
                states.append("TRUST")
            elif s <= self.tau_abstain:
                states.append("REVIEW")
            else:
                states.append("ABSTAIN")
        return states

    def generate_explanation_payload(
        self,
        sbp_pred: float,
        dbp_pred: float,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates structured JSON evidence object defining the interface boundary
        for future LLM explanation layers (Section 24).
        """
        df_single = pd.DataFrame([features])
        # Ensure all expected columns exist
        for col in self.feature_names:
            if col not in df_single.columns:
                df_single[col] = 0.0
                
        risk_score = float(self.predict_risk(df_single)[0])
        
        if risk_score <= self.tau_trust:
            state = "TRUST"
            recommendation = "Reliable prediction meeting consistency and quality criteria."
        elif risk_score <= self.tau_abstain:
            state = "REVIEW"
            recommendation = "Usable estimate with moderate uncertainty; remeasurement advised if clinically indicated."
        else:
            state = "ABSTAIN"
            recommendation = "High risk of estimation error; do not rely on estimate; repeat measurement."
            
        reasons = []
        # Signal Quality Check
        qc_status = "PASS" if features.get("qc_pass", 1) == 1 else "WARN/REJECT"
        if qc_status != "PASS":
            reasons.append("Marginal PPG signal quality detected during acquisition.")
        else:
            reasons.append("Optical waveform signal quality verified (PASS).")
            
        # Epistemic Uncertainty Check
        u_s = features.get("sbp_uncertainty", 14.0)
        u_d = features.get("dbp_uncertainty", 7.5)
        if u_s > 16.0 or u_d > 9.0:
            unc_level = "ELEVATED"
            reasons.append(f"Elevated model uncertainty (SBP sigma={u_s:.1f} mmHg, DBP sigma={u_d:.1f} mmHg).")
        elif u_s < 12.0 and u_d < 6.5:
            unc_level = "LOW"
            reasons.append("Model predictive dispersion is low and consistent.")
        else:
            unc_level = "MODERATE"
            reasons.append("Model predictive dispersion is within expected bounds.")
            
        # Conformal Width Check
        w_s = features.get("sbp_conformal_width_90", 50.0)
        if w_s >= 55.0:
            conf_level = "WIDE"
            reasons.append(f"Conformal prediction interval is wide ({w_s:.1f} mmHg) reflecting extreme-range variance.")
        else:
            conf_level = "STANDARD"
            reasons.append(f"Conformal prediction interval is standard ({w_s:.1f} mmHg).")
            
        # Temporal Stability Check
        jump_s = features.get("max_abs_sbp_jump_3", 0.0)
        std_s = features.get("rolling_sbp_std_3", 0.0)
        if jump_s > 12.0 or std_s > 8.0:
            temp_level = "UNSTABLE"
            reasons.append(f"Elevated temporal fluctuation between consecutive windows (jump = {jump_s:.1f} mmHg).")
        else:
            temp_level = "STABLE"
            reasons.append("Rolling 60-second temporal predictions show strong stability.")

        payload = {
            "prediction": {
                "sbp": round(float(sbp_pred), 1),
                "dbp": round(float(dbp_pred), 1)
            },
            "reliability_state": state,
            "reliability_score": round(risk_score, 4),
            "reliability_reasons": reasons,
            "signal_quality": qc_status,
            "model_uncertainty": unc_level,
            "conformal_width": conf_level,
            "temporal_stability": temp_level,
            "recommendation": recommendation,
            "disclaimer": "Research reliability indicator — not a medical diagnosis or guarantee of BP accuracy."
        }
        return payload
