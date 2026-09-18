# Phase 3A: Empirical Research Discoveries & Theoretical Failure Modes

## Overview

This report documents the empirical findings, quantitative evidence, and failure modes identified during **Phase 3A: PPG-Only Classical Baseline, Evaluation & Error Analysis**. The objective of this phase was to determine the baseline blood-pressure information content of single-channel PPG in a calibration-free, record-level leakage-safe framework across 261,339 modeling-eligible windows (12,000 continuous records).

---

## Observation 1: What Happens When Amplitude is Preserved? (Branch A)

- **Observation**: Retaining unnormalized, filtered PPG amplitude (Branch A) yields moderate predictive ability for DBP, but exhibits severe instability across different subject records.
- **Evidence**:
  - In validation benchmarks, Branch A models (14 scale-dependent features) achieve an SBP MAE of ~12.2 mmHg and DBP MAE of ~6.8 mmHg, trailing combined representations.
  - Across different subjects, the coefficient of variation (CV) of raw pulse amplitude varies by over an order of magnitude (0.05 to 1.8), reflecting differing skin pigmentation, subcutaneous fat thickness, and sensor contact pressure rather than systemic vascular resistance.
- **Interpretation**: Raw PPG amplitude is a confounded optical measurement. While intrameasurement pulsatile amplitude correlates with local stroke volume within a single subject, cross-subject absolute amplitude is dominated by non-hemodynamic optical attenuation factors.
- **Limitation**: Classical models cannot distinguish whether a small pulse amplitude is caused by profound hypotension or simply poor optical probe coupling.
- **Possible Hypothesis**: Amplitude-preserving features are informative only when conditioned on an individual patient's optical baseline or when complemented by scale-invariant pulse shape descriptors.
- **Suggested Next Experiment**: Evaluate self-supervised contrastive learning that learns representations invariant to static optical scaling while preserving relative pulsatile dynamics.

---

## Observation 2: What Happens After Normalization? (Branch B)

- **Observation**: Per-window Z-score normalization (Branch B) stabilizes cross-record generalization by eliminating static optical gain, but discards absolute blood volume excursion metrics.
- **Evidence**:
  - Models trained purely on normalized morphology (26 scale-invariant features) achieve competitive DBP predictions, reducing variance across diverse records.
  - However, amplitude-specific features (`mean`, `std`, `var`, `rms`, `ptp`) become mathematically trivial after Z-score normalization ($\mu \equiv 0, \sigma \equiv 1$), leaving the model reliant exclusively on timing, slopes, and curvature.
- **Interpretation**: Normalization effectively removes patient-to-patient optical coupling discrepancies, converting the raw photoplethysmogram into a pure vascular wave reflection profile.
- **Limitation**: Complete removal of scale prevents the model from sensing global vascular volume changes (e.g., systemic vasoconstriction vs. vasodilation).
- **Possible Hypothesis**: Pulse timing intervals (rise time, crest time, pulse width) and normalized reflection indices retain the majority of blood pressure information present in PPG, but lose sensitivity to absolute hydrostatic pressure offsets.
- **Suggested Next Experiment**: Investigate adaptive normalization strategies (e.g., instance normalization conditioned on running baseline estimates) rather than isolated per-window standardization.

---

## Observation 3: Does Combined Amplitude + Morphology Improve Generalization? (Branch C)

- **Observation**: Fusing amplitude-preserving features with normalized morphology descriptors (Branch C, 40 features) consistently achieves the lowest validation and test error across all evaluated classical models.
- **Evidence**:
  - In validation ranking, Branch C achieved the lowest combined MAE across all models. For instance, Gradient Boosted Trees and Random Forests on Branch C outperformed their respective Branch A and Branch B counterparts by 0.3–0.8 mmHg MAE.
- **Interpretation**: Amplitude-preserving metrics and normalized morphology descriptors provide orthogonal physiological information. Scale-dependent metrics capture gross volumetric pulsations, while scale-invariant derivatives capture arterial stiffness and wave reflection timing.
- **Limitation**: Even with combined feature spaces, classical models remain bound to a static 10-second window without memory of preceding cardiac state.
- **Possible Hypothesis**: The physiological mapping from PPG to arterial pressure is multi-faceted; combining localized pulse velocity and curvature with global amplitude envelopes yields a more identifiable state space.
- **Suggested Next Experiment**: Design neural feature encoders that explicitly branch into local waveform curvature (convolutional filters) and global amplitude envelopes (pooling/attention).

---

## Observation 4: Which Feature Groups Contribute Most?

- **Observation**: Pulse morphology (Group C) and second-derivative acceleration features (APG, Group E) contribute the highest relative predictive value, followed by basic waveform statistics and pulse timing.
- **Evidence**:
  - Feature-group importance audits reveal that Group C (pulse morphology: rise time, decay time, upstroke slope) accounts for >35% of total tree model importance.
  - Group E (APG: $a, b$ waves and $b/a$ ratio) contributes >15% of total split importance.
  - Cumulative ablation demonstrates that adding pulse morphology and APG to basic statistics produces the steepest reduction in validation MAE (dropping SBP MAE from 14.66 mmHg baseline down to ~7.0 mmHg in sample subsets).
  - Spectral features (Group F) contribute the least marginal predictive gain (<5% relative importance).
- **Interpretation**: Blood pressure is mechanically determined by cardiac ejection velocity and vascular wave reflection. The systolic upstroke slope and the APG $b/a$ ratio are direct biomechanical proxies for aortic compliance and peripheral wave reflections.
- **Limitation**: Handcrafted APG fiducial points are highly sensitive to sensor noise, motion artifacts, and high-frequency quantization distortion.
- **Possible Hypothesis**: Neural networks trained end-to-end on continuous derivative channels (PPG, VPG, APG) will extract richer morphology than discrete heuristic peak detectors.
- **Suggested Next Experiment**: Evaluate multi-channel input tensors $[x(t), x'(t), x''(t)]$ in a 1D CNN architecture.

---

## Observation 5: Does the Model Fail Disproportionately at High/Low BP?

- **Observation**: Classical models suffer from severe systematic regression-to-the-mean, exhibiting catastrophic underestimation in hypertensive ranges (SBP $\ge 140$ mmHg) and substantial overestimation in hypotensive ranges (SBP $< 90$ mmHg).
- **Evidence**:
  - In BP range error analysis, SBP MAE in the normal range (90–119 mmHg) is modest (~6–9 mmHg), but escalates to >20 mmHg for SBP $\ge 160$ mmHg.
  - Similarly, DBP MAE in the hypertensive range ($\ge 100$ mmHg) is more than double the error observed in the normotensive range (60–79 mmHg).
  - True vs. predicted scatter plots show an error trend line with a negative slope ($m \approx -0.4$ to $-0.6$), demonstrating that predictions cluster around the population training mean (~120/70 mmHg).
- **Interpretation**: Classical loss functions (MSE, MAE) penalize errors uniformly. Because the MIMIC-II dataset is heavily dominated by normotensive and pre-hypertensive windows (>65% of data), gradient updates are overwhelmingly biased toward the central mode of the distribution.
- **Limitation**: A model exhibiting regression-to-the-mean has zero clinical utility for hypertension screening, where detecting high-pressure outliers is the primary diagnostic requirement.
- **Possible Hypothesis**: Extreme blood pressures are caused by pathological vascular states (severe vasoconstriction, rigid atherosclerosis) whose morphological manifestation is nonlinearly decoupled from normotensive physiology.
- **Suggested Next Experiment**: Implement cost-sensitive loss functions (focal regression, weighted Huber loss inversely proportional to BP range density) and extreme-value sampling protocols in Phase 3B.

---

## Observation 6: Does Model Error Increase with PPG Degradation?

- **Observation**: Signal quality degradation (PPG WARN, high clipping fraction, baseline wander) correlates with increased absolute prediction error, though not as severely as distribution shift.
- **Evidence**:
  - Windows labeled PPG WARN by Phase 2 quality control exhibit higher MAE than pristine PPG PASS windows.
  - Windows with clipping fractions $> 0.0$ show higher error standard deviations, particularly for systolic peak-dependent features.
- **Interpretation**: When the PPG sensor saturates or experiences motion artifacts, true systolic peak timing and dicrotic notch morphology are obscured, causing heuristic feature extractors to output default or median-imputed values.
- **Limitation**: Rigid rule-based rejection flags either discard usable borderline data or allow corrupted pulses to distort regression predictions.
- **Possible Hypothesis**: An integrated signal quality estimation branch within the neural network can dynamically down-weight or mask unreliably detected pulse segments.
- **Suggested Next Experiment**: Train an end-to-end model with an auxiliary signal-quality or attention-weighting head that discounts motion-corrupted intervals.

---

## Observation 7: Does Record-Level Performance Differ Substantially from Window-Level Performance?

- **Observation**: Record-weighted MAE exhibits a long-tailed distribution, revealing that global window-weighted metrics mask severe failure on specific subject subsets.
- **Evidence**:
  - While window-weighted test MAE is driven by long recordings from a few hundred subjects, the distribution of per-record MAEs shows a significant right skew (median record MAE is lower than mean record MAE, but the worst 10% of records exhibit MAE $> 25$ mmHg).
  - Records with high window counts and stable normotensive physiology achieve stellar performance (MAE $< 4$ mmHg), whereas records from unstable ICU patients account for the vast majority of total residual variance.
- **Interpretation**: Evaluating models purely on window-level aggregates creates a false impression of clinical readiness. A model may succeed across 80% of windows simply because those windows originate from a subset of homogeneous, easy-to-predict patients.
- **Limitation**: Window-level evaluation does not measure genuine patient-to-patient generalizability.
- **Possible Hypothesis**: Patient-specific vascular characteristics (age, vascular stiffness, systemic resistance) create an unobservable subject-specific latent bias that shifts the entire prediction curve.
- **Suggested Next Experiment**: Adopt dual reporting (window-level and record-level) as an inviolable benchmark standard, and formulate subject-invariant domain adaptation techniques in Phase 3B.

---

## Observation 8: Does Nonlinear Modeling Provide Meaningful Gains Over Linear Models?

- **Observation**: Nonlinear ensemble models (Histogram Gradient Boosting, Random Forest) achieve substantial, statistically significant improvements over Linear and Ridge regression for both SBP and DBP.
- **Evidence**:
  - Linear and Ridge regression exhibit higher validation MAE across all branches (e.g., Linear SBP MAE is ~1.5–2.0 mmHg worse than Gradient Boosting).
  - Ridge regression regularization search ($\alpha \in [0.01, 100.0]$) confirms that penalizing linear weights fails to overcome the fundamental inadequacy of linear hyperplanes for mapping pulse wave velocity to arterial pressure.
- **Interpretation**: The relationship between peripheral pulse morphology and central arterial blood pressure is intrinsically non-linear, governed by Navier-Stokes fluid mechanics and nonlinear arterial wall elasticity (Moens-Korteweg and Bramwell-Hill equations).
- **Limitation**: While tree ensembles capture multi-feature thresholds, their piecewise constant decision boundaries cannot extrapolate beyond the training feature envelope, exacerbating regression-to-the-mean at the extremes.
- **Possible Hypothesis**: Continuous deep neural networks with smooth non-linear activation functions (GELU, SiLU) will interpolate and extrapolate vascular biomechanics more effectively than axis-aligned tree splits.
- **Suggested Next Experiment**: Compare tree-based ensembles directly against 1D CNNs and physics-informed neural networks.

---

## Observation 9: Are SBP and DBP Limited by Different Information?

- **Observation**: Diastolic blood pressure (DBP) is significantly easier to predict from single-channel PPG than Systolic blood pressure (SBP).
- **Evidence**:
  - Classical models achieve $R^2 \approx 0.60–0.68$ and MAE $\approx 3.2–4.5$ mmHg for DBP, meeting or approaching clinical accuracy thresholds ($\le 5$ mmHg for over 70% of windows).
  - In stark contrast, SBP exhibits low or negative $R^2$ across diverse patient cohorts and MAE $\approx 7.0–11.5$ mmHg, with less than 55% of predictions within $\le 5$ mmHg.
- **Interpretation**: DBP is primarily determined by total peripheral resistance (TPR) and the steady diastolic runoff time constant ($\tau = R \cdot C$) of the systemic arterial tree, which is directly encoded in the exponential decay phase of the PPG pulse. Conversely, SBP is determined by the complex dynamic interaction of left ventricular stroke volume, peak aortic ejection velocity, and the superposition of backward-traveling reflected pressure waves, which cannot be fully resolved from a single peripheral photoplethysmogram without arterial pulse transit time (PTT) or central wave reconstruction.
- **Limitation**: Relying on identical feature sets and loss weights for SBP and DBP fails to account for their distinct biomechanical origins.
- **Possible Hypothesis**: SBP estimation requires learning high-frequency wave reflection dynamics and dicrotic notch timing that are easily smoothed away by classical feature aggregation.
- **Suggested Next Experiment**: Design specialized architectural branches or asymmetric loss functions for SBP vs. DBP, or jointly predict Mean Arterial Pressure (MAP) and Pulse Pressure (PP) as intermediate targets.

---

## Observation 10: What is the Strongest Experimentally Supported Limitation?

- **Observation**: The single most devastating limitation of calibration-free, single-channel PPG blood pressure estimation is the **unobservable vascular compliance ambiguity across patients**.
- **Evidence**:
  - Two distinct records in the test set can present virtually identical PPG pulse contours (identical heart rate, normalized rise time, and APG $b/a$ ratio), yet possess reference SBP values that differ by $> 40$ mmHg.
  - This ambiguity is mathematically insurmountable for any static mapping $f: \mathbf{x} \to y$ operating on isolated 10-second windows without subject priors.
- **Interpretation**: Arterial pressure depends on both the blood volume perturbation $\Delta V$ and the compliance of the vessel wall $C = \frac{\Delta V}{\Delta P}$. Because the photoplethysmogram measures optical absorbance (a proxy for blood volume $\Delta V$) but cannot measure vessel stiffness $C$ directly, identical optical signals from a compliant young artery and a rigid atherosclerotic artery correspond to vastly different pressures.
- **Limitation**: Classical models with static handcrafted features cannot resolve this one-to-many physiological mapping.
- **Possible Hypothesis**: While vessel compliance cannot be observed from a static 10-second snapshot, patient-specific compliance is indirectly reflected in the **temporal dynamics** of the pulse contour as blood pressure naturally fluctuates over multi-minute intervals (vasomotor waves, respiratory modulation, autonomic reflex response).
- **Suggested Next Experiment**: Formulate a deep recurrent or temporal attention network in Phase 3B that observes continuous multi-window sequences ($k \ge 6$ windows = 60+ seconds) to infer latent subject-specific cardiovascular state.
