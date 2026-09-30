# 06. Rigorous Scientific Limitations & Methodological Boundaries

**Project**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone  
**Purpose**: Comprehensive documentation of technical, physiological, and clinical limitations framed as rigorous research findings.

---

## 1. Dataset & Partitioning Limitations

### 1.1 Lack of Explicit Subject Identifiers in Kaggle Release
- **Finding**: The public MIMIC-II extract hosted on Kaggle (`BloodPressureDataset`) provides 12,000 multi-minute records partitioned into 12 MATLAB files, but omits master patient demographic keys (Subject IDs).
- **Impact**: While we strictly enforced a **record-level partition** (guaranteeing that 100% of windows from record $i$ reside exclusively in Train, Val, or Test), we cannot definitively exclude the possibility that two distinct records originated from the same ICU patient during different hospital admissions.
- **Scientific Implication**: Our offline generalization is formally **record-independent**, which is drastically superior to leaky random window splitting, but remains an approximation of true multi-patient subject-independent evaluation.

### 1.2 ICU Cohort Bias & Hemodynamic Homogeneity
- **Finding**: MIMIC-II records are derived exclusively from critically ill intensive care unit (ICU) patients subjected to mechanical ventilation, pharmacological vasoactive infusions, and severe comorbidities.
- **Impact**: The learned statistical mapping from optical waveform morphology to blood pressure reflects the altered arterial compliance and elevated vascular resistance typical of an ICU population, rather than young, ambulatory outpatients.

---

## 2. Physiological & Signal-Modality Limitations

### 2.1 Theoretical Boundaries of PPG-Only Sensing
- **Finding**: Photoplethysmography measures cutaneous blood volume pulsatility at the fingertip. It does NOT directly measure arterial wall tension, internal vessel lumen diameter, or central aortic pressure generation.
- **Impact**: In a strictly calibration-free regime, two individuals with identical resting peripheral PPG pulse shapes can possess different central aortic pressures if their aortic compliance, systemic vascular resistance, or arterial dimensions diverge significantly.
- **Research Implication**: This theoretical boundary explains why cuffless models often exhibit static baseline offsets (bias) across diverse individuals when uncalibrated.

### 2.2 Sensitivity to Optical Contact & Motion Artifacts
- **Finding**: Photoplethysmography is highly susceptible to displacement, variable contact pressure, and micro-motion.
- **Impact**: In Phase 6C, we observed that even a transient reduction in finger pressure changes the DC baseline and distorts the dicrotic notch, triggering quality-control rejections. Continuous resting contact is strictly required for stable 60-second temporal inference.

---

## 3. Algorithmic & Modeling Limitations

### 3.1 Regression-to-the-Mean at Blood Pressure Extremes
- **Finding**: Across all classical models (Phase 3A) and deep neural architectures (Phase 4A/4B), estimation errors increase substantially at the distribution tails.
  - In hypotension (SBP $< 90\text{ mmHg}$), the model over-predicts SBP by $+30.9\text{ mmHg}$.
  - In severe hypertension (SBP $\ge 160\text{ mmHg}$), the model under-predicts SBP by $-28.6\text{ mmHg}$.
- **Impact**: Minimizing a continuous regression loss (Huber / MSE) over an imbalanced normotensive dataset penalizes conservative shrinkage toward the population mean (~120/80 mmHg). While temporal GRUs soften this effect, regression-to-the-mean remains a fundamental challenge of uncalibrated neural estimation.

### 3.2 Modest Correlation Between Uncertainty and Absolute Error
- **Finding**: In Phase 5A, Monte Carlo Dropout predictive standard deviation ($\sigma_{\text{MC}}$) exhibited a statistically significant but modest correlation with true absolute error ($r = 0.081$ for SBP, $r = 0.138$ for DBP).
- **Impact**: While high uncertainty reliably identifies increased average error across deciles, individual uncertainty values cannot serve as a deterministic point-by-point error flag. Epistemic dispersion alone does not capture directional bias caused by vascular domain shift.

### 3.3 Subgroup Coverage Disparities in Conformal Intervals
- **Finding**: Standard split conformal prediction (Phase 5B) provides valid marginal coverage over the aggregate test population ($\ge 90\%$), but exhibits conditional under-coverage in extreme hypertensive subgroups ($\le 85\%$).
- **Impact**: Although Phase 5C partially corrected this via asymmetric regime-specific quantiles, intervals must widen significantly (to $\sim 65\text{ mmHg}$) in high-pressure regimes to preserve coverage guarantees, which diminishes clinical utility in acute hypertension.

---

## 4. Hardware Implementation & Physical Pilot Limitations

### 4.1 Exploratory Physical Pilot Sample Size ($N = 7$)
- **Finding**: The Phase 6C physical reference-cuff pilot comprised 5 session packages across 4 healthy human subjects, yielding 7 valid matched prediction-reference pairs.
- **Impact**: While $N = 7$ demonstrates successful operational execution of the entire physical acquisition, streaming DSP, and inference pipeline, it is an **exploratory engineering pilot**, NOT a clinical trial. It does not possess statistical power to demonstrate clinical efficacy or population-level agreement.

### 4.2 Narrow Physiological Blood Pressure Range in Pilot
- **Finding**: All reference oscillometric cuff measurements collected in the physical pilot were clustered within the normal resting range (~120/80 mmHg).
- **Impact**: The physical hardware has not yet been tested against induced hypotensive (e.g. Valsalva maneuver, postural tilt) or hypertensive states (e.g. cold pressor, isometric handgrip), leaving dynamic responsiveness unverified on physical hardware.

### 4.3 Host-Workstation Inference vs. True On-Chip Execution
- **Finding**: The ESP32 firmware currently executes 100 Hz timer acquisition and serial transmission, while causal DSP, neural inference, and reference pairing execute on a host computer.
- **Impact**: While processing latency on the host is sub-millisecond, standalone on-device embedded inference on the microcontroller itself has not yet been implemented or benchmarked for RAM and battery power consumption.

### 4.4 Non-Medical Research Classification
- **Finding**: The entire software and hardware system is an academic research prototype.
- **Impact**: The system has NOT been submitted for FDA, CE, or CDSCO medical device clearance and cannot be used for diagnostic, treatment, or therapeutic decision-making.
