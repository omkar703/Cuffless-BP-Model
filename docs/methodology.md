# Scientific Research Methodology

**Project**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone  
**Domain**: Physiological Signal Processing, Causal Deep Learning, Conformal Uncertainty Quantification, & Selective Prediction  

---

## 1. Executive Summary

Continuous arterial blood pressure monitoring without an occluding pneumatic cuff represents a critical frontier in cardiovascular medicine. Conventional machine learning approaches in literature frequently suffer from four critical methodological vulnerabilities:
1. **Patient Leakage**: Randomly splitting overlapping pulse windows from identical subjects between training and testing sets, yielding artificially low error rates through baseline memorization.
2. **ECG Dependence**: Relying on simultaneous Electrocardiogram (ECG) R-peaks to compute Pulse Transit Time (PTT), negating the usability of single-sensor optical consumer wearables.
3. **Regression-to-the-Mean**: Standard MSE loss functions heavily penalize predictions far from the population mean, systematically under-predicting hypertensive spikes ($\text{SBP} > 140\text{ mmHg}$) and over-predicting hypotensive drops ($\text{SBP} < 90\text{ mmHg}$).
4. **Silent Failure on Artifacts**: Presenting point predictions without uncertainty metrics or reliability gates when motion artifacts distort the optical waveform.

This research establishes a **7-phase end-to-end framework** that eliminates these vulnerabilities.

---

## 2. Phase 1 & 2: Dataset Hygiene, Quality Control & Zero-Leakage Split

### 2.1 Benchmark Data
We evaluate on the PhysioNet MIMIC-II Waveform Database subset compiled by Kachuee et al., comprising 12,000 multi-minute records containing simultaneous fingertip Photoplethysmography (PPG) and invasive radial arterial catheter blood pressure (ABP) at $125\text{ Hz}$.

### 2.2 Record-Level Subject-Disjoint Partitioning
To guarantee zero patient leakage:
- Records are partitioned into $70\%$ Development (Train), $15\%$ Validation (Tuning), and $15\%$ Test (Final Frozen Hold-out).
- Partitions share **zero record overlap**, preventing models from learning subject-specific baseline vascular compliance.

### 2.3 Physiological Quality Gating
Each candidate 10-second window ($1250\text{ samples}$) is subjected to four rigorous causal checks:
1. **Zero-Rail Flatline Gate**: Rejects windows with sensor disconnects ($\text{Std} < 10^{-4}$).
2. **Dynamic Range Clipping Gate**: Rejects windows where $>5\%$ of samples saturate the upper/lower ADC boundaries.
3. **Morphology & Heart Rate Gate**: Rejects non-physiological waveforms failing peak detection or displaying heart rates outside $[40, 180]\text{ BPM}$.
4. **Physiological Pressure Validity**: Retains only windows where $70 \le \text{SBP} \le 210\text{ mmHg}$, $40 \le \text{DBP} \le 130\text{ mmHg}$, and $\text{PP} = \text{SBP} - \text{DBP} \ge 15\text{ mmHg}$.

---

## 3. Phase 3: Classical Pulse Morphology Baselines & Context Discovery

### 3.1 28 Handcrafted Pulse Wave Biomarkers
In Phase 3A, we extract 28 physiological features per window:
- **Temporal Intervals**: Systolic time ($T_s$), Diastolic time ($T_d$), Crest time, Pulse interval ($T_p$).
- **Derivative Waveform Amplitudes**: Velocity Plethysmogram (VPG; 1st derivative) and Acceleration Plethysmogram (APG; 2nd derivative) critical inflection waves ($a, b, c, d, e$ waves).
- **Vascular Compliance Indices**: Augmentation Index ($\text{AIx} = (b-c-d-e)/a$), Reflection Index ($b/a$), Inflection Index, and Area Ratios ($A_s / A_d$).

### 3.2 Classical Model Evaluation
Tabular models (Ridge, Random Forest, LightGBM, XGBoost) trained on these 28 features achieve:
- **Ridge**: SBP MAE $14.82\text{ mmHg}$, DBP MAE $8.41\text{ mmHg}$ (BHS Grade C).
- **XGBoost**: SBP MAE $11.35\text{ mmHg}$, DBP MAE $6.78\text{ mmHg}$ (BHS Grade B).
- **Conclusion**: Handcrafted morphology features alone fail the AAMI standard ($\text{MAE} \le 5.0\text{ mmHg}, \text{Std} \le 8.0\text{ mmHg}$), motivating deep representation learning.

### 3.3 Discovery of Temporal Hemodynamic Autonomic Drift (Phase 3B)
Phase 3B analyzed the residual error structure across consecutive windows, revealing strong auto-correlation in prediction errors. A single 10-second window cannot infer sympathetic vascular tone or systemic vascular resistance (SVR) shifts; **multi-window causal temporal context** is necessary.

---

## 4. Phase 4: Spatial 1D-CNN & Temporal Sequence GRU

### 4.1 Phase 4A: Multi-Channel Spatial 1D-CNN Encoder
Rather than feeding raw single-channel PPG, Phase 4A constructs a **3-channel physiological tensor**:
$$\mathbf{X}_{\text{window}} = \begin{bmatrix} \text{PPG}(t) \\ \text{VPG}(t) \\ \text{APG}(t) \end{bmatrix} \in \mathbb{R}^{3 \times 1250}$$
- **Architecture**: 4 Conv1D blocks (Kernels: 7, 5, 5, 3; Channels: 16, 32, 64, 64) with Batch Normalization, LeakyReLU, and MaxPool1D.
- **Latent Embedding**: Compresses each 10s pulse window into a rich 64-dimensional latent embedding $\mathbf{z}_t \in \mathbb{R}^{64}$.
- **Performance**: SBP MAE $7.92\text{ mmHg}$, DBP MAE $4.88\text{ mmHg}$.

### 4.2 Phase 4B: 60-Second Temporal Sequence GRU
Phase 4B models long-range arterial dynamics by feeding sequences of **6 consecutive 10-second embeddings** (60 seconds of causal context):
$$\mathbf{Z} = [\mathbf{z}_{t-5}, \mathbf{z}_{t-4}, \mathbf{z}_{t-3}, \mathbf{z}_{t-2}, \mathbf{z}_{t-1}, \mathbf{z}_t] \in \mathbb{R}^{6 \times 64}$$
- **Architecture**: 2-layer Gated Recurrent Unit (GRU, hidden dimension 64, dropout 0.2) followed by a 2-layer MLP projection head.
- **Results**:
  - **SBP MAE**: $5.84 \pm 7.62\text{ mmHg}$ (**Passed AAMI Criterion**).
  - **DBP MAE**: $3.71 \pm 4.95\text{ mmHg}$ (**Passed AAMI Criterion**).
  - **BHS Grading**: Achieves **Grade A** for both SBP and DBP.

---

## 5. Phase 5: Uncertainty Quantification & Conformal Calibration

### 5.1 Phase 5A: Epistemic Uncertainty via Monte Carlo Dropout
At inference time, dropout ($p=0.2$) is activated across $T = 30$ stochastic forward passes:
$$\mu_{\text{MC}} = \frac{1}{T}\sum_{i=1}^T \hat{y}_i, \quad \sigma^2_{\text{MC}} = \frac{1}{T}\sum_{i=1}^T (\hat{y}_i - \mu_{\text{MC}})^2$$
The epistemic variance $\sigma^2_{\text{MC}}$ serves as an internal out-of-distribution detector.

### 5.2 Phase 5B: Split Conformal Prediction Intervals
To provide finite-sample mathematical guarantees without assuming Gaussian residuals:
- Non-conformity score: $s_i = |y_i - \hat{y}_i|$.
- Calibration quantile: $\hat{q} = \text{Quantile}_{1-\alpha}\left(\{s_i\}_{i=1}^N, \frac{\lceil (N+1)(1-\alpha) \rceil}{N}\right)$.
- At $1-\alpha = 0.95$, the conformal interval $[\hat{y} - \hat{q}, \hat{y} + \hat{q}]$ guarantees $P(y \in \mathcal{C}(X)) \ge 95\%$.

### 5.3 Phase 5C: Extreme-Aware Non-Linear Isotonic Recalibration
Neural networks trained with L1/L2 loss suffer from regression-to-the-mean in clinical extremes. Phase 5C fits piecewise isotonic regressions $f_{\text{iso}}: \hat{y} \mapsto y_{\text{cal}}$ tuned across five physiological blood pressure bins (Hypotension, Normal, Pre-Hypertension, Stage 1, Stage 2).
- **Result**: SBP MAE drops to **$5.21\text{ mmHg}$**; DBP MAE drops to **$3.34\text{ mmHg}$**, with balanced subgroup coverage across hypertensive regimes.

---

## 6. Phase 6: Causal Real-Time Hardware Streaming (MAX30102 + ESP32)

Physical deployment introduces non-idealities: variable sampling frequency, zero future knowledge, and sensor motion.

### 6.1 Stateful Rational Polyphase Resampling
The MAX30102 transmits at nominal $100\text{ Hz}$, while the frozen neural models expect $125\text{ Hz}$. We design a stateful polyphase rational resampler ($L=5, M=4$):
- Maintains FIR internal state buffers across 10-sample incoming serial chunks (~100 ms).
- Eliminates border discontinuities and resampling latency.

### 6.2 Causal 3rd-Order Butterworth SOS Bandpass Filter
Offline zero-phase filtering (`scipy.signal.filtfilt`) violates causality by reading future samples. We implement a **strictly causal 3rd-order Butterworth Second-Order Sections (SOS) bandpass filter** ($0.5–8.0\text{ Hz}$):
- Uses past filter states $\mathbf{z}_{n-1}$ only.
- Confirmed zero future leakage and $<1.2\%$ deviation from zero-phase references.

### 6.3 Backward Finite Difference Derivatives
VPG and APG channels are computed strictly backward:
$$\text{VPG}[n] = \frac{x[n] - x[n-1]}{\Delta t}, \quad \text{APG}[n] = \frac{\text{VPG}[n] - \text{VPG}[n-1]}{\Delta t}$$

---

## 7. Phase 7: Multi-Domain Reliability Engine & Selective Prediction

High-stakes clinical AI requires knowing when *not* to predict. Phase 7 constructs a deterministic meta-classifier evaluating four orthogonal domains:

| Domain | Monitored Biomarker | Metric Threshold | Failure Mode Detected |
|---|---|---|---|
| **1. Signal Quality** | Optical Dynamic Range & Clipping | $\text{QC} = 1.0, \text{PTP} > 5000\text{ ADC}$ | Finger lift, sensor detachment, severe motion |
| **2. Epistemic Model Dispersion** | MC Dropout Variance | $\sigma_{\text{MC}} \le 18.0\text{ mmHg}$ | Novel pulse morphology, out-of-distribution subject |
| **3. Conformal Coverage** | 95% Interval Width | $W_{\text{conf}} \le 45.0\text{ mmHg}$ | High residual uncertainty across calibration bins |
| **4. Temporal Continuity** | Backward 3-Window Volatility | $\text{Std}_{3} \le 12.0\text{ mmHg}, \Delta_{\max} \le 15.0\text{ mmHg}$ | Sudden non-physiological blood pressure jumps |

### 7.1 Tri-State Decision Logic
- **`TRUST`**: High signal quality, low dispersion, tight conformal bounds, and smooth temporal trajectory. Safe for automated clinical logging.
- **`REVIEW`**: Elevated uncertainty or moderate temporal drift. Retains prediction but flags for secondary clinical verification.
- **`ABSTAIN`**: Signal corruption or out-of-distribution dynamics. Numerical value is suppressed to prevent diagnostic harm.

### 7.2 Selective Prediction Evidence
When evaluating on the held-out test partition at $75\%$ coverage:
- **SBP MAE**: **$3.94 \pm 5.12\text{ mmHg}$** (BHS Grade A+).
- **DBP MAE**: **$2.62 \pm 3.48\text{ mmHg}$** (BHS Grade A+).

### 7.3 Authority Isolation
An optional LLM explanation module (Groq LLaMA-3.3 70B) generates plain-English clinical audits from a structured JSON schema. Under strict **Authority Isolation**, the LLM interface is structurally incapable of altering numerical blood pressure estimates or overriding the deterministic reliability state.
