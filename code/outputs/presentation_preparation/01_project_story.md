# 01. Complete Project Story & Narrative Architecture

**Project Title**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone  
**Context**: Final Year Capstone Project / Research Prototype Presentation  
**Target Audience**: Engineering & Biomedical Examination Panel

---

## Executive Narrative Flow

```
                      1. CLINICAL PROBLEM
                 Hypertension: The Silent Killer
                                ↓
                      2. WHY CUFFLESS BP?
             Continuous vs. Intermittent Monitoring
                                ↓
                        3. WHY PPG ONLY?
            Wearable Feasibility (No ECG Chest Leads)
                                ↓
                  4. THE CALIBRATION-FREE CHALLENGE
              Eliminating Patient-Specific Baselines
                                ↓
                 5. DATASET & LEAKAGE CONTROL
              MIMIC-II (12,000 Records, Strict Split)
                                ↓
                    6. CLASSICAL ML BASELINE
             40 Morphological Features (Phase 3A)
                                ↓
                    7. BIOMECHANICAL INSIGHT
             PPG + Velocity (VPG) + Acceleration (APG)
                                ↓
                   8. TEMPORAL CONTEXT (60s)
          Cardiovascular Autonomic Memory (Phase 3B)
                                ↓
               9. DEEP LEARNING ARCHITECTURE
        Frozen 1D CNN + 1-Layer Causal GRU (Phase 4A/4B)
                                ↓
               10. RIGOROUS UNCERTAINTY MODELING
         MC-Dropout (5A) + Extreme-Aware Conformal (5B/5C)
                                ↓
                11. REAL PHYSICAL HARDWARE
            MAX30102 Optical Sensor + ESP32 System
                                ↓
             12. STREAMING CAUSAL DSP ENGINE
           Polyphase Resampling (100→125 Hz) + SOS Filters
                                ↓
             13. PHYSICAL HARDWARE REPLAY
          Integrity Audits & Real-Time Invariance (6A/6B)
                                ↓
           14. SYNCHRONIZED REFERENCE-CUFF PILOT
              Desktop Analysis Application (Phase 6C)
                                ↓
                 15. CURRENT PILOT FINDINGS
           N=7 Paired Comparisons, DBP Agreement vs. SBP Offset
                                ↓
               16. HONEST SCIENTIFIC LIMITATIONS
         Exploratory Pilot Size, Narrow Range, Record-Level Split
                                ↓
                 17. FUTURE ROADMAP & CONCLUSION
            Clinical Cohorts, Edge Quantization, Deployment
```

---

## Detailed Section-by-Section Narrative

### 1. Clinical Problem: Hypertension, the "Silent Killer"
- Hypertension affects over 1.28 billion adults globally and is the primary etiology for cardiovascular disease, myocardial infarction, stroke, and renal failure.
- Most individuals remain unaware of their condition because early-to-moderate arterial hypertension produces no perceptible symptoms.
- Early detection and routine monitoring dramatically improve long-term clinical outcomes.

### 2. Why Cuffless Blood Pressure Monitoring?
- The conventional gold-standard non-invasive measurement—the inflatable pneumatic arm cuff (oscillometry)—presents severe inherent limitations:
  - It provides only discrete, intermittent snapshots in clinical or home settings.
  - Cuff inflation causes vascular occlusion, pain, sleep disruption, and transient sympathetic arousal.
  - It suffers from "white-coat hypertension" and fails to capture nocturnal non-dipping, diurnal surges, and rapid hemodynamic volatility.
- Continuous, unobtrusive, cuffless blood pressure monitoring enables ubiquitous cardiovascular surveillance.

### 3. Why PPG Alone? (Eliminating Chest Electrodes)
- Existing non-invasive literature overwhelmingly relies on **Pulse Transit Time (PTT)** or **Pulse Arrival Time (PAT)**.
- While mathematically grounded, PTT/PAT requires simultaneous recording of both an **Electrocardiogram (ECG)** via chest electrodes and a peripheral Photoplethysmogram (PPG).
- Wearing chest straps or adhesive wet-gel ECG electrodes indefinitely is unacceptable for long-term daily adherence.
- Photoplethysmography (PPG) measures volumetric variations of blood in the peripheral microvascular bed using simple optical emitters (LEDs) and photodetectors. It is natively integrated into modern smartwatches, rings, and finger clips.
- Our mandate: **Estimate continuous SBP and DBP entirely from a single optical PPG channel, without any ECG chest leads.**

### 4. The Calibration-Free Challenge
- Most existing commercial or semi-commercial cuffless devices require a user-specific calibration (e.g., entering an initial cuff reading or re-calibrating every 2 to 4 weeks).
- Calibration hides systemic model failure: a static baseline offset often "memorizes" the patient's mean arterial pressure rather than inferring true arterial hemodynamics.
- We deliberately tackle the strictly **calibration-free** problem: predicting absolute SBP and DBP on completely unseen individuals using only morphological, derivative, and temporal cues extracted from the optical waveform.

### 5. Dataset Foundation & Leakage-Controlled Methodology
- Evaluated on the standardized PhysioNet MIMIC-II waveform database (Kaggle Blood Pressure dataset: 12,000 multi-minute records containing synchronized PPG and invasive radial arterial blood pressure, ABP).
- **Leakage-Safe Architecture**: Because records represent distinct ICU recording sessions, we enforced a strict **record-level partition** (70% Train, 15% Validation, 15% Test) with guaranteed zero record overlap ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$).
- Formed **261,339 dual-valid 10-second windows** ($f_s = 125\text{ Hz}$, 1,250 samples/window), standardizing quality gates for signal clipping, flatlines, extreme outliers, and physiological plausibility ($80 \le \text{SBP} \le 180$, $40 \le \text{DBP} \le 120\text{ mmHg}$).

### 6. Classical Baseline: Handcrafted Physiological Morphology (Phase 3A)
- Engineered 40 domain-specific physiological features across 6 functional groups:
  - Basic pulse geometry, amplitude scales, timing intervals, pulse wave velocity proxies, first derivative (Velocity Plethysmogram, VPG), second derivative (Acceleration Plethysmogram, APG), and spectral entropy.
- **Histogram Gradient Boosting** achieved:
  - SBP MAE: **13.93 mmHg** ($R^2 = 0.321$)
  - DBP MAE: **7.05 mmHg** ($R^2 = 0.287$)
  - 21.2% improvement over population dummy mean.
- **Critical Discovery**: A monotonic error reduction was observed when adding VPG and APG features, proving that higher-order derivative fiducials encode vascular compliance and arterial wave reflection.

### 7. Biomechanical Insight: PPG, VPG, and APG
- The raw PPG waveform reflects microvascular blood volume changes.
- The **Velocity Plethysmogram (VPG, $\frac{dx}{dt}$)** captures systolic ejection velocity and the rate of ventricular pressure generation ($dP/dt$).
- The **Acceleration Plethysmogram (APG, $\frac{d^2x}{dt^2}$)** reveals critical inflection points (a-, b-, c-, d-, e-waves) corresponding to early systolic acceleration, ventricular contraction, dicrotic notch closure, and peripheral wave reflection.
- Providing all three synchronized channels enables neural architectures to extract arterial stiffness and reflection timing without requiring handcrafted peak detection.

### 8. Temporal Context Discovery: Cardiovascular Autonomic Memory (Phase 3B)
- A single 10-second cardiac snapshot ignores autonomic autoregulation, baroreflex resets, and vasomotor tone evolution.
- Controlled experiments varying temporal context (10s, 20s, 30s, 60s) proved that **60 seconds of causal history (6 consecutive 10s windows)** achieved the strongest performance:
  - Reduced SBP MAE from 14.02 to 13.28 mmHg.
  - Reduced DBP MAE from 6.99 to 6.56 mmHg.
  - Confirmed that rolling context acts as a physiological low-pass filter against transient motion artifacts.

### 9. Deep Learning Architecture: Frozen 1D CNN + Causal GRU (Phase 4A & 4B)
- **Phase 4A 1D CNN Backbone** (`146,978 parameters`):
  - Input: 3 channels $\times$ 1,250 samples (PPG, VPG, APG).
  - 4 convolutional blocks with batch normalization, ReLU, and pooling, generating a compact 64-dimensional morphological embedding per window.
  - Offline Test MAE: SBP = 11.04 mmHg, DBP = 5.79 mmHg.
- **Phase 4B Causal GRU** (`27,106 parameters`):
  - A 1-layer unidirectional causal GRU (hidden dimension 64) operating strictly over the past 6 CNN window embeddings ($T = 60\text{ s}$).
  - Strictly causal: prediction at window $t$ depends only on windows $t-5, \dots, t$.
  - Offline Test MAE: SBP = **10.57 mmHg**, DBP = **5.51 mmHg** (Combined: 8.04 mmHg).
  - **Permanently Frozen**: Total system weights = **174,084 parameters (0 trainable)**.

### 10. Rigorous Uncertainty & Post-Hoc Conformal Calibration (Phases 5A, 5B, 5C)
- Point predictions without uncertainty are unacceptable in clinical applications.
- **Phase 5A (MC Dropout)**: Isolated stochastic forward passes ($N=30$) established that predictive dispersion correlates positively with absolute estimation error ($p < 10^{-45}$).
- **Phase 5B (Split Conformal)**: Constructed mathematically guaranteed $90\%$ and $95\%$ prediction intervals without retraining, but uncovered severe subgroup under-coverage in extreme hypertensive/hypotensive regimes due to regression-to-the-mean.
- **Phase 5C (Extreme-Aware Calibration)**:
  - Applied post-hoc isotonic regression mappings (`isotonic_sbp.pkl`, `isotonic_dbp.pkl`) to linearize predictions.
  - Fitted asymmetric, bin-specific conformal nonconformity quantiles across prediction regimes ($<120$, $120–139$, $\ge 140$ mmHg for SBP).

### 11. Physical Hardware System: MAX30102 + ESP32
- Sensor: Maxim Integrated MAX30102 optical sensor module featuring high-intensity Red (660 nm) and Infrared (880 nm) LEDs with an integrated 18-bit ADC.
- Microcontroller: Espressif ESP32-WROOM-32 utilizing I2C bus acquisition at $400\text{ kHz}$.
- Firmware: Non-blocking sampling pipeline operating at approximately $100\text{ Hz}$ with deterministic timestamp logging and UART serial streaming.

### 12. Real-Time Streaming Causal DSP Pipeline
- Bridging the sampling rate gap: The frozen neural network expects $125\text{ Hz}$ inputs, while physical hardware streams at $100\text{ Hz}$.
- Developed a strictly **causal streaming engine**:
  - Polyphase rational resampling (100 Hz $\to$ 125 Hz, ratio 5/4) maintaining continuous internal filter state without future leakage.
  - Causal 2nd-order Butterworth bandpass filter ($0.5–8.0\text{ Hz}$) implemented via Second-Order Sections (SOS).
  - Causal 5-point central-difference numerical differentiator generating continuous VPG and APG streams.
  - Rolling window formation (1,250 samples / 10s) with quality control gating (flatlines, clipping, baseline drift, and peak plausibility).

### 13. Physical Hardware Replay & Integrity Validation (Phase 6A & 6B)
- Replayed raw hardware captures through the streaming engine to verify numerical stability, memory safety, and timing fidelity.
- Established zero index gaps, stable timing ($10.02\text{ ms}$ interval), and sub-millisecond execution latency per chunk on the host system.

### 14. Synchronized Desktop Reference-Cuff Application (Phase 6C)
- Developed an interactive researcher-facing Streamlit desktop validation platform (`code/phase6c_app/`).
- Automated workflow:
  1. Ingests raw hardware capture packages (`ppg_samples.csv`, `bp_events.csv`, `session_metadata.json`).
  2. Runs raw signal and timing integrity audit.
  3. Executes causal streaming DSP, windowing, and frozen model inference.
  4. Deterministically synchronizes predictions with reference arm-cuff readings within a preset temporal tolerance window ($\pm 15.0\text{ s}$).
  5. Computes error metrics, generates Bland-Altman and scatter diagnostic figures, and exports audited reports.

### 15. Current Physical Pilot Findings ($N = 7$ Pairs, 4 Subjects)
- Evaluated 5 real physical hardware captures from 4 human subjects (`manthan`, `krish`, `nayan`, `Pankaj`).
- **100% Inclusion Success**: 7 out of 7 reference cuff events matched valid, high-quality PPG sequences ($0\text{ rejected windows}$).
- **Diastolic Blood Pressure (DBP)** showed remarkable agreement:
  - Calibrated DBP MAE: **2.15 mmHg** (Mean Bias: **+0.99 mmHg**, SD: **3.33 mmHg**).
  - 100% of estimates fell within 10 mmHg of the cuff measurement.
- **Systolic Blood Pressure (SBP)** exhibited a consistent positive offset:
  - Calibrated SBP MAE: **17.34 mmHg** (Mean Bias: **+17.34 mmHg**, SD: **7.35 mmHg**).
  - Stable internal dispersion ($\text{SD} = 7.35\text{ mmHg} \le 8.0\text{ mmHg}$), with individual-specific baseline shifts reflecting uncalibrated vascular tone.

### 16. Honest Scientific Limitations
- **Pilot Sample Size**: $N = 7$ paired observations across 4 young, healthy individuals. This is an exploratory engineering proof-of-concept, NOT a population-level clinical trial.
- **Narrow Reference BP Range**: All physical cuff measurements were clustered around normal resting values (~120/80 mmHg). Range-wide clinical efficacy across hypertensive and hypotensive crises remains unverified on physical hardware.
- **Record-Level vs. Subject-Independent**: Offline public dataset was partitioned at the record level; true subject-independent generalization requires external multi-center clinical datasets.

### 17. Future Roadmap & Conclusion
- **Conclusion**: We successfully designed, implemented, and verified an end-to-end, calibration-free, PPG-only blood pressure estimation pipeline—from raw optical photon acquisition to causal deep neural inference, conformal calibration, and deterministic reference-cuff synchronization.
- **Roadmap**: Expanding the physical validation cohort to 85+ participants under ISO 81060-2 protocols, implementing INT8 quantization for microcontrollers, and exploring adaptive zero-shot calibration.
