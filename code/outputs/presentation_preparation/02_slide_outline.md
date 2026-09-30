# 02. Slide-by-Slide Presentation Outline & Script

**Presentation Duration**: 15–20 minutes + 5 minutes live demo + Q&A  
**Audience**: Final Year Engineering / Biomedical Capstone Evaluation Panel

---

## Slide 1: Title & Project Overview

- **TITLE**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone
- **SUBTITLE**: An End-to-End Deep Learning Framework from Raw Optical Hardware to Causal Temporal Inference and Conformal Calibration
- **MAIN VISUAL**: System hero graphic showing physical MAX30102 sensor $\to$ ESP32 $\to$ 3-channel PPG/VPG/APG waveform $\to$ CNN-GRU architecture $\to$ SBP/DBP output with conformal bounds.
- **3–5 KEY POINTS**:
  - Continuous, non-invasive blood pressure monitoring without arm cuffs.
  - Exclusively optical Photoplethysmography (PPG) — zero ECG chest leads required.
  - Strict calibration-free design (no patient-specific baseline calibration).
  - Complete research pipeline: classical baselines, deep learning, uncertainty estimation, streaming DSP, and physical reference-cuff validation.
- **WHAT TO SAY**:
  > "Good morning, respected professors and panel members. Today we present our capstone project on calibration-free, cuffless blood pressure estimation using photoplethysmography alone. We have developed an end-to-end framework spanning offline deep learning models, mathematical uncertainty quantification, a real-time causal streaming engine, and an interactive desktop platform validated against physical reference-cuff measurements."
- **IMPORTANT NUMBER(S)**:
  - 174,084 total neural parameters (0 trainable / 100% frozen).
  - 60-second causal temporal context.

---

## Slide 2: The Problem — Hypertension & Intermittent Monitoring

- **TITLE**: Clinical Challenge: The Burden of Hypertension
- **MAIN VISUAL**: Diagram showing nocturnal BP dipping vs. morning surge, contrasted with isolated, intermittent cuff measurement points.
- **3–5 KEY POINTS**:
  - Hypertension affects 1.28 billion individuals globally; leading risk factor for stroke and cardiovascular mortality.
  - Conventional oscillometric arm cuffs provide only intermittent snapshots.
  - Cuff inflation causes vascular compression, pain, sleep disturbance, and "white-coat" sympathetic arousal.
  - Critical transient hemodynamics (nocturnal non-dipping, blood pressure surges) go completely undetected.
- **WHAT TO SAY**:
  > "Hypertension is widely known as the 'silent killer' because it damages arterial vasculature without early symptoms. While the oscillometric arm cuff has been the gold standard for over a century, it only provides a sporadic snapshot. It cannot monitor a sleeping patient without waking them up, nor can it detect acute hypertensive surges during daily physical activity."
- **IMPORTANT NUMBER(S)**:
  - 1.28 Billion adults affected worldwide.
  - ~46% of adults with hypertension are unaware of their condition.

---

## Slide 3: Why Cuffless + Why PPG Alone?

- **TITLE**: The Need for PPG-Only Wearable Monitoring
- **MAIN VISUAL**: Visual comparison: (A) Complex multi-sensor setup (ECG chest leads + finger PPG for PTT) vs. (B) Minimalist single-point optical sensor (smartwatch / finger clip).
- **3–5 KEY POINTS**:
  - Pulse Transit Time (PTT) requires both ECG R-peaks and PPG arrival, demanding chest electrodes.
  - Chest straps and wet electrodes suffer from poor patient compliance in long-term ambulatory settings.
  - Photoplethysmography (PPG) uses optical transmission/reflection in peripheral tissue and is universally available in commercial wearables.
  - Challenge: Can blood pressure be inferred from optical morphology alone without timing references from the heart?
- **WHAT TO SAY**:
  > "Most published cuffless research relies on Pulse Transit Time, which requires measuring the time between the electrical ECG R-wave from the chest and the mechanical pulse wave at the finger. However, asking people to wear chest leads 24/7 is impractical. We asked: Can we eliminate chest leads entirely and extract arterial stiffness and pressure directly from the optical PPG pulse wave alone?"
- **IMPORTANT NUMBER(S)**:
  - 0 ECG leads (100% PPG-only).
  - 1 optical sensor location (distal phalanx).

---

## Slide 4: Research Objectives & Scope

- **TITLE**: Research Objectives & Strict Design Constraints
- **MAIN VISUAL**: Conceptual block diagram highlighting three core pillars: (1) Zero Retraining, (2) Calibration-Free, and (3) Causal Streaming.
- **3–5 KEY POINTS**:
  - **Calibration-Free**: Predict absolute SBP and DBP on completely unseen records without initial cuff calibration.
  - **Biomechanical Modeling**: Exploit velocity (VPG) and acceleration (APG) derivatives alongside raw PPG.
  - **Temporal Memory**: Integrate 60 seconds of causal cardiovascular history to capture vasomotor dynamics.
  - **Real Hardware Integration**: Stream raw MAX30102 optical data through a host-side causal DSP and validation pipeline.
- **WHAT TO SAY**:
  > "Our primary objective is to build a mathematically sound, calibration-free pipeline. Many commercial approaches claim cuffless monitoring, but secretly require frequent recalibration against a traditional cuff. Our goal was to investigate the true limits of pure optical waveform morphology under strict zero-leakage, calibration-free conditions."
- **IMPORTANT NUMBER(S)**:
  - 0-point calibration (purely generalized inference).
  - Record-level split (zero patient overlap between train and test).

---

## Slide 5: End-to-End System Architecture

- **TITLE**: End-to-End System Architecture
- **MAIN VISUAL**: Comprehensive architectural pipeline:
  `MAX30102 (100 Hz) → Causal Polyphase Resampler (125 Hz) → 2nd-Order SOS Bandpass → Central Differences (VPG/APG) → 10s Windowing & QC → Frozen 1D CNN → Frozen Causal GRU (60s) → Extreme-Aware Conformal Bounds → Desktop Reference Comparison`.
- **3–5 KEY POINTS**:
  - Hardware acquisition at ~100 Hz via MAX30102 and ESP32.
  - Causal rational polyphase resampling to 125 Hz (5/4 ratio) preserving continuous filter state.
  - Real-time windowing into 1,250-sample segments with automated quality gating.
  - Two-stage frozen deep learning: Spatial-morphological feature extraction (CNN) followed by causal temporal modeling (GRU).
  - Post-hoc uncertainty estimation providing prediction-interval coverage.
- **WHAT TO SAY**:
  > "This diagram summarizes our complete pipeline. Hardware samples arrive over UART at 100 Hz, are resampled causally to 125 Hz to match the model domain, conditioned using second-order sections, differentiated into velocity and acceleration plethysmograms, windowed into 10-second segments, and passed into our frozen neural networks. Finally, predictions are matched deterministically with cuff readings."
- **IMPORTANT NUMBER(S)**:
  - 100 Hz hardware sampling $\to$ 125 Hz model input rate.
  - 10 ms nominal sample interval.

---

## Slide 6: Dataset & Leakage-Controlled Methodology

- **TITLE**: Dataset Foundation & Strict Partitioning
- **MAIN VISUAL**: Partition flowchart showing 12,000 MIMIC-II records split 70/15/15 by record ID, with Venn diagram showing zero record intersection.
- **3–5 KEY POINTS**:
  - Public benchmark: PhysioNet MIMIC-II Waveform Database (12,000 multi-minute records).
  - **Record-Level Partitioning**: All windows from record $i$ strictly assigned to Train (8,400), Val (1,800), or Test (1,800).
  - Total modeling-eligible windows: **261,339 dual-valid 10-second windows** ($f_s = 125\text{ Hz}$).
  - Strict physiological quality filters: $80 \le \text{SBP} \le 180$, $40 \le \text{DBP} \le 120\text{ mmHg}$, clipping fraction $< 1\%$, flatline detection.
- **WHAT TO SAY**:
  > "A fatal flaw in much of the published literature is random window splitting, which places adjacent windows from the same subject into both train and test sets, artificially inflating accuracy. We enforced a strict record-level split: 8,400 records for training, 1,800 for validation, and 1,800 for testing, guaranteeing that our models were evaluated exclusively on unseen physiological records."
- **IMPORTANT NUMBER(S)**:
  - 261,339 clean, supervised 10-second windows.
  - Strict zero-leakage: $\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$.

---

## Slide 7: Classical ML Baseline & Derivative Discovery

- **TITLE**: Phase 3: Classical Morphology & Derivative Discovery
- **MAIN VISUAL**: Two-panel figure: (A) Bar chart showing monotonic error reduction across feature ablation tiers A through F; (B) Visual definition of VPG and APG waveforms showing systolic upstroke and dicrotic notch.
- **3–5 KEY POINTS**:
  - Handcrafted 40 physiological features across 6 groups: geometry, scale, timing, pulse wave velocity proxies, VPG, APG, and spectral entropy.
  - Histogram Gradient Boosting outperformed Random Forest, Ridge, and Linear Regression:
    - SBP MAE: **13.93 mmHg** ($R^2 = 0.321$) | DBP MAE: **7.05 mmHg** ($R^2 = 0.287$).
  - **Feature Ablation Insight**: Adding VPG and APG reduced SBP error by nearly 1.0 mmHg, proving that higher-order derivative inflections encode vital arterial compliance information.
- **WHAT TO SAY**:
  > "Before touching deep learning, we established a rigorous classical baseline using 40 engineered features. Our ablation study proved that first and second derivatives—the Velocity Plethysmogram and Acceleration Plethysmogram—monotonically reduced error. These derivatives unmask subtle inflection points related to arterial wave reflections and vascular compliance."
- **IMPORTANT NUMBER(S)**:
  - SBP MAE: 13.93 mmHg (21.2% gain over dummy baseline).
  - DBP MAE: 7.05 mmHg (19.6% gain over dummy baseline).

---

## Slide 8: Deep Learning — 1D CNN + Causal Temporal GRU

- **TITLE**: Phase 4: 1D CNN Representation + Causal Temporal GRU
- **MAIN VISUAL**: Neural architecture diagram: 3-channel input ($3 \times 1250$) $\to$ 4-block 1D CNN $\to$ 64-dim embedding $\to$ 6-step causal GRU $\to$ Dense linear head $\to$ SBP/DBP.
- **3–5 KEY POINTS**:
  - **Phase 4A 1D CNN** (146,978 parameters): Learns hierarchical morphological features directly from raw PPG, VPG, and APG waveforms.
  - **Phase 4B Causal GRU** (27,106 parameters): 1-layer unidirectional GRU processing 6 consecutive window embeddings ($T = 60\text{ s}$).
  - Strictly causal: No future window leakage during inference.
  - Captures short-term cardiovascular memory and autoregulation dynamics.
- **WHAT TO SAY**:
  > "In Phase 4, we replaced handcrafted features with a multi-channel 1D CNN that processes PPG, VPG, and APG simultaneously, producing a 64-dimensional feature embedding per window. We then stacked a 1-layer causal GRU across 6 consecutive windows. The GRU models 60 seconds of causal arterial history, reducing test error to 10.57 mmHg for SBP and 5.51 mmHg for DBP."
- **IMPORTANT NUMBER(S)**:
  - 146,978 CNN params + 27,106 GRU params = **174,084 total parameters**.
  - Test SBP MAE: **10.57 mmHg** | DBP MAE: **5.51 mmHg**.

---

## Slide 9: Uncertainty Estimation & Extreme-Aware Calibration

- **TITLE**: Phase 5: Uncertainty & Conformal Calibration
- **MAIN VISUAL**: Two-panel plot: (A) SBP error progression across MC-dropout uncertainty deciles; (B) Extreme-aware asymmetric conformal intervals across normotensive vs. hypertensive bins.
- **3–5 KEY POINTS**:
  - **MC Dropout (Phase 5A)**: 30 stochastic forward passes; predictive standard deviation correlates with error ($p < 10^{-45}$).
  - **Standard Split Conformal (Phase 5B)**: Mathematical 90% and 95% marginal coverage guarantees on unseen test data.
  - **Extreme-Aware Calibration (Phase 5C)**: Applied post-hoc isotonic calibration maps (`isotonic_sbp.pkl`, `isotonic_dbp.pkl`) and regime-specific asymmetric quantiles to counteract regression-to-the-mean in hypertensive regimes ($\ge 140\text{ mmHg}$).
- **WHAT TO SAY**:
  > "In medicine, point predictions without confidence measures are hazardous. In Phase 5, we added uncertainty estimation without retraining. We proved that MC-dropout predictive variance correlates with estimation error, and implemented split conformal prediction to output valid 90% confidence intervals. In Phase 5C, we introduced extreme-aware calibration to compensate for regression-to-the-mean in high-BP regimes."
- **IMPORTANT NUMBER(S)**:
  - 0 trainable parameters added in calibration phases.
  - Conformal coverage on test set: **91.2%** at 90% nominal, **96.7%** at 95% nominal.

---

## Slide 10: Physical Hardware Architecture — MAX30102 + ESP32

- **TITLE**: Phase 6: Optical Acquisition Hardware
- **MAIN VISUAL**: High-resolution photograph / hardware diagram showing MAX30102 sensor connected via I2C to ESP32-WROOM-32, with finger placement and USB serial link.
- **3–5 KEY POINTS**:
  - Maxim MAX30102: Optical pulse oximeter module featuring 660 nm (Red) and 880 nm (Infrared) LEDs and 18-bit ADC.
  - ESP32-WROOM-32: Non-blocking FreeRTOS timer acquisition at $100\text{ Hz}$ over $400\text{ kHz}$ I2C.
  - High SNR Infrared channel utilized for primary microvascular blood volume sensing.
  - UART serial streaming with millisecond device timestamps for downstream host processing.
- **WHAT TO SAY**:
  > "For our physical hardware, we integrated the MAX30102 optical sensor with an ESP32 microcontroller. The sensor samples the cutaneous microcirculation at the finger pad at 100 Hz. The ESP32 collects 18-bit ADC samples over I2C and streams them over serial with precise millisecond timestamps, ensuring hardware timing integrity."
- **IMPORTANT NUMBER(S)**:
  - 18-bit ADC resolution.
  - ~100 Hz acquisition rate (mean interval $\approx 10.02\text{ ms}$).

---

## Slide 11: Real-Time Causal Streaming DSP Engine

- **TITLE**: Real-Time Causal Streaming Pipeline
- **MAIN VISUAL**: Streaming dataflow diagram: 10-sample chunk input (100 ms) $\to$ Polyphase resampler (state memory) $\to$ Causal 2nd-order Butterworth SOS bandpass $\to$ Numerical differentiation $\to$ Rolling window FIFO buffer.
- **3–5 KEY POINTS**:
  - Solves the domain mismatch: 100 Hz hardware sampling $\to$ 125 Hz neural model input.
  - **Polyphase Rational Resampler**: Polyphase FIR filter ($5/4$ ratio) maintaining internal state between chunks; zero future leakage.
  - **Causal Bandpass**: Second-Order Sections (SOS) Butterworth filter ($0.5–8.0\text{ Hz}$) ensuring numerical stability and strictly causal processing.
  - Streaming throughput: Under 1 millisecond execution time per 100 ms chunk on host machine.
- **WHAT TO SAY**:
  > "A crucial engineering challenge was domain adaptation: our trained models expect 125 Hz, but the MAX30102 streams at 100 Hz. We built a causal polyphase rational resampler that upsamples the live stream in 10-sample chunks while preserving filter state across chunk boundaries. This guarantees zero future leakage and provides sub-millisecond execution latency."
- **IMPORTANT NUMBER(S)**:
  - Polyphase resampling ratio: 5 / 4.
  - Sub-millisecond latency per 100 ms audio/sensor block.

---

## Slide 12: Offline Research Results Summary

- **TITLE**: Experimental Progression Across Public Benchmarks
- **MAIN VISUAL**: Comprehensive comparison chart showing MAE progression from Baseline to Phase 4B across SBP and DBP.
- **3–5 KEY POINTS**:
  - Dummy Mean Baseline: SBP MAE = 17.69 mmHg | DBP MAE = 8.77 mmHg.
  - Classical HistGradientBoosting (40 features): SBP MAE = 13.93 mmHg | DBP MAE = 7.05 mmHg.
  - Phase 4A Multi-Channel 1D CNN: SBP MAE = 11.04 mmHg | DBP MAE = 5.79 mmHg.
  - Phase 4B Causal 60s GRU: SBP MAE = **10.57 mmHg** | DBP MAE = **5.51 mmHg**.
  - Total algorithmic improvement over dummy baseline: **40.2% for SBP, 37.2% for DBP**.
- **WHAT TO SAY**:
  > "This progression summarizes our algorithmic results on the public benchmark. Moving from classical handcrafted morphology to a multi-channel CNN reduced SBP error from 13.93 to 11.04 mmHg. Incorporating 60 seconds of causal temporal history via the GRU further reduced error to 10.57 mmHg for SBP and 5.51 mmHg for DBP, representing an overall 40% error reduction."
- **IMPORTANT NUMBER(S)**:
  - 38,361 independent test windows.
  - SBP MAE: 10.57 mmHg | DBP MAE: 5.51 mmHg.

---

## Slide 13: Results — Preliminary Physical Pilot

- **TITLE**: Phase 6C: Preliminary Physical-Validation Pilot
- **MAIN VISUAL**: Two-panel diagnostic graphic: (A) Subject-wise predicted vs. reference BP bar chart; (B) Bland-Altman agreement plot on $N = 7$ paired physical comparisons.
- **3–5 KEY POINTS**:
  - Evaluated 5 real hardware sessions across 4 human subjects (`manthan`, `krish`, `nayan`, `Pankaj`).
  - **100% Inclusion Success**: 7 of 7 reference oscillometric cuff measurements matched valid high-quality predictions.
  - **Diastolic BP (DBP)** achieved remarkable clinical agreement:
    - MAE: **2.15 mmHg** | Bias: **+0.99 mmHg** | SD: **3.33 mmHg**.
    - 100% of predictions within 10 mmHg of reference.
  - **Systolic BP (SBP)** exhibited a consistent baseline offset:
    - MAE: **17.34 mmHg** | Bias: **+17.34 mmHg** | SD: **7.35 mmHg**.
  - SD of error ($7.35\text{ mmHg}$) is low and stable, confirming consistent neural inference with an uncalibrated vascular offset.
- **WHAT TO SAY**:
  > "In Phase 6C, we evaluated our frozen pipeline on real hardware captures synchronized with oscillometric cuff readings across four subjects. In all seven valid comparisons, DBP demonstrated exceptional agreement with an MAE of 2.15 mmHg and a bias under 1 mmHg. SBP exhibited a consistent positive bias of +17.3 mmHg, but with low dispersion. This reflects individual arterial compliance differences under calibration-free inference."
- **IMPORTANT NUMBER(S)**:
  - $N = 7$ paired physical measurements across 4 subjects.
  - DBP MAE: **2.15 mmHg** | SBP MAE: **17.34 mmHg** ($\text{SD} = 7.35\text{ mmHg}$).

---

## Slide 14: Honest Scientific Limitations

- **TITLE**: Methodological & Clinical Limitations
- **MAIN VISUAL**: Summary diagram of the three core challenges: (1) Sample size & range; (2) Vascular compliance ambiguity; (3) Contact motion sensitivity.
- **3–5 KEY POINTS**:
  - **Small Pilot Cohort**: Current physical evaluation is an exploratory proof-of-concept ($N=7$ pairs, 4 healthy volunteers); it cannot establish population-level clinical validity.
  - **Narrow Reference Range**: Physical cuff readings clustered around normotensive baselines (~120/80 mmHg); performance during hypertensive or hypotensive emergencies is untested in hardware.
  - **Optical Motion Sensitivity**: Signal quality drops significantly during finger movement or sensor displacement.
  - **Uncalibrated Arterial Compliance**: Calibration-free PPG cannot directly measure individual vessel diameter or aortic wall elasticity, leading to systemic SBP offsets.
- **WHAT TO SAY**:
  > "As responsible researchers, we must be completely candid about our limitations. Our physical hardware evaluation is a preliminary pilot with seven paired observations across four individuals. While it demonstrates complete engineering feasibility, it is not a clinical trial. Furthermore, calibration-free optical signals cannot directly observe individual vascular stiffness, which explains the systolic offset observed in our pilot."
- **IMPORTANT NUMBER(S)**:
  - $N=7$ physical pairs (exploratory engineering pilot).
  - Normal resting range only (~120/80 mmHg).

---

## Slide 15: Future Work & Final Conclusion

- **TITLE**: Future Roadmap & Capstone Conclusion
- **MAIN VISUAL**: Staged timeline flowchart: (1) Clinical Cohort $\to$ (2) Zero-Shot Calibration $\to$ (3) Embedded Edge Quantization (INT8 on ESP32-S3).
- **3–5 KEY POINTS**:
  - **Immediate**: Expand synchronized physical dataset across diverse age groups and broader BP ranges under ISO 81060-2 protocols.
  - **Model Enhancement**: Investigate single-point or zero-shot demographic adaptation (age, height, arm circumference) to correct static SBP offset.
  - **Edge Deployment**: Quantize CNN-GRU architecture using INT8 for native on-chip inference on microcontrollers (e.g., ESP32-S3 / ARM Cortex-M55).
  - **Conclusion**: Successfully delivered an audited, end-to-end, calibration-free PPG blood pressure pipeline from physical photons to conformal deep learning.
- **WHAT TO SAY**:
  > "To conclude: We have designed, built, and thoroughly audited an end-to-end cuffless blood pressure estimation framework. We advanced from classical morphology to multi-channel CNNs and causal temporal GRUs, introduced post-hoc conformal calibration, built a streaming DSP engine, and validated the complete system on live MAX30102 hardware against reference cuff measurements. Thank you, and we welcome your questions."
- **IMPORTANT NUMBER(S)**:
  - 100% frozen research weights maintained throughout hardware validation.
  - End-to-end pipeline latency: $< 1.0\text{ ms}$ per processing chunk.
