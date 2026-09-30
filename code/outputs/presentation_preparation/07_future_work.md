# 07. Future Work & Staged Engineering Roadmap

**Project**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone  
**Framework**: 4-Tier Staged Roadmap for Translational Research and Edge Deployment

---

## 1. Staged Engineering & Research Roadmap

```
               TIER 1: IMMEDIATE PHYSICAL VALIDATION
           (More Subjects, Dynamic BP Swings, Repeated Sessions)
                                  ↓
                TIER 2: ADVANCED ALGORITHMIC RESEARCH
           (External Datasets, Extreme-BP Loss, Multi-Task)
                                  ↓
                 TIER 3: EMBEDDED EDGE DEPLOYMENT
          (INT8 Quantization, Microcontroller Benchmarking)
                                  ↓
               TIER 4: FORMAL CLINICAL VALIDATION
           (ISO 81060-2 Protocol, Multi-Center Hospital Trial)
```

---

### Tier 1: Immediate Experimental Expansion (0–3 Months)
- **Synchronized Hardware Expansion**:
  - Collect synchronized MAX30102 + reference cuff sessions across 20+ additional human participants across diverse age groups (18–65 years).
  - Perform repeated, multi-day longitudinal sessions on the same individuals to quantify intra-subject diurnal tracking stability.
- **Dynamic Physiological Provocation**:
  - Induce acute hemodynamic swings using controlled non-invasive interventions:
    - *Isometric Handgrip Exercise*: Induces transient elevation in both SBP and systemic vascular resistance.
    - *Cold Pressor Test*: Activates sympathetic vasoconstriction to evaluate model responsiveness to peripheral resistance changes.
    - *Valsalva Maneuver / Postural Shift*: Evaluates transient hypotensive response and baroreflex compensation.
- **Improved Acquisition Hardware**:
  - Implement a customized 3D-printed finger clip with an integrated tension spring to eliminate variable fingertip pressure and reduce motion artifacts.

---

### Tier 2: Advanced Algorithmic Research (3–6 Months)
- **External Multi-Center Public Dataset Validation**:
  - Evaluate the frozen CNN-GRU pipeline across external independent datasets (e.g. VitalDB, University of Queensland Vital Signs Dataset, and MIMIC-III) to evaluate true cross-dataset distribution shift.
- **Zero-Shot / Single-Point Calibration Adaptation**:
  - Investigate hybrid zero-shot demographic conditioning: providing non-invasive patient metadata (age, sex, height, BMI) as auxiliary tabular inputs to adjust the static baseline SBP offset without requiring cuff calibration.
  - Implement a single initial calibration mode ($t=0$) that updates only a scalar bias term while keeping all neural representations frozen.
- **Loss Formulations for Extreme BP Regimes**:
  - Implement focal regression loss, asymmetric Huber loss, or cost-sensitive reweighting to penalize under-prediction in hypertensive crisis ($\ge 160\text{ mmHg}$) and over-prediction in hypotension ($< 90\text{ mmHg}$).
- **Advanced Conformal Calibration**:
  - Explore localized conformal prediction using kernel-density nonconformity scores to produce narrow intervals in high-density regions and wide intervals only where epistemic variance is genuinely high.

---

### Tier 3: Embedded Edge Deployment (6–12 Months)
- **Model Quantization & Pruning**:
  - Post-training quantization of the 1D CNN and causal GRU to 8-bit integer (INT8) representation using TensorFlow Lite for Microcontrollers (TFLM) or ESP-DL.
  - Compress model footprint from ~700 KB (FP32) to under 180 KB (INT8) with $< 0.1\text{ mmHg}$ quantization loss.
- **Native Microcontroller Execution**:
  - Port the causal streaming polyphase resampler, SOS bandpass filter, and quantized neural engine directly onto the **ESP32-S3** microcontroller (utilizing its dual-core 240 MHz Xtensa LX7 processor with vector instructions).
  - Eliminate the host computer entirely, outputting real-time blood pressure estimates directly to an on-board OLED display or BLE smartphone app.
- **System Resource Benchmarking**:
  - Measure true edge latency per 10-second window (target: $< 50\text{ ms}$).
  - Quantify SRAM utilization (target: $< 256\text{ KB}$) and battery power draw during continuous 24-hour acquisition.

---

### Tier 4: Formal Clinical Validation & Regulatory Pathway (12+ Months)
- **Standard-Compliant Protocol Design**:
  - Design a formal clinical trial compliant with **ANSI/AAMI/ISO 81060-2:2018** (Non-invasive sphygmomanometers — Part 2: Clinical investigation of intermittent automated measurement type) and **IEEE Standard 1708-2014** (Standard for Wearable Cuffless Blood Pressure Measuring Devices).
- **Cohort Requirements**:
  - Enforce a minimum of 85 human subjects with demographic stratification:
    - At least 30% male and 30% female participants.
    - Specific blood pressure distribution requirements: at least 5% with SBP $\le 100\text{ mmHg}$, at least 5% with SBP $\ge 160\text{ mmHg}$, and at least 20% with SBP $\ge 140\text{ mmHg}$.
  - Double-blind simultaneous auscultation via two trained independent observers using calibrated mercury/aneroid sphygmomanometers.
- **Regulatory Objective**:
  - Achieve formal AAMI compliance criteria: Mean error (bias) $\le \pm 5.0\text{ mmHg}$ and standard deviation $\le 8.0\text{ mmHg}$ across the standard clinical population.

---

## 2. Final Presentation Conclusion Statement

> *"We have successfully developed, audited, and verified a calibration-free, photoplethysmography-only cuffless blood pressure estimation framework combining classical pulse morphology, multi-channel CNN representation learning, 60-second causal temporal modeling, extreme-aware conformal uncertainty quantification, and a real-time streaming causal DSP engine.*
> 
> *The frozen system has been validated end-to-end: progressing from large-scale public dataset benchmarks through physical MAX30102 hardware replay and an initial synchronized reference-cuff pilot.*
> 
> *While preliminary physical results confirm operational feasibility and outstanding diastolic agreement, broader reference-BP datasets across diverse clinical cohorts are required for definitive accuracy and clinical validation."*
