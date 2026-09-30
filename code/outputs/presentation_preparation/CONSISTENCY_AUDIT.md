# 08. Project Consistency Audit & Claim Verification Report

**Project**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone  
**Execution Date**: 2026-09-30  
**Scope**: Full repository audit across all documentation, reports, metadata, and presentation-facing materials (Phases 1–6C).

---

## 1. Executive Summary of Consistency Findings

A systematic audit across all experimental artifacts revealed strong architectural and numerical alignment across frozen phases, but identified critical presentation risks regarding **overclaiming**, **sample size interpretation**, **dataset splitting terminology**, and an apparent **window count discrepancy**.

### Key Audit Outcomes
1. **Window Count Discrepancy Resolved**: The apparent conflict between "75 complete windows" and "73 windows" was audited against the raw CSV tables. Exactly **73 windows were classified as PASS** and **2 as WARN**, yielding **75 total formed windows (and 0 REJECT)**. Both numbers are factual but describe different subsets.
2. **AAMI / BHS Compliance Claims Removed**: Claims of "AAMI compliance" or "BHS Grade A" in Phase 6C reports were identified as scientifically invalid due to the pilot sample size ($N=7$ paired measurements across 4 subjects). These must strictly be presented as preliminary descriptive observations.
3. **"Subject-Independent" Terminology Corrected**: Because the Kaggle release of MIMIC-II lacks master patient IDs, partitioning is strictly **record-independent** rather than verified subject-independent.
4. **Zero-Retraining Invariant Confirmed**: All neural checkpoints and post-hoc calibration maps remain byte-for-byte identical to their frozen state (174,084 total parameters, 0 trainable).

---

## 2. Comprehensive Itemized Consistency Audit Table

### Issue 1: Window Count Discrepancy in Phase 6C Batch Report
- **File**: `code/outputs/phase6c_samples_evaluation/phase6c_multi_sample_validation_report.md`
- **Location**: Section 1 (Executive Summary) vs. Section 2 (Session Table)
- **Current Statement**: Executive summary states: *"Total 10-Second Windows Formed: 75"*, while the session table lists: `18/0/0`, `15/1/0`, `13/0/0`, `13/1/0`, `14/0/0` (sum of leading numbers = 73).
- **Problem**: Readers summing only the first column of the triplet ($18 + 15 + 13 + 13 + 14 = 73$) perceive a numerical contradiction.
- **Root Cause Verified via Code**:
  - `session_02_20260930_145454`: 18 PASS, 0 WARN, 0 REJECT = 18 total
  - `session_03_20260930_160648`: 15 PASS, 1 WARN, 0 REJECT = 16 total
  - `session_03_20260930_161457`: 13 PASS, 0 WARN, 0 REJECT = 13 total
  - `session_04_20260930_161820`: 13 PASS, 1 WARN, 0 REJECT = 14 total
  - `session_05_20260930_162345`: 14 PASS, 0 WARN, 0 REJECT = 14 total
  - Total PASS windows = $18 + 15 + 13 + 13 + 14 = 73$.
  - Total WARN windows = $0 + 1 + 0 + 1 + 0 = 2$.
  - Total Windows Formed = $73 + 2 = 75$ complete 10-second windows.
- **Correct Presentation Wording**:
  > *"75 complete 10-second windows were formed across the 5 sessions (comprising 73 PASS and 2 WARN quality windows, with 0 REJECT). Both PASS and WARN windows are admitted to the rolling sequence buffer, forming 50 temporal sequences."*
- **Source Code/Data Modification**: No data modification needed. Presentation text and report clarification updated.

---

### Issue 2: AAMI and BHS Compliance Claims on Exploratory Pilot Data
- **File**: `code/outputs/phase6c_samples_evaluation/phase6c_multi_sample_validation_report.md`
- **Location**: Section 4 (Pooled Statistical Accuracy Metrics) and Section 6
- **Current Statement**: *"AAMI Compliance Status: MET (for DBP)"* and *"BHS Accuracy Grade: Grade A (for DBP)"*.
- **Problem**: ANSI/AAMI/ISO 81060-2 and BHS validation protocols strictly mandate a minimum of 85 human subjects with rigorous demographic and blood-pressure range stratification. Evaluating 7 paired measurements from 4 healthy subjects and claiming "AAMI compliant" is scientifically invalid and an overclaim.
- **Correct Presentation Wording**:
  > *"Pilot physical-validation observations ($N=7$ paired measurements across 4 subjects). For Diastolic BP, preliminary errors were low (MAE 2.15 mmHg, bias +0.99 mmHg, SD 3.33 mmHg), with 100% of pilot estimates falling within 10 mmHg of reference cuff readings. However, this dataset is too small and narrow in BP range to establish formal AAMI compliance or population-level clinical validity."*
- **Source Code/Data Modification**: Presentation materials strictly updated. `phase6c_multi_sample_validation_report.md` updated to remove formal compliance claims.

---

### Issue 3: "Subject-Independent" vs. "Record-Independent" Partitioning
- **File**: `code/outputs/reports/PHASE3A_PAPER_NOTES.md` and historical markdown files
- **Location**: Section 1
- **Current Statement**: *"predict continuous SBP and DBP in a strictly calibration-free, subject-independent setting"*.
- **Problem**: The public Kaggle release of the MIMIC-II database provides 12,000 multi-minute records without linking master patient/subject identifiers across files. While our partitioning strictly separated records (guaranteeing that 100% of windows from record $i$ reside in exactly one partition with zero record leakage), we cannot mathematically guarantee that two records did not originate from the same patient across ICU stays.
- **Correct Presentation Wording**:
  > *"Evaluated under a strict record-level, leakage-controlled partition across 12,000 records. Because patient demographic identifiers were omitted from the public Kaggle extract, this represents a rigorous record-independent evaluation rather than a verified subject-independent benchmark."*
- **Source Code/Data Modification**: No code changes needed; presentation terminology updated.

---

### Issue 4: Obsolete ECG Literature vs. PPG-Only Architecture
- **File**: `EXISTING_WORK_ANALYSIS.md`
- **Location**: Sections 5.7, 5.8, 8.0
- **Current Statement**: Detailed analysis of prior Kaggle notebooks showing heavy dependence on ECG leads (PTT, PAT, ECG Hjorth mobility, ECG entropy).
- **Problem**: A casual reviewer might mistakenly assume the present project requires ECG or PTT.
- **Correct Presentation Wording**:
  > *"Prior literature relied heavily on Pulse Transit Time (PTT) derived from dual-channel ECG + PPG. In this project, ECG chest leads were completely eliminated. Our architecture is 100% PPG-only, extracting hemodynamic and stiffness cues exclusively from optical pulse morphology and its higher-order derivatives (VPG and APG)."*
- **Source Code/Data Modification**: None. Clarification highlighted in Slide 3 and Q&A 1.

---

### Issue 5: Obsolete IMU / Motion Accelerometer Assumptions
- **File**: Various preliminary proposal notes
- **Location**: Methodology section
- **Current Statement**: Preliminary queries occasionally asked whether an IMU was incorporated.
- **Problem**: No IMU is present on the physical MAX30102 hardware or in any neural architecture.
- **Correct Presentation Wording**:
  > *"No IMU or accelerometer is used. All signal conditioning, quality gating, and blood pressure inferences are derived exclusively from the optical Infrared PPG waveform."*
- **Source Code/Data Modification**: None. Verified that zero IMU dependencies exist in codebase.

---

### Issue 6: Causal GRU Architecture Parameter Breakdown
- **File**: `code/outputs/phase4b_temporal_gru/reports/PHASE4B_TEMPORAL_GRU_REPORT.md`
- **Location**: Section 3
- **Current Statement**: *"Total trainable parameters: 27,106; Frozen CNN: 146,978"*.
- **Problem**: Ensure consistent reporting of total system parameters vs. component parameters across all slides and documents.
- **Verified Breakdown**:
  - Phase 4A Multi-Channel 1D CNN: **146,978 parameters**
  - Phase 4B 1-Layer Causal GRU + Head: **27,106 parameters**
  - Combined Total Neural Parameters: **174,084 parameters**
  - Trainable parameters during Phase 5 and Phase 6: **Exactly 0 (100% frozen)**
- **Correct Presentation Wording**:
  > *"Total system consists of 174,084 parameters (146,978 CNN backbone + 27,106 causal GRU), permanently frozen with zero trainable parameters during hardware validation."*
- **Source Code/Data Modification**: None. Verified in checkpoint audit.

---

### Issue 7: Python Environment & Execution Path
- **File**: `code/README.md`
- **Location**: Section "Getting Started"
- **Current Statement**: `conda activate ppg_bp`
- **Problem**: Local project execution environment is standardized on `.venv` (`/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/.venv/bin/python`, Python 3.10), not Conda.
- **Correct Presentation Wording**:
  > *"Use the local project virtual environment: `./.venv/bin/python` and `./.venv/bin/streamlit run code/phase6c_app/app.py`."*
- **Source Code/Data Modification**: Documented in `04_demo_script.md` and `PRESENTATION_CHECKLIST.md`.

---

## 3. Verified Frozen Research Numbers Summary

The following authoritative metrics must be used consistently across all presentation deliverables:

| Benchmark / Phase | Metric Key | Authoritative Value | Note |
|:---|:---|:---:|:---|
| **Phase 3A Classical** | SBP MAE / DBP MAE | **13.93 / 7.05 mmHg** | 38,361 test windows; 40 features; HGB |
| **Phase 4A 1D CNN** | SBP MAE / DBP MAE | **11.04 / 5.79 mmHg** | 38,361 test windows; 3 channels (PPG/VPG/APG) |
| **Phase 4B Causal GRU** | SBP MAE / DBP MAE | **10.57 / 5.51 mmHg** | 31,192 test sequences; 60s causal context |
| **Phase 5C Conformal** | SBP / DBP Point MAE | **10.73 / 5.66 mmHg** | Post-hoc isotonic calibration |
| **Phase 5C Coverage** | 90% Nominal Coverage | **91.8% SBP / 91.5% DBP** | Valid marginal test coverage |
| **Phase 6C Physical Pilot**| Matched Pairs ($N$) | **7 pairs (4 subjects)** | 100% inclusion rate |
| **Phase 6C Physical Pilot**| Calibrated DBP MAE | **2.15 mmHg** (Bias: +0.99) | Descriptive pilot observation |
| **Phase 6C Physical Pilot**| Calibrated SBP MAE | **17.34 mmHg** (Bias: +17.34)| Descriptive pilot observation |
| **Phase 6C Windows** | Total Windows Formed | **75 (73 PASS, 2 WARN, 0 REJECT)** | Verified from `window_quality.csv` |

---

## 4. Conclusion of Consistency Audit

All historical research values from Phases 1 through 6B are verified as sound, reproducible, and internally consistent. Presentation-facing language has been strictly sanitized to eliminate overclaiming, clarify exploratory pilot sample boundaries, and provide an accurate engineering narrative for tomorrow's final project presentation.
