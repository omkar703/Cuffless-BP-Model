# Phase 2: Window Generation, Quality Control & Leakage-Safe Dataset Report

**Project**: Cuffless Blood Pressure Estimation from PPG (MAX30102 + ESP32)  
**Pipeline Phase**: Phase 2 — Window Extraction, Decoupled Window QC & Partitioning  
**Status**: COMPLETE (Ready for Phase 3 Modeling)  
**Primary Sensor**: PPG ONLY (125 Hz)  
**Reference Signal**: Arterial Blood Pressure (ABP) for ground truth targets only (ECG ignored)  

---

## 1. Executive Summary & Window Yield

- **Window Duration**: 10.0 seconds (1250 samples at 125 Hz)
- **Primary Overlap**: 0% (Independent non-overlapping baseline)
- **Total Master Records Processed**: 12,000
- **Records Eligible for Window Extraction (≥ 10s)**: 10,874 (90.6%)
- **Short Records Preserved (< 10s)**: 1,126 (Retained in master manifest; 0 windows generated)
- **Total Possible Windows Extracted**: 261,658
- **PPG-Valid Windows**: 261,653 (100.0%)
- **ABP-Valid Windows**: 261,344 (99.9%)
- **Final Modeling-Eligible Windows**: **261,339** (99.9%)
- **Rejected Windows**: 319 (0.1%)

> [!NOTE]
> **Decoupled Quality Architecture**: PPG signals and ABP catheter signals are validated independently. Corrupted ABP lines do not cause loss of valid PPG windows; those windows are preserved as `PPG_VALID_ABP_INVALID` for unsupervised or morphological research.

---

## 2. Window Quality Status Breakdown

| Quality Status Category | Description | Window Count | Percentage | Modeling Eligible |
| :--- | :--- | :---: | :---: | :---: |
| **PPG_VALID_ABP_VALID** | Both PPG signal and ABP beat targets are verified | **261,339** | **99.9%** | **YES (Primary Candidate)** |
| **PPG_VALID_ABP_INVALID** | Clean PPG, corrupted/unphysiological ABP | 314 | 0.1% | NO (Preserved for SSL) |
| **PPG_INVALID_ABP_VALID** | Corrupted PPG, valid ABP reference | 5 | 0.0% | NO |
| **PPG_INVALID_ABP_INVALID** | Both PPG and ABP corrupted / artifactual | 0 | 0.0% | NO |

---

## 3. Signal Quality & Primary Artifact Breakdown

### 3.1 PPG Quality Failures
The primary reasons for PPG window rejection:
- **Implausible Pulse Structure / Severe Motion Noise**: Disruption of cardiac pulsatile morphology due to high-frequency motion or baseline swings where fewer than 4 or more than 42 pulses are detected.
- **Clipping / ADC Saturation**: Signal flatlining at top/bottom ADC limits exceeding 5% of window duration.
- **Discontinuities & Step Jumps**: Sensor displacement resulting in non-physiological sample-to-sample amplitude jumps exceeding 30× standard deviation.

### 3.2 ABP Target Failures
The primary reasons for ABP target rejection:
- **Low Valid Beat Ratio (< 60%)**: Arterial line damping, flushing, or motion causing corrupted diastolic/systolic detection.
- **Insufficient Cardiac Cycles (< 3 beats)**: Severe arrhythmia, catheter decoupling, or partial line clamping.
- **Unphysiological Blood Pressure Limits**: Non-physiological pressure values outside [50, 240] mmHg SBP or [30, 140] mmHg DBP.

---

## 4. Blood Pressure Target Distribution (Modeling-Eligible Windows)

| Parameter | Mean ± Std | Min | 25th Pct | Median | 75th Pct | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Systolic BP (SBP)** | 129.3 ± 22.0 mmHg | 68.4 | 112.9 | 127.6 | 144.4 | 198.4 |
| **Diastolic BP (DBP)** | 66.7 ± 11.0 mmHg | 50.0 | 58.5 | 64.5 | 72.5 | 139.4 |
| **Mean Arterial (MAP)** | 87.6 ± 12.8 mmHg | 58.4 | 77.9 | 85.8 | 95.4 | 157.6 |
| **Pulse Pressure (PP)** | 62.6 ± 18.8 mmHg | 10.8 | — | 61.7 | — | 134.3 |
| **Estimated Heart Rate** | 98.7 ± 31.4 bpm | 43.0 | — | 91.5 | — | 227.3 |

---

## 5. Record-Level Partitioning & Leakage Verification

Splitting is strictly enforced on **unique `record_id`** using a deterministic random seed (42).

| Partition | Target % | Total Records Partitioned | Records with Windows (≥10s) | Total Windows | Eligible Windows | SBP Mean ± Std | DBP Mean ± Std | MAP Mean ± Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 70% | 8,400 (70.0%) | 7,622 (70.1%) | 183,747 | **183,517** | 129.3 ± 22.1 | 66.6 ± 11.0 | 87.5 ± 12.8 |
| **Validation** | 15% | 1,800 (15.0%) | 1,630 (15.0%) | 39,502 | **39,461** | 128.9 ± 21.7 | 66.6 ± 10.9 | 87.3 ± 12.7 |
| **Test** | 15% | 1,800 (15.0%) | 1,622 (14.9%) | 38,409 | **38,361** | 130.0 ± 21.8 | 67.3 ± 11.5 | 88.2 ± 12.9 |


### Leakage Audit Results:
- **Train ∩ Validation Record Overlap**: **0** records
- **Train ∩ Test Record Overlap**: **0** records
- **Validation ∩ Test Record Overlap**: **0** records
- **ECG-Derived Columns in Manifest**: **NONE (0 detected)**
- **NaN / Inf in Eligible Targets**: **NONE (0 detected)**

---

## 6. Scientific & Clinical Caveats

> [!CAUTION]
> **Explicit Patient ID Limitation**: Patient identifiers are not available in the public Kaggle / Kachuee MIMIC-II distribution. While record-level partitioning is the strongest achievable defense, it does not strictly prevent identical patients from appearing in different records if recorded across separate ICU sessions.

> [!IMPORTANT]
> **Filter Non-Causality Warning**: The Butterworth filter applied in this phase uses `scipy.signal.filtfilt` (zero-phase forward-backward filtering). This is strictly an **offline research reference filter**. A causal filter (e.g. streaming biquad IIR) must be implemented for embedded ESP32 streaming deployment.

> [!NOTE]
> **No Model Training Undertaken**: As strictly specified for Phase 2, zero machine learning or deep learning models have been trained. All target distributions and splits are documented without model optimization or class manipulation.
