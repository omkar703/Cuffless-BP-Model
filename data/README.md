# Dataset Documentation & Provenance

This directory documents the physiological datasets, partitioning protocols, and physical sensor acquisitions used in this research.

---

## 1. Primary Benchmark: UCI MIMIC-II Cuffless Blood Pressure Dataset

### Overview
- **Name**: Cuff-Less Blood Pressure Estimation Dataset
- **Curator**: Mohamad Kachuee et al.
- **Source**: Derived from the PhysioNet MIMIC-II (Medical Information Mart for Intensive Care) Clinical Waveform Database.
- **Repository Link**: [Kaggle Dataset: mkachuee/BloodPressureDataset](https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset)
- **Local Location**: `BloodPressureDataset/` (`part_1.mat` through `part_12.mat`, ~4.6 GB).

### Signal Characteristics
Each record part contains paired biological signals sampled synchronously at **$125\text{ Hz}$**:
1. **PPG Channel**: Fingertip optical photoplethysmogram (Pulsatile blood volume).
2. **ABP Channel**: Ground-truth invasive radial arterial catheter blood pressure waveform (mmHg).
3. **ECG Channel**: Electrocardiogram Lead II ($\text{mV}$). *Note: In accordance with this project's clinical goals, ECG is strictly excluded from model inference to achieve single-sensor optical estimation.*

---

## 2. Leakage Prevention: Record-Level Subject-Disjoint Partitioning

A critical flaw identified in earlier literature (e.g. random window shuffling) is patient data leakage, where 10-second segments from the same patient appear in both training and test sets, artificially inflating reported accuracy by memorizing baseline arterial compliance.

To enforce scientific validity, this project implements **Strict Record-Level Partitioning**:
- **Zero Subject Overlap**: Training, validation, and test sets share zero records.
- **Partition Distribution**:
  - **Development / Train**: $70\%$ of records.
  - **Calibration / Validation**: $15\%$ of records (reserved strictly for tuning conformal quantiles and reliability operating thresholds).
  - **Evaluation / Test**: $15\%$ of records (held out completely until final frozen evaluation).
- Partition manifest is saved at: `code/outputs/splits/record_split.csv`.

---

## 3. Window Extraction & Quality Gating

From the 12,000 raw patient records, the pipeline extracts non-overlapping **10-second windows** ($1250\text{ samples}$ @ $125\text{ Hz}$). Every candidate window passes through 4 deterministic physiological quality gates:
1. **Flatline Detection**: Rejects windows with sensor disconnects or ADC zero-rail states.
2. **Clipping & Saturation Gate**: Rejects windows where $>5\%$ of samples saturate the upper or lower dynamic range.
3. **Pulsatile Morphology Gate**: Computes peak-to-peak amplitude ($>0.05$ normalized units) and enforces physiological heart rates ($40\text{ to }180\text{ BPM}$).
4. **Physiological BP Range**: Restricts invasive reference values to clinical survival limits ($70 \le \text{SBP} \le 210\text{ mmHg}$, $40 \le \text{DBP} \le 130\text{ mmHg}$, and $\text{Pulse Pressure} \ge 15\text{ mmHg}$).

**Yield**: $261,658$ fully qualified physiological windows cataloged in `code/outputs/windows/window_manifest.csv`.

---

## 4. Physical Wearable Acquisitions (MAX30102 + ESP32)

In addition to the public MIMIC-II database, this research validates model generalization against real-world optical pulse waveforms acquired using consumer-grade optoelectronic hardware:
- **Sensor**: Maxim Integrated MAX30102 (Dual Red and Infrared optical reflective sensor, 18-bit delta-sigma ADC).
- **Controller**: Espressif ESP32-WROOM-32 (240 MHz dual-core microcontroller).
- **Physical Samples**: Located in `hardware/samples/` (`02_20260930_145454.zip` through `05_20260930_162345.zip`), containing raw $100\text{ Hz}$ serial logs paired with simultaneous Omron oscillometric arm-cuff reference measurements.

---

## 5. How to Obtain the Raw Dataset

If replicating from scratch without bundled MATLAB files:
```bash
# Using Kaggle CLI
kaggle datasets download -d mkachuee/BloodPressureDataset -p BloodPressureDataset/ --unzip
```
For testing and development without the 4.6 GB download, use the sample records bundled in `BloodPressureDataset/Samples/` (`rec_1.csv` through `rec_100.csv`).
