# Phase 6C: Desktop Reference-Cuff Validation & Analysis Application
**Project**: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only

A local Streamlit desktop analysis application designed for researcher evaluation of physical MAX30102 optical PPG recordings synchronized with reference oscillometric blood pressure measurements.

---

## 1. Scientific Protocol & Safeguards

- **Zero-Retraining Invariant**: Absolutely no neural weights are trained, fine-tuned, or updated. All inferences are generated using the frozen Phase 4A CNN (`best_model_ppg_vpg_apg.pt`, 146,978 parameters) and frozen Phase 4B 1-Layer Unidirectional Causal GRU (`best_temporal_gru.pt`, 27,106 parameters). Total neural parameters = 174,084 (0 trainable).
- **Frozen Calibration**: Extreme-aware post-hoc calibration maps (`isotonic_sbp.pkl`, `isotonic_dbp.pkl`, and `conformal_quantiles_by_bin.json`) from Phase 5C are applied without modification. No curves are fitted to incoming test data.
- **Deterministic Synchronization**: Predictions are paired with reference BP events using a predefined rule (nearest target timestamp within a configurable $\pm 15.0\text{ s}$ tolerance). Manual selection after viewing BP errors is strictly prohibited.
- **Data Preservation**: Raw acquisition inputs are read-only and never modified or overwritten.
- **Non-Medical Research Disclaimer**: This system is strictly an engineering research platform and does not constitute an approved medical device.

---

## 2. Expected Input Package Schema

When receiving data from the acquisition setup, the application expects an archive named `phase6c_session_capture.zip` containing:

```text
phase6c_session_capture.zip
├── ppg_samples.csv
├── bp_events.csv
├── session_metadata.json
└── README.txt
```

### `ppg_samples.csv`
Preferred schema:
```csv
sample_index,device_timestamp_ms,host_timestamp_ms,ir,red
100000,0,1727650000000,165420.0,42.0
100010,10,1727650000010,165890.0,41.5
```
*Tolerant fallback*: If only `sample_index, timestamp_ms, ir, red` are present, `timestamp_ms` is mapped to `host_timestamp_ms`.

### `bp_events.csv`
```csv
event_id,host_timestamp_ms,elapsed_s,event_type,label,sbp,dbp,notes
EVT_01,1727650062000,62.0,REFERENCE_BP_START,Cuff Inflation,NaN,NaN,Inflation started
EVT_02,1727650070000,70.0,REFERENCE_BP_RESULT,OMRON M3,122.0,78.0,Seated resting measurement
```

### `session_metadata.json`
```json
{
  "session_id": "session_subject01_rest",
  "subject_id": "SUBJ_01",
  "sensor_model": "MAX30102",
  "reference_device": "OMRON M3",
  "nominal_fs_hz": 100.0
}
```

---

## 3. Installation & Dependencies

To launch the desktop application, run using the project virtual environment:

```bash
# If needed, install dependencies via uv:
/home/op/.local/bin/uv pip install streamlit plotly

# Launch Streamlit application:
./.venv/bin/streamlit run code/phase6c_app/app.py
```

---

## 4. Desktop Researcher Workflow

1. **Step 1: Load Session**
   - Upload the received `phase6c_session_capture.zip`, upload individual files, or select a local dataset.
   - For offline testing before physical data arrives, select **🧪 Software Demo Mode (Simulation)** to generate synthetic data labeled `DEMO / NOT REAL HARDWARE DATA / NOT RESEARCH RESULT`.
2. **Step 2: Inspect Hardware Data**
   - Review timing jitter, continuity, rate, ADC bounds, and IR statistics. Inspect the interactive waveform timeline.
3. **Step 3: Review BP Events**
   - Audit timestamped reference events and verify temporal alignment with the PPG timeline.
4. **Step 4: Run Frozen Pipeline**
   - Execute the streaming replay simulation (resampling, causal Butterworth SOS, derivatives, CNN embeddings, GRU inference, Phase 5C calibration).
5. **Step 5: Review Predictions**
   - Inspect window QC statuses, temporal BP predictions, and causal waveform traces.
6. **Step 6: Reference-BP Comparison**
   - Deterministically synchronize predictions with reference measurements within the $\pm 15.0\text{ s}$ window.
   - Inspect MAE, RMSE, Mean Error / Bias, SD, Pearson $r$, Spearman $\rho$, BHS error distributions ($\le 5, \le 10\text{ mmHg}$), and AAMI benchmark thresholds (descriptive comparison).
   - Review Bland-Altman and scatter diagnostic plots.
7. **Step 7: Export Results**
   - Save deliverables to `code/outputs/phase6c_reference_validation/` and download the `.zip` archive.

---

## 5. Generated Deliverables

The application produces the following deliverables in `code/outputs/phase6c_reference_validation/`:

- `session_audit.json`: Hardware acquisition integrity statistics.
- `window_quality.csv`: Window-by-window QC metrics.
- `predictions.csv`: Continuous temporal sequence predictions.
- `reference_bp_events.csv`: Standardized reference BP event records.
- `phase6c_prediction_reference_pairs.csv`: Matched prediction-reference pairs with errors and exclusion notes.
- `figures/`: High-resolution diagnostic figures (waveforms, scatter, Bland-Altman, error distributions).
- `phase6c_validation_report.md`: Formal Markdown validation report.
- `phase6c_validation_report.json`: Machine-readable validation summary.
- `run_metadata.json`: Environment, model checksums, and execution timestamps.
- `phase6c_<session_id>_validation_export.zip`: Downloadable archive bundle.
