# Final Presentation & Demonstration Checklist

**Event**: Final-Year Engineering Capstone Project Presentation & Live Demo  
**Date**: Tomorrow  
**Project**: Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone

---

## 1. Pre-Presentation Setup Checklist (Do 60–90 Minutes Before)

### Environment & Software
- [ ] **Terminal & Working Directory**: Terminal is opened at `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone`.
- [ ] **Virtual Environment**: `.venv` is verified and functional:
  ```bash
  ./.venv/bin/python -c "import torch, streamlit, plotly, scipy, sklearn; print('All core libraries verified!')"
  ```
- [ ] **Streamlit Test Run**: Launch Streamlit locally to ensure port `8501` is free and browser opens cleanly:
  ```bash
  ./.venv/bin/streamlit run code/phase6c_app/app.py
  ```
- [ ] **Display & Resolution**: Set browser zoom to 90% or 100% so Plotly waveforms and metric cards render comfortably on external HDMI/VGA projectors.

### Data Assets & Samples
- [ ] **Primary Physical Session Package Available**: Ensure `hardware/samples/04_20260930_161820.zip` is present and readable.
- [ ] **Secondary Backup Session Package Available**: Ensure `hardware/samples/02_20260930_145454.zip` and `03_20260930_160648.zip` are accessible.
- [ ] **Precomputed Figures Verified**: Ensure all static figures exist in `code/outputs/phase6c_samples_evaluation/figures/`:
  - `pooled_scatter_sbp_dbp.png`
  - `pooled_bland_altman.png`
  - `subject_wise_bp_comparison.png`

### Presentation Slides & Script Alignment
- [ ] **Zero Retraining Verified**: Confirm slides clearly state that models are 100% frozen (174,084 parameters, 0 trainable).
- [ ] **No Unsupported Compliance Claims**: Ensure slides DO NOT claim "AAMI compliant" or "BHS Grade A" for the prototype; verify wording uses *"Preliminary physical-validation pilot ($N=7$ paired measurements)"*.
- [ ] **Window Count Exact**: Verify slide text reflects **75 total complete windows formed** (73 PASS, 2 WARN, 0 REJECT).
- [ ] **Presenter Notes Ready**: Print or open `02_slide_outline.md` and `05_qna.md` for quick reference during evaluation.

---

## 2. Live Demo Sequence Checklist (3–5 Minute Execution)

- [ ] **Step 1: Ingestion**: Load `04_20260930_161820.zip` via sidebar.
- [ ] **Step 2: Signal Audit**: Point to `14,818` samples, `99.70 Hz` rate, `10.03 ms` interval, and `0 index gaps`.
- [ ] **Step 3: Waveform Inspection**: Zoom in on raw Infrared PPG pulse to show clean arterial pulsatility.
- [ ] **Step 4: Cuff Alignment**: Point out reference oscillometric event markers at $t = 88.0\text{ s}$ and $t = 143.0\text{ s}$.
- [ ] **Step 5: Pipeline Execution**: Click *"Run Frozen Pipeline Replay"* (completes in ~2 seconds).
- [ ] **Step 6: Waveform Derivatives**: Display 3-channel trace showing PPG (red), VPG (blue velocity), and APG (green acceleration).
- [ ] **Step 7: Temporal Predictions**: Show 60-second rolling causal sequence context (6 consecutive 10s windows).
- [ ] **Step 8: Paired Comparison**: Show matched prediction vs reference:
  - Event 2: Ref 120/80 $\to$ Pred 127.8 / 78.2 mmHg (Errors: +7.8 SBP, -1.8 DBP).
  - Event 3: Ref 120/80 $\to$ Pred 127.5 / 78.2 mmHg (Errors: +7.5 SBP, -1.8 DBP).
- [ ] **Step 9: State Limitations**: Conclude demo by stating that this is an exploratory pilot demonstrating functional execution.

---

## 3. Emergency Fallback Plan (If Live App or Projector Glitches)

If the live demo fails for any unforeseen reason:
- [ ] **DO NOT PANIC**: Immediately switch to precomputed static artifacts.
- [ ] **Open Precomputed Figures Folder**: `code/outputs/phase6c_samples_evaluation/figures/`
- [ ] **Display Pooled Analysis Plot**: Open `subject_wise_bp_comparison.png` and `pooled_scatter_sbp_dbp.png`.
- [ ] **Display Verified Matched Pairs Table**: Open `code/outputs/phase6c_samples_evaluation/cross_session_matched_pairs.csv`.
- [ ] **Spoken Pivot**:
  > *"To respect the panel's time, we have pre-rendered the audited results from this exact physical hardware capture. Here you see the identical waveform, derivative channels, and synchronized prediction-reference pairs."*

---

## 4. Key Numbers to Remember When Speaking

- **Total Frozen Parameters**: `174,084` (146,978 CNN + 27,106 GRU; 0 trainable).
- **Temporal Context**: `60 seconds` (6 windows $\times$ 10 seconds).
- **Hardware Sampling Rate**: `~100 Hz` (MAX30102 via ESP32 I2C).
- **Model Sampling Rate**: `125 Hz` (via causal polyphase rational resampler).
- **Public Benchmark Offline MAE**: SBP = `10.57 mmHg`, DBP = `5.51 mmHg` ($N = 31,192$ sequences).
- **Physical Pilot Sample Size**: $N = 7$ paired measurements across 4 subjects.
- **Physical Pilot DBP**: MAE = `2.15 mmHg`, Bias = `+0.99 mmHg`, SD = `3.33 mmHg`.
- **Physical Pilot SBP**: MAE = `17.34 mmHg`, Bias = `+17.34 mmHg`, SD = `7.35 mmHg`.
- **Total Hardware Windows**: `75 windows` formed (73 PASS, 2 WARN, 0 REJECT).
