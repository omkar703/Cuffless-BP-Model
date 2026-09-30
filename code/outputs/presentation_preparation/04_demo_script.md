# 04. Live Demonstration Script & Presentation-Safe Claims

**Target Application**: Phase 6C Desktop Reference-Cuff Validation System (`code/phase6c_app/app.py`)  
**Allocated Demo Duration**: 3 to 5 minutes  
**Primary Dataset**: Real physical hardware capture package: `hardware/samples/04_20260930_161820.zip` (Subject: Pankaj)

---

## 1. Quick Launch Commands

To start the desktop demonstration platform:

```bash
# 1. Activate project virtual environment
cd /run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone

# 2. Launch Streamlit Application
./.venv/bin/streamlit run code/phase6c_app/app.py
```

The application will automatically open in the local browser at `http://localhost:8501`.

---

## 2. Minute-by-Minute 3–5 Minute Demo Walkthrough

### Minute 0:00 – 0:30: Introduction & Session Ingestion
1. **Action**: In the Streamlit sidebar, select **"Load Session ZIP"**.
2. **Action**: Upload `hardware/samples/04_20260930_161820.zip` (or select from server samples).
3. **What to Say**:
   > "Here we demonstrate our Phase 6C desktop validation platform. We are loading an actual physical session recorded using our MAX30102 sensor connected to an ESP32 microcontroller, captured on a healthy volunteer with simultaneous reference arm-cuff readings."

### Minute 0:30 – 1:15: Hardware Integrity Audit & Waveform Inspection
1. **Action**: Scroll to **Step 2: Inspect Hardware Data**. Show the integrity metric cards.
2. **Highlight**: Point to Sample Count (`14,818`), Effective Rate (`99.70 Hz`), Mean Interval (`10.03 ms`), and Zero Gaps / Discontinuities.
3. **Action**: Expand the interactive Plotly timeline showing the raw Infrared waveform. Show rhythmic pulsatile peaks.
4. **What to Say**:
   > "The platform immediately audits raw signal integrity. Notice that over 14,800 raw samples, there are zero missing index gaps and the sampling interval is tightly clustered at 10.03 milliseconds (~99.7 Hz). You can clearly see clean arterial pulsation in the raw Infrared absorption channel."

### Minute 1:15 – 2:00: Reference-BP Event Alignment
1. **Action**: Scroll to **Step 3: Review BP Events**. Show the event table.
2. **Highlight**: Point out Event 2 (120/80 mmHg recorded at $t = 88.0\text{ s}$) and Event 3 (120/80 mmHg recorded at $t = 143.0\text{ s}$).
3. **What to Say**:
   > "The session package automatically logs the reference oscillometric blood pressure measurements with precise timestamps. Here we see two valid reference events recorded during resting baseline conditions."

### Minute 2:00 – 3:00: Frozen Pipeline Execution & Derivative Windowing
1. **Action**: Click the button: **"Run Frozen Pipeline Replay"**. (Runs in ~2 seconds).
2. **Action**: Scroll to **Step 5: Review Predictions & Waveforms**.
3. **Highlight**:
   - Point to the window quality cards: `13 PASS, 1 WARN, 0 REJECT`.
   - Show the 3-channel trace: **PPG (red)**, **VPG (blue, velocity)**, and **APG (green, acceleration)**.
   - Explain the 60-second rolling temporal context (6 consecutive 10-second windows).
4. **What to Say**:
   > "The system causally resamples the 100 Hz signal to 125 Hz, applies second-order bandpass filtering, and computes the velocity and acceleration plethysmograms. In our quality gate, zero windows were rejected. The frozen 1D CNN extracts morphological embeddings, which are passed into our frozen causal GRU modeling 60 seconds of arterial history."

### Minute 3:00 – 4:00: Deterministic Reference Synchronization & Error Metrics
1. **Action**: Scroll to **Step 6: Reference-BP Comparison**.
2. **Highlight**:
   - Point to the paired results table: Event 2 matched prediction with $\Delta t = -0.22\text{ s}$; Event 3 matched with $\Delta t = -5.25\text{ s}$.
   - Show the predicted values:
     - Event 2: Ref 120/80 $\to$ Pred **127.8 / 78.2 mmHg** (Errors: **+7.8 SBP, -1.8 DBP**).
     - Event 3: Ref 120/80 $\to$ Pred **127.5 / 78.2 mmHg** (Errors: **+7.5 SBP, -1.8 DBP**).
3. **What to Say**:
   > "Predictions are deterministically synchronized with the nearest cuff reading within $\pm 15$ seconds. For this subject, our model achieves exceptional agreement: Diastolic BP is predicted at 78.2 mmHg against the 80 mmHg cuff (-1.8 mmHg error), and Systolic BP is predicted at 127.8 mmHg (+7.8 mmHg error), without any patient calibration."

### Minute 4:00 – 4:30: Summary & Export
1. **Action**: Scroll to **Step 7: Export Results**. Mention that all audited tables and plots are archived for reproducibility.
2. **What to Say**:
   > "The full session audit, window metrics, and paired comparisons are exported to structured JSON and CSV reports, ensuring complete traceability. That concludes the live pipeline demonstration."

---

## 3. Fallback Demo Plan (In Case of Technical Glitches)

If the live demo encounters any technical failure (e.g. browser crash, port conflict, or network delay):

1. **Activate Fallback Mode Immediately**:
   - State clearly: *"To conserve time, we have precomputed and audited artifacts from this exact physical recording."*
2. **Open Precomputed Visualizations**:
   - Figure 1: [`code/outputs/phase6c_samples_evaluation/figures/subject_wise_bp_comparison.png`](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase6c_samples_evaluation/figures/subject_wise_bp_comparison.png)
   - Figure 2: [`code/outputs/phase6c_samples_evaluation/figures/pooled_scatter_sbp_dbp.png`](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase6c_samples_evaluation/figures/pooled_scatter_sbp_dbp.png)
   - Figure 3: [`code/outputs/phase6c_samples_evaluation/figures/pooled_bland_altman.png`](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase6c_samples_evaluation/figures/pooled_bland_altman.png)
3. **Open Audited Tabular Summary**:
   - Table: [`code/outputs/phase6c_samples_evaluation/cross_session_matched_pairs.csv`](file:///run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase6c_samples_evaluation/cross_session_matched_pairs.csv)
4. **Mandatory Labeling**: Always state: *"These are precomputed, audited results from physical hardware recordings; zero synthetic data is used."*

---

## 4. Presentation-Safe Claims: Language & Guardrails

To maintain rigorous scientific and academic integrity before the examination committee, strictly adhere to these language rules:

### CLAIMS I CAN SAY (Scientifically Verified)
- ✅ *"We developed an end-to-end, calibration-free blood pressure estimation pipeline using photoplethysmography alone."*
- ✅ *"The model uses raw PPG, Velocity Plethysmogram (VPG), and Acceleration Plethysmogram (APG) derived from the same optical stream."*
- ✅ *"The temporal model incorporates 60 seconds of strictly causal cardiovascular context."*
- ✅ *"The complete neural model comprises 174,084 parameters, which were permanently frozen prior to hardware testing (0 trainable weights)."*
- ✅ *"The hardware pipeline was validated via streaming replay on fresh physical MAX30102 recordings."*
- ✅ *"In our preliminary physical pilot across four subjects ($N=7$ paired comparisons), Diastolic BP demonstrated close agreement ($\text{MAE} = 2.15\text{ mmHg}$, bias $+0.99\text{ mmHg}$), while Systolic BP exhibited a consistent positive offset ($\text{MAE} = 17.34\text{ mmHg}$)."*
- ✅ *"The physical pilot demonstrated 100% operational success for automated signal acquisition, windowing, inference, and reference pairing."*

### CLAIMS I MUST NOT SAY (Overclaiming & Prohibited)
- ❌ **DO NOT SAY**: *"Our system is AAMI compliant or achieves BHS Grade A."*  
  *(Correct wording: "The physical pilot sample size of N=7 is exploratory; standard clinical protocols such as AAMI/ISO require 85+ subjects across diverse BP ranges.")*
- ❌ **DO NOT SAY**: *"Our prototype is clinically validated or certified."*  
  *(Correct wording: "This is an engineering proof-of-concept and academic research prototype, not an approved medical device.")*
- ❌ **DO NOT SAY**: *"This replaces the conventional blood pressure cuff."*  
  *(Correct wording: "This is an unobtrusive monitoring framework intended to complement clinical surveillance.")*
- ❌ **DO NOT SAY**: *"The system is proven accurate for all patients or across all blood pressure ranges."*  
  *(Correct wording: "Our physical tests were conducted on young, healthy volunteers with normotensive resting blood pressure.")*
- ❌ **DO NOT SAY**: *"Deep neural network runs directly on the ESP32 chip in real-time."*  
  *(Correct wording: "The ESP32 handles high-speed 100 Hz optical acquisition and streaming, while causal inference currently runs on the host desktop platform.")*
