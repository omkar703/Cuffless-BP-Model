# Research Notebooks Index

This directory archives the Jupyter research notebooks for:
**"Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone"**.

---

## 1. Directory Structure

```text
notebooks/
├── README.md                                       # This notebook index and guide
├── 01_dataset_exploration_and_cleaning.ipynb       # Signal distributions, clipping, and outlier filtering
├── 02_window_extraction_and_quality_control.ipynb  # 10s window extraction & 4 physiological quality gates
├── 03_pulse_morphology_classical_baselines.ipynb   # 28 handcrafted morphology features & tabular ML
├── 04_autonomic_drift_temporal_analysis.ipynb      # Inter-window residual auto-correlation analysis
├── 05_single_channel_cnn_baseline.ipynb            # Baseline single-channel 1D-CNN (PPG only)
├── 06_multichannel_cnn_spatial_encoder.ipynb       # 3-channel [PPG+VPG+APG] 1D-CNN spatial encoder
├── 07_temporal_sequence_hemodynamic_gru.ipynb      # 2-layer Sequence GRU over 60s causal context
├── 08_epistemic_uncertainty_mc_dropout.ipynb       # Test-time Monte Carlo Dropout variance estimation
├── 09_split_conformal_prediction_intervals.ipynb   # Distribution-free 95% split conformal intervals
├── 10_extreme_aware_isotonic_recalibration.ipynb   # Monotonic isotonic recalibration across BP regimes
├── 11_hardware_causal_dsp_conditioning.ipynb      # Stateful polyphase resampler & 3rd-order Butterworth SOS
├── 12_realtime_streaming_inference_buffer.ipynb    # 100 ms chunk streaming buffer & rolling sequence inference
├── 13_colab_gpu_training_pipeline.ipynb            # Google Colab cloud GPU training pipeline
└── benchmarks/                                     # Literature Baseline Implementations
    ├── 01_abp_estimation.ipynb
    ├── 02_blood_pressure_trial_2.ipynb
    ├── 03_bloodpressure_abp.ipynb
    ├── 04_bloodpressure_analysis.ipynb
    ├── 05_cnn_lstm.ipynb
    ├── 06_comparing_models_ppg.ipynb
    ├── 07_ecg_ppg_fusion.ipynb
    ├── 08_health_monitoring.ipynb
    ├── 09_ml_bp.ipynb
    └── 10_predict_min_max_bp_ecg.ipynb
```

---

## 2. Research Progression & Execution Sequence

```text
[Dataset & Quality Control]
01_dataset_exploration_and_cleaning.ipynb ──> 02_window_extraction_and_quality_control.ipynb
                                                           │
                                                           ▼
                                                [261,658 Clean Windows]
                                                           │
                        ┌──────────────────────────────────┴──────────────────────────────────┐
                        ▼                                                                     ▼
           [Morphology Baselines & Context]                                       [Spatial Feature Learning]
           03_pulse_morphology_classical_baselines.ipynb                          05_single_channel_cnn_baseline.ipynb
           04_autonomic_drift_temporal_analysis.ipynb                             06_multichannel_cnn_spatial_encoder.ipynb
                                                                                              │
                                                                                              ▼
                                                                                  [Temporal Modeling]
                                                                                  07_temporal_sequence_hemodynamic_gru.ipynb
                                                                                              │
                                                                                              ▼
                                                                                  [Uncertainty & Calibration]
                                                                                  08_epistemic_uncertainty_mc_dropout.ipynb
                                                                                  09_split_conformal_prediction_intervals.ipynb
                                                                                  10_extreme_aware_isotonic_recalibration.ipynb
                                                                                              │
                                                                                              ▼
                                                                                  [Hardware Streaming]
                                                                                  11_hardware_causal_dsp_conditioning.ipynb
                                                                                  12_realtime_streaming_inference_buffer.ipynb
```

---

## 3. Detailed Notebook Table

| Notebook Filename | Research Stage | Inputs & Data | Key Outputs | Key Finding |
|---|---|---|---|---|
| `01_dataset_exploration_and_cleaning.ipynb` | Data Audit | Raw `BloodPressureDataset/*.mat` | Signal distributions, flatline & clipping stats | Characterized 12,000 multi-minute records from PhysioNet MIMIC-II; established physiological dynamic boundaries. |
| `02_window_extraction_and_quality_control.ipynb` | Preprocessing | Raw records | `window_manifest.csv`, `record_split.csv` | Extracted 261,658 10-second windows ($1250\text{ samples}$); enforced zero patient leakage across splits. |
| `03_pulse_morphology_classical_baselines.ipynb` | Feature Extraction | Window manifest | `baseline_test_predictions.csv` | 28 pulse wave morphology features with XGBoost yield SBP MAE 11.35 mmHg (BHS Grade B; fails AAMI SP10). |
| `04_autonomic_drift_temporal_analysis.ipynb` | Context Analysis | Baseline residuals | Error correlation heatmaps | Residuals show strong temporal auto-correlation; isolated 10s windows cannot capture autonomic vasomotor drift. |
| `05_single_channel_cnn_baseline.ipynb` | Neural Baseline | Raw PPG ($1 \times 1250$) | Baseline 1D-CNN weights | Single-channel 1D-CNN achieves SBP MAE 8.42 mmHg; struggles with vascular wave reflections. |
| `06_multichannel_cnn_spatial_encoder.ipynb` | Deep Architecture | 3 channels: [PPG, VPG, APG] | `best_model_ppg_vpg_apg.pt` | Multi-channel derivatives compress 10s windows into 64-dim embeddings; SBP MAE drops to 7.92 mmHg. |
| `07_temporal_sequence_hemodynamic_gru.ipynb` | Sequence Modeling | 6 consecutive embeddings (60s) | `best_temporal_gru.pt` | 2-layer GRU captures long-range hemodynamic dynamics; SBP MAE drops to 5.84 mmHg (**Meets AAMI SP10**). |
| `08_epistemic_uncertainty_mc_dropout.ipynb` | Uncertainty | Trained sequence model | MC variance logs | Test-time MC dropout variance ($T=30$) strongly correlates with out-of-distribution pulse waveforms. |
| `09_split_conformal_prediction_intervals.ipynb` | Conformal Calibration | Validation residuals | Non-conformity quantiles $\hat{q}$ | Distribution-free finite-sample guarantees: 95% of ground-truth points fall within conformal bounds. |
| `10_extreme_aware_isotonic_recalibration.ipynb` | Recalibration | Sequence predictions | `isotonic_sbp.pkl`, `isotonic_dbp.pkl` | Monotonic isotonic regression restores linear fidelity in hypertensive ($>140$) and hypotensive ($<90$) regimes; SBP MAE 5.21 mmHg. |
| `11_hardware_causal_dsp_conditioning.ipynb` | Physical DSP | MAX30102 serial logs | Causal vs. offline filter comparisons | Stateful polyphase resampler ($100\text{ Hz} \to 125\text{ Hz}$) and causal 3rd-order Butterworth SOS bandpass introduce zero future leakage. |
| `12_realtime_streaming_inference_buffer.ipynb` | Real-Time Engine | 100 ms serial chunks | Streaming trajectories | Stateful 6-window sequence buffer enables real-time continuous inference with sub-10 ms latency. |
| `13_colab_gpu_training_pipeline.ipynb` | Cloud Training | Google Colab GPU | Synchronized checkpoint bundle | Standalone training script for executing neural model training on cloud GPU instances. |

---

## 4. Execution & Environment

All notebooks use the project virtual environment (Python 3.10+):

```bash
source .venv/bin/activate
jupyter lab
```

### Path Resolution
Notebooks automatically detect the repository root and append `code/` to `sys.path`. They execute cleanly whether launched from `notebooks/` or the repository root.
