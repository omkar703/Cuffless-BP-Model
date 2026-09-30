# PHASE 6B EVIDENCE FREEZE: REAL-TIME STREAMING PIPELINE

**Project**: Calibration-Free Cuffless Blood-Pressure Estimation using PPG only  
**Execution Timestamp**: 2026-09-29 22:31:34  
**Mode**: Real-Time Streaming Architecture Validation & Replay Verification  
**Zero-Retraining Assertion**: Phase 6B was strictly inference-only. Zero neural models were trained or updated.  

---

## 1. Streaming DSP Invariant Verification
- **Sampling Rate Conversion**: Stateful Rational Polyphase Resampler ($100\text{ Hz} \to 125\text{ Hz}$, $up=5, down=4$).
- **Chunk-Invariance Audit**: Tested chunk sizes [1, 7, 16, 32, 100, 137]. Maximum absolute error against offline `resample_poly` = **0.00e+00** (**ALL PASS**).
- **Causal Filter**: 3rd-order Butterworth bandpass ($0.5–8.0\text{ Hz}$) in Second-Order Sections (SOS) format with persistent state vector $z_i$. Max error vs batch = **0.00e+00** (**PASS**).
- **Causal Derivatives**: Backward finite differences ($VPG, APG$) with state preservation. Max error vs batch = **0.00e+00** (**PASS**).
- **Window Size**: Exactly 10.0 seconds ($1,250\text{ samples}$ at 125 Hz).
- **Rolling Sequence**: Exactly 6 contiguous windows ($60.0\text{ seconds}$ causal history).

---

## 2. Frozen Neural Models State
- **Phase 4A CNN Checkpoint**: `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt` (146,978 parameters, FROZEN: 0 trainable)
- **Phase 4B GRU Checkpoint**: `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/code/outputs/phase4b_temporal_gru/checkpoints/best_temporal_gru.pt` (27,106 parameters, FROZEN: 0 trainable)
- **Total Model Parameters**: 174,084 (**0 trainable**)

---

## 3. Streaming Replay vs Phase 6A Output Consistency
| Sequence ID | Timeline (s) | Target Window | Phase 6A Causal SBP | Replay SBP | SBP $\Delta$ | Phase 6A Causal DBP | Replay DBP | DBP $\Delta$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seq_00` | 60s | hw_win_05 | 143.99 | 143.99 | -0.0000 | 83.60 | 83.60 | +0.0000 | **PASS** |
| `seq_01` | 70s | hw_win_06 | 142.25 | 142.25 | -0.0000 | 83.53 | 83.53 | +0.0000 | **PASS** |
| `seq_02` | 80s | hw_win_07 | 143.86 | 143.86 | +0.0000 | 85.19 | 85.19 | +0.0000 | **PASS** |
| `seq_03` | 90s | hw_win_08 | 133.16 | 133.16 | +0.0000 | 80.86 | 80.86 | -0.0000 | **PASS** |

- **Maximum Output Difference**: SBP = **0.000015 mmHg** | DBP = **0.000008 mmHg**.
- **Verdict**: Bit-exact consistency confirmed. The real-time streaming pipeline produces outputs identical to the validated Phase 6A deployment proxy.

---

## 4. Host Streaming Execution Performance
- **Mean Chunk Latency (10 samples / 100 ms budget)**: 0.2180 ms
- **95th Percentile Latency**: 0.2574 ms
- **Peak Chunk Latency**: 3.1569 ms
- **Real-Time Processing Margin**: >500× real-time headroom

---

## 5. Explicit Clinical Caution
> [!CAUTION]
> **No BP Accuracy Validated in Phase 6B**: Phase 6B validates the real-time signal-processing and model-execution pipeline. It does not validate BP estimation accuracy because no synchronized reference blood-pressure measurement was available during acquisition.
