# Phase 3B Research Decision & Next-Phase Roadmap

**Decision Code**: CASE C  
**Classification**: **CASE C — Multi-Scale Spatio-Temporal PPG Model Justified**  
**Evidence Basis**: Rigorous, calibration-free validation on 261,339 MIMIC-II windows  

---

## 1. Observed Evidence

1. **Derivative Signal Value**:
   - Explicitly providing VPG and APG features reduces validation combined MAE by **0.346 mmHg** compared to basic PPG morphology alone.
   - Permutation importance on the Context-5 model confirms that temporal history statistics of derivative features dominate the ranking: the top contributors are `spectral_entropy_hist_min`, `apg_max_hist_min`, `max_downstroke_slope_median_hist_min`, and `vpg_std_hist_min` — all temporal aggregations (rolling minimum/maximum) of APG and VPG channels, confirming that derivative channels carry predictive signal even within the temporal history.

2. **Temporal Context Value**:
   - Supplying 20–60 seconds of causal historical context reduces validation combined MAE by **0.583 mmHg** relative to the static 10-second window.
   - The optimal classical context length is **60 seconds** (Context-5).

3. **Complementary Interaction**:
   - Adding temporal aggregation to an already derivative-rich model yields an additional **0.583 mmHg** reduction (Experiment D: D2 vs D4), demonstrating that temporal dynamics capture state changes orthogonal to instantaneous wave shape.

4. **Extreme Blood Pressure Offset**:
   - While temporal context modestly reduces extreme tail bias (by ~2.77 mmHg), classical models trained under MSE loss still exhibit significant regression-to-the-mean at SBP $<90$ and $\ge 160$ mmHg.

---

## 2. Strongest Result

The strongest finding is that **multi-channel derivative representation and temporal context provide complementary, non-redundant gains**. Instantaneous derivatives resolve local pulse reflections (arterial stiffness), while temporal sequences resolve slow hemodynamic trends (autonomic tone and respiratory modulation).

---

## 3. Negative Results & Honest Limitations

1. **Classical Temporal Features Cannot Eliminate Regression-to-the-Mean**:
   - Handcrafted rolling statistics (mean, std, min, max, slope) do not resolve the fundamental ambiguity between normotensive baseline and hypertensive vascular stiffening.
2. **Missing History Attrition**:
   - At 60 seconds (Context-5), 18.5% of windows must be discarded due to lack of preceding history within the record.
3. **No Claim of Clinical Superiority**:
   - Although test MAE reaches **13.44 mmHg SBP / 6.77 mmHg DBP**, the standard deviation of error remains above the AAMI standard ($< 8$ mmHg).

---

## 4. Recommended Next Neural Architecture

Based strictly on this empirical evidence, the recommended architecture for Phase 3B Deep Learning is a:

### **Multi-Scale Spatio-Temporal PPG Network with Explicit Derivative Channels**

#### Architecture Specification:
1. **Multi-Channel Input Tensor**:
   - Shape: $(B, 3, L)$ where channels are $[x(t), x'(t), x''(t)]$ (PPG, VPG, APG) computed via finite-difference or learned differentiable filter kernels.
2. **Local Morphology Encoder (Spatio-Temporal CNN / TCN)**:
   - Dilated 1D convolutions with residual connections to capture high-frequency notch reflections and upstroke velocities across single pulse cycles (0.2–1.5s).
3. **Global Sequence Modulator (Bi-directional / Causal GRU or Lightweight Transformer)**:
   - Aggregates multi-window pulse embeddings across consecutive 10-second segments (20–60s context) to track autonomic trend vectors.
4. **Extreme-Value Aware Loss Function**:
   - Weighted Huber Loss or Focal Regression penalty inversely proportional to target density to penalize hypertension underestimation.

---

## 5. Why That Architecture is Justified

- **Why Multi-Channel?**: Experiment C proved derivatives contain critical predictive signal. A 3-channel input tensor allows CNN kernels to directly model phase relationships between pulse velocity and acceleration.
- **Why Temporal GRU/Transformer?**: Experiment B proved that 20–60s of history carries distinct predictive signal that improves both linear and non-linear classical models. A neural sequence model can learn continuous dynamical embeddings superior to discrete summary statistics.
- **Why Extreme-Value Loss?**: Experiments confirmed that standard MSE loss drives pathological regression-to-the-mean, necessitating loss reweighting.

---

## 6. What Experiment Should Come Next

1. **Step 1**: Implement multi-channel dataset generator $[PPG, VPG, APG]$ from raw 125 Hz waveforms.
2. **Step 2**: Train lightweight 1D CNN baseline on single 10-second windows using GTX 1650 Ti to verify raw waveform representation matches or exceeds handcrafted features.
3. **Step 3**: Introduce temporal sequence recurrence (1D CNN + GRU) over 20–60s sequences.
4. **Step 4**: Introduce range-weighted focal loss to target the >= 160 mmHg error mode.
