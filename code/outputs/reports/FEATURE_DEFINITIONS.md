# Phase 3A: Comprehensive PPG Feature Definitions & Mathematical Taxonomy

## 1. Overview & Architectural Framework

In accordance with Phase 3A specifications and the Research-Grade Modification Patch, feature extraction operates strictly on **PPG Channel 0** (125 Hz, 1250 samples per 10-second window). No arterial blood pressure (ABP) or electrocardiogram (ECG) signals enter the feature extraction pipeline. Ground-truth systolic blood pressure (SBP) and diastolic blood pressure (DBP) targets are derived strictly from reference invasive ABP beats and are never exposed to feature extraction.

Features are structured across **Three Physiological Representation Branches**:
- **Branch A (Amplitude-Preserving PPG)**: 0.5–8.0 Hz Butterworth 3rd-order zero-phase bandpass filtered signal ($x_{\text{filt}}$). No amplitude normalization is applied, intentionally preserving raw sensor/tissue optical attenuation scale.
- **Branch B (Normalized Morphology PPG)**: Per-window Z-score normalized signal ($x_{\text{norm}} = (x_{\text{filt}} - \mu) / \sigma$). Preserves scale-invariant geometric pulse morphology, timing intervals, and normalized derivatives.
- **Branch C (Combined Representation)**: The concatenation of Branch A and Branch B feature vectors, testing whether absolute scale and shape morphology provide complementary predictive information.

---

## 2. Feature Taxonomy & Classification

Every feature is classified into one of five functional physiological categories:
1. **Scale-Dependent**: Features whose numerical values scale directly with the physical magnitude of the photoplethysmogram (e.g., tissue blood volume change, skin pigmentation, emitter-detector coupling).
2. **Scale-Invariant**: Waveform geometry features invariant to uniform vertical scaling.
3. **Timing-Based**: Temporal intervals and cardiac pulse rate descriptors (independent of vertical scale).
4. **Derivative-Based**: Velocity (VPG) and acceleration (APG) descriptors capturing upstroke rate and inflection waves.
5. **Spectral**: Frequency-domain power distributions and information entropy.

---

## 3. Detailed Feature Dictionary

### Group A: Basic Waveform Statistics

| Feature Name | Branch | Taxonomy Category | Mathematical Definition | Physiological Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `mean_a` | Branch A | Scale-Dependent | $\frac{1}{N} \sum_{i=1}^N x_{\text{filt}}[i]$ | Baseline DC optical offset after 0.5 Hz highpass. Reflects residual low-frequency vascular tone. |
| `std_a` | Branch A | Scale-Dependent | $\sqrt{\frac{1}{N} \sum_{i=1}^N (x_{\text{filt}}[i] - \bar{x})^2}$ | Total AC pulsatile energy. Directly related to pulsatile blood volume displacement. |
| `var_a` | Branch A | Scale-Dependent | $\sigma^2 = \text{Var}(x_{\text{filt}})$ | Variance of the bandpass filtered optical signal. |
| `rms_a` | Branch A | Scale-Dependent | $\sqrt{\frac{1}{N} \sum_{i=1}^N x_{\text{filt}}[i]^2}$ | Quadratic mean of amplitude. Measures effective signal power. |
| `ptp_a` | Branch A | Scale-Dependent | $\max(x_{\text{filt}}) - \min(x_{\text{filt}})$ | Full dynamic range across the 10-second window. |
| `min_a` | Branch A | Scale-Dependent | $\min(x_{\text{filt}})$ | Minimum amplitude trough level. |
| `max_a` | Branch A | Scale-Dependent | $\max(x_{\text{filt}})$ | Peak systolic optical deflection across window. |
| `median_a` | Branch A | Scale-Dependent | $\text{Median}(x_{\text{filt}})$ | Median amplitude; robust central tendency against transient spikes. |
| `iqr_a` | Branch A | Scale-Dependent | $Q_{75}(x_{\text{filt}}) - Q_{25}(x_{\text{filt}})$ | Interquartile range of amplitude; robust measure of dispersion. |
| `p10_a` | Branch A | Scale-Dependent | 10th percentile of $x_{\text{filt}}$ | Near-trough baseline level. |
| `p90_a` | Branch A | Scale-Dependent | 90th percentile of $x_{\text{filt}}$ | Near-peak systolic level. |
| `skew_b` | Branch B | Scale-Invariant | $\frac{\frac{1}{N} \sum (x_{\text{norm}} - \bar{x})^3}{\sigma^3}$ | Asymmetry of the pulse amplitude distribution. Systolic peaks produce positive skewness. |
| `kurt_b` | Branch B | Scale-Invariant | $\frac{\frac{1}{N} \sum (x_{\text{norm}} - \bar{x})^4}{\sigma^4} - 3$ | Peakedness / heavy-tailedness of pulse profile. High values indicate sharp systolic spikes. |

---

### Group B: Pulse Rate & Timing Features

| Feature Name | Branch | Taxonomy Category | Mathematical Definition | Physiological Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `pulse_count` | Branch B | Timing-Based | Count of detected systolic peaks in 10 s | Number of complete cardiac cycles observed in the window. |
| `hr_bpm` | Branch B | Timing-Based | $\frac{60}{\text{mean}(\text{IBI})}$ | Instantaneous heart rate in beats per minute. Highly correlated with sympathetic vascular tone. |
| `ibi_mean` | Branch B | Timing-Based | $\frac{1}{M-1} \sum_{k=1}^{M-1} (t_{k+1}^{\text{peak}} - t_k^{\text{peak}})$ | Average inter-beat interval (seconds). |
| `ibi_median` | Branch B | Timing-Based | $\text{Median}(\Delta t^{\text{peak}})$ | Robust inter-beat interval unaffected by ectopic or missing peak detections. |
| `ibi_std` | Branch B | Timing-Based | $\text{Std}(\Delta t^{\text{peak}})$ | Heart rate variability (SDNN equivalent across 10 s window). Reflects autonomic regulation. |
| `ibi_cv` | Branch B | Timing-Based | $\frac{\text{Std}(\text{IBI})}{\text{Mean}(\text{IBI})}$ | Coefficient of variation of inter-beat intervals. Relative measure of rhythm stability. |

---

### Group C: Pulse Morphology Features

| Feature Name | Branch | Taxonomy Category | Mathematical Definition | Physiological Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `pulse_amp_median_a`| Branch A | Scale-Dependent | $\text{Median}(x_{\text{filt}}[t_k^{\text{peak}}] - x_{\text{filt}}[t_k^{\text{foot}}])$ | Median beat pulsatile amplitude. Directly proportional to stroke volume and arterial distensibility. |
| `pulse_amp_iqr_a` | Branch A | Scale-Dependent | $Q_{75}(\text{Amp}_k) - Q_{25}(\text{Amp}_k)$ | Beat-to-beat variability of pulse amplitude, influenced by respiratory modulation. |
| `pulse_amp_cv_a` | Branch A | Scale-Invariant | $\frac{\text{Std}(\text{Amp}_k)}{\text{Mean}(\text{Amp}_k)}$ | Relative pulse amplitude variation (pulsus paradoxus indicator). |
| `pulse_width_median`| Branch B | Timing-Based | $\text{Median}(T_{\text{rise}} + T_{\text{decay}})$ | Total pulse duration per cardiac cycle (seconds). |
| `rise_time_median` | Branch B | Timing-Based | $\text{Median}(t_k^{\text{peak}} - t_k^{\text{foot}})$ | Systolic upstroke time. Prolonged upstroke indicates arterial stiffening or aortic valve resistance. |
| `decay_time_median`| Branch B | Timing-Based | $\text{Median}(t_{k+1}^{\text{foot}} - t_k^{\text{peak}})$ | Diastolic runoff time. Sensitive to total peripheral resistance (TPR). |
| `max_upstroke_slope_median`| Branch B | Derivative-Based | $\text{Median}\left(\frac{\Delta x_{\text{norm}}}{\Delta t_{\text{rise}}}\right)$ | Maximum rate of pressure increase during early systolic ventricular ejection. |
| `max_downstroke_slope_median`| Branch B| Derivative-Based | $\text{Median}\left(\frac{\Delta x_{\text{norm}}}{\Delta t_{\text{decay}}}\right)$ | Maximum rate of diastolic runoff during arterial recoil. |
| `pulse_area_median`| Branch B | Scale-Invariant | $\text{Median}\left(\int_{t_{\text{foot}}}^{t_{\text{next\_foot}}} x_{\text{norm}}(t) dt\right)$ | Total normalized pulse area. Integral of normalized volumetric expansion over cardiac cycle. |

---

### Group D: First Derivative (Velocity Photoplethysmogram — VPG)

The first derivative is computed via central finite differences: $v[n] = \frac{x_{\text{norm}}[n+1] - x_{\text{norm}}[n-1]}{2 \cdot \Delta t}$, where $\Delta t = \frac{1}{125} = 0.008\text{ s}$.

| Feature Name | Branch | Taxonomy Category | Mathematical Definition | Physiological Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `vpg_max` | Branch B | Derivative-Based | $\max(v[n])$ | Peak systolic blood flow velocity in microcirculation. |
| `vpg_min` | Branch B | Derivative-Based | $\min(v[n])$ | Peak diastolic deceleration rate. |
| `vpg_std` | Branch B | Derivative-Based | $\text{Std}(v[n])$ | Overall dispersion of pulse wave velocity slopes. |
| `vpg_rms` | Branch B | Derivative-Based | $\sqrt{\frac{1}{N}\sum v[n]^2}$ | RMS power of velocity waveform. |
| `vpg_max_upstroke` | Branch B | Derivative-Based | $Q_{95}(v[n])$ | 95th percentile of velocity, capturing maximum sustained ventricular ejection velocity. |

---

### Group E: Second Derivative (Acceleration Photoplethysmogram — APG)

The second derivative is computed via central finite differences of VPG: $a[n] = \frac{v[n+1] - v[n-1]}{2 \cdot \Delta t}$.
In clinical physiology (Takazawa et al.), the APG contains characteristic waves:
- $a$-wave: Early systolic positive acceleration.
- $b$-wave: Early systolic deceleration.
- $c$-wave: Late systolic re-acceleration.
- $d$-wave: Late systolic deceleration.
- $e$-wave: Early diastolic dicrotic notch.

| Feature Name | Branch | Taxonomy Category | Mathematical Definition | Physiological Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `apg_max` | Branch B | Derivative-Based | $\max(a[n])$ | Corresponds to the $a$-wave (early systolic acceleration). |
| `apg_min` | Branch B | Derivative-Based | $\min(a[n])$ | Corresponds to the $b$-wave (early systolic deceleration). |
| `apg_std` | Branch B | Derivative-Based | $\text{Std}(a[n])$ | Variation of acceleration across cardiac cycles. |
| `apg_b_to_a_ratio` | Branch B | Scale-Invariant | $\frac{\vert b \vert}{\vert a \vert}$ | Standard vascular aging index. Increases markedly with arterial stiffness and hypertension. |

---

### Group F: Spectral Properties

Power spectral density (PSD) $S_{xx}(f)$ is estimated via Welch's periodogram (Hanning window, segment length 256 samples = 2.05 s, 50% overlap).

| Feature Name | Branch | Taxonomy Category | Mathematical Definition | Physiological Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `dominant_freq` | Branch B | Spectral | $\arg\max_f S_{xx}(f)$ | Fundamental pulse repetition frequency (Hz), corresponding directly to heart rate. |
| `pulse_band_power` | Branch B | Spectral | $\frac{\int_{0.5}^{3.5} S_{xx}(f) df}{\int_{0.5}^{Nyq} S_{xx}(f) df}$ | Relative concentration of energy within the physiological heart rate band [0.5–3.5 Hz]. |
| `spectral_entropy`| Branch B | Spectral | $-\sum_{k} p_k \ln(p_k + 10^{-12})$ | Flatness / complexity of the frequency spectrum. Low entropy = pure periodic pulse; high = noise/fibrillation. |

---

## 4. Leakage-Safe Imputation and Scaling Rules

1. **Missing Feature Values (NaN/Inf)**:
   - Waveform features where peak detection fails (e.g., severe flatline or movement artifact) produce `np.nan`.
   - The imputation model is `sklearn.impute.SimpleImputer(strategy='median')`.
   - **CRITICAL**: The imputer is fitted strictly on the `train` partition:
     $$\text{median}_{\text{train}} = \text{median}(X_{\text{train}})$$
     Validation and test partitions are transformed using $\text{median}_{\text{train}}$. No validation or test statistics are ever computed during imputation fitting.
2. **Feature Standardization**:
   - The scaler is `sklearn.preprocessing.StandardScaler()`.
   - Fitted strictly on the imputed `train` partition:
     $$\mu_{\text{train}} = \text{mean}(X_{\text{train}}), \quad \sigma_{\text{train}} = \text{std}(X_{\text{train}})$$
     Validation and test partitions are standardized using $(\mu_{\text{train}}, \sigma_{\text{train}})$.
