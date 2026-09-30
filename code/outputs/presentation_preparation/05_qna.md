# 05. Comprehensive Defense Q&A Preparation

**Target Examination**: Final Year Capstone Project Defense  
**Format**: Spoken response duration: 20 to 60 seconds per question.

---

### Q1: Why did you choose Photoplethysmography (PPG) instead of Electrocardiogram (ECG)?
**Spoken Answer (30s)**:
> "Most published literature relies on Pulse Transit Time (PTT), which requires both an ECG R-wave from the chest and a peripheral PPG arrival. However, wearing chest electrodes or conductive adhesive patches 24/7 is clinically impractical and severely limits patient compliance. PPG measures cutaneous blood volume pulsatility using simple optical LEDs and photodetectors, which are already ubiquitous in smartwatches and consumer wearables. Our goal was to investigate how much cardiovascular hemodynamic information can be extracted from a truly unobtrusive, single-site optical sensor alone."

---

### Q2: Why did you explicitly compute the Velocity (VPG) and Acceleration (APG) Plethysmograms?
**Spoken Answer (40s)**:
> "Raw PPG records blood volume, but arterial stiffness and pressure dynamics are fundamentally governed by rates of change. The first derivative, VPG ($\frac{dx}{dt}$), represents the velocity of blood flow and the rate of ventricular pressure generation ($dP/dt$). The second derivative, APG ($\frac{d^2x}{dt^2}$), exposes distinctive inflection points—the a-, b-, c-, d-, and e-waves—which reflect early systolic acceleration, peak contraction, and peripheral wave reflection at the dicrotic notch. Our Phase 3A ablation study proved that providing explicit VPG and APG channels monotonically reduces estimation error by approximately 1 mmHg without increasing neural network complexity."

---

### Q3: Why does the neural network accept 125 Hz input if your MAX30102 hardware samples at 100 Hz?
**Spoken Answer (45s)**:
> "Our deep neural networks were trained on the benchmark MIMIC-II database, which was digitized at a standard clinical frequency of 125 Hz. Physical optical sensors like the MAX30102, however, are clocked internally by hardware timers that operate stably at 100 Hz. Rather than retraining our models on downsampled data, we engineered a streaming causal polyphase rational resampler with a rational ratio of 5 to 4. Crucially, our polyphase implementation maintains its internal FIR filter state across consecutive 10-sample chunks, preventing filter ring-up and ensuring zero future information leakage."

---

### Q4: Why did you select a 60-second temporal context window instead of predicting from a single pulse?
**Spoken Answer (45s)**:
> "A single pulse or 10-second window captures only an isolated snapshot. However, human blood pressure is dynamically modulated by autonomic tone, baroreceptor feedback, and respiration over 30 to 60-second cycles. In Phase 3B, we conducted controlled experiments evaluating context lengths from 10 to 60 seconds. We discovered that a 60-second causal context (6 consecutive 10-second windows) achieved the lowest validation error, reducing SBP MAE by 0.58 mmHg. Furthermore, a temporal GRU acts as a physiological low-pass filter, smoothing out transient motion artifacts that corrupt individual beats."

---

### Q5: Why is your system strictly "calibration-free"?
**Spoken Answer (40s)**:
> "Many commercial cuffless devices claim to be cuffless, but require frequent user calibration—such as entering a cuff measurement every two weeks. In machine learning, a calibration offset often allows a weak model to simply 'memorize' the patient's mean blood pressure rather than learning true physiological hemodynamics. We deliberately constrained our project to calibration-free estimation to assess the true generalization capability of optical morphology across completely unseen subjects."

---

### Q6: Why did you not include an Inertial Measurement Unit (IMU) / accelerometer?
**Spoken Answer (35s)**:
> "Our primary research hypothesis was strictly physiological: Can arterial blood pressure be inferred from pure optical microvascular hemodynamics? Adding an accelerometer would measure gross physical activity, but it does not measure arterial blood pressure. Furthermore, in clinical resting protocols, subjects remain seated still. Introducing an IMU would confound our core research question regarding the information content of optical plethysmography."

---

### Q7: Why did you train a single multi-output model instead of separate neural networks for SBP and DBP?
**Spoken Answer (35s)**:
> "Systolic and diastolic blood pressure are not physiologically independent; they are coupled by stroke volume, total peripheral resistance, and arterial compliance within the same cardiac cycle. Training a shared 1D CNN backbone to predict both SBP and DBP forces the hidden representations to capture joint vascular mechanics. Additionally, sharing the convolutional backbone halves the computational footprint, requiring only 146,978 parameters instead of nearly 300,000 for two separate networks."

---

### Q8: What is the single biggest limitation of this project?
**Spoken Answer (45s)**:
> "The primary limitation is the sample size and demographic range of our physical reference-cuff pilot. We tested 7 paired events across 4 young, healthy volunteers with resting blood pressures around 120/80 mmHg. While this proved that our end-to-end hardware-to-model streaming pipeline operates flawlessly, it cannot prove population-level clinical validity. Establishing true clinical accuracy requires a formal trial under ISO 81060-2 with at least 85 subjects covering hypotensive, normotensive, and severely hypertensive populations."

---

### Q9: Why was Systolic BP (SBP) estimation error higher than Diastolic BP (DBP) in your physical pilot?
**Spoken Answer (45s)**:
> "This is a well-documented phenomenon in cuffless physiological literature. Diastolic pressure is largely governed by steady peripheral vascular resistance and microvascular runoff during cardiac rest, which directly shapes the exponential diastolic decay contour of the PPG. In contrast, systolic pressure depends heavily on central aortic stiffness, pulse wave velocity, and left-ventricular contractility. In a calibration-free setting without patient height, age, or baseline arterial stiffness, central aortic pressure generation cannot be directly observed from peripheral finger PPG alone, resulting in a static systolic offset."

---

### Q10: What is the practical value of Conformal Prediction in this project?
**Spoken Answer (45s)**:
> "Deep neural networks are typically point estimators; they output a single number without indicating how confident they are. In clinical monitoring, a bare number is dangerous. Conformal prediction provides mathematically guaranteed prediction intervals that achieve a predefined coverage level (such as 90% or 95%) without retraining the model. In Phase 5C, we introduced extreme-aware asymmetric conformal intervals that widen automatically in hypertensive regimes where model uncertainty is highest, providing clinicians with rigorous bounds rather than misleading point estimates."

---

### Q11: Why did your very first physical hardware recording fail in Phase 6C?
**Spoken Answer (40s)**:
> "In our initial hardware session, all reference-BP pairs were rejected by the quality gate. Our diagnostic audit revealed that the subject completed their cuff measurement and prematurely lifted their finger from the sensor while the host was still recording. Because our causal GRU requires 60 continuous seconds of high-quality data (6 consecutive valid windows), lifting the finger triggered the flatline and clipping rejection gate, correctly excluding the corrupted predictions."

---

### Q12: How did you diagnose and resolve that hardware failure?
**Spoken Answer (40s)**:
> "We performed an automated window-by-window signal audit that tracked raw amplitude peak-to-peak, baseline drift, and zero fractions. The audit conclusively proved that the windows formed after the cuff measurement had near-zero signal power and zero detected peaks, characteristic of optical sensor detachment. We resolved this by updating our physical acquisition protocol to require subjects to maintain continuous finger contact for at least 60 seconds before and after cuff inflation. In our next 5 sessions, 100% of reference measurements were successfully included."

---

### Q13: How did you prevent data leakage during offline training?
**Spoken Answer (40s)**:
> "Data leakage is the most prevalent flaw in published cuffless BP papers, where random segment splitting puts adjacent 10-second windows from the same ICU patient into both training and testing sets. We strictly enforced record-level partitioning: all 12,000 MIMIC-II records were assigned exclusively to Train (70%), Validation (15%), or Test (15%). We verified through set intersection that zero records, windows, or patient data overlap across partitions ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$)."

---

### Q14: Why can you NOT claim that your device is "clinically validated"?
**Spoken Answer (40s)**:
> "International clinical validation standards, such as ANSI/AAMI/ISO 81060-2 and IEEE 1708, enforce stringent protocol requirements: a minimum of 85 human subjects with specific age distributions, gender balance, arm circumferences, and broad blood pressure ranges (including severe hypertension and hypotension), with each measurement verified by two simultaneous auscultatory observers. Our physical pilot tested 4 healthy individuals ($N=7$ pairs). Claiming clinical validation based on an exploratory academic pilot would be scientifically dishonest."

---

### Q15: How would you improve the system in a future version?
**Spoken Answer (45s)**:
> "First, we would incorporate single-point zero-shot demographic calibration (such as age, sex, and arm circumference) or an initial resting baseline cuff entry to eliminate the static SBP bias. Second, we would implement adaptive motion artifact cancellation using the dual-wavelength Red and Infrared channels. Third, we would quantize the neural network to INT8 precision using TensorFlow Lite Micro or ESP-DL, allowing the entire model to run natively on the ESP32 microcontroller without requiring a host PC."

---

### Q16: What is the immediate next experiment you would perform?
**Spoken Answer (35s)**:
> "The immediate next experiment would be a physiological provocation study on physical hardware—such as a cold pressor test or mild isometric handgrip exercise. This would induce dynamic blood pressure swings within the same individual, allowing us to evaluate whether our frozen causal GRU accurately tracks transient pressure changes rather than just static resting baselines."

---

### Q17: Why did the model predict above the 120 mmHg cuff reference across multiple subjects?
**Spoken Answer (40s)**:
> "In Phase 4B and Phase 5, our models were trained on ICU patients from MIMIC-II, where average arterial stiffness and vascular tone are systematically elevated compared to young college students. Because our model is calibration-free and has no knowledge of the subject's young age or healthy arterial compliance, it applies the generalized ICU mapping learned from training data, resulting in a systemic upward bias of approximately +17 mmHg."

---

### Q18: What is scientifically novel about your project compared to existing work?
**Spoken Answer (45s)**:
> "Our novelty lies in four architectural contributions: First, we strictly eliminated ECG leads, relying entirely on a single optical site. Second, we demonstrated that raw PPG, VPG, and APG channels act complementarily inside a 1D CNN without requiring handcrafted peak detection. Third, we introduced a 60-second causal temporal GRU that models short-term cardiovascular memory. Fourth, we integrated post-hoc extreme-aware conformal prediction, providing mathematically sound uncertainty intervals for cuffless blood pressure estimation."

---

### Q19: What happens to your pipeline if the subject moves their finger?
**Spoken Answer (40s)**:
> "Optical plethysmography is inherently sensitive to motion artifacts. In our streaming pipeline, we implemented a real-time Quality Control (QC) gate that evaluates every 10-second window for amplitude clipping, zero fractions, baseline wander, and unphysiological peak counts. If a finger movement occurs, the affected window is automatically flagged as WARN or REJECT. In Phase 6C, any sequence containing a REJECT window is excluded from clinical comparison, ensuring that corrupted signals never produce erroneous blood pressure readings."

---

### Q20: Can your deep learning model run directly on the ESP32 microcontroller?
**Spoken Answer (40s)**:
> "Currently, the ESP32 handles high-speed 100 Hz optical acquisition, hardware buffering, and serial streaming, while inference executes on the host workstation. However, our total architecture has only 174,084 parameters, which occupies approximately 700 KB in 32-bit floating point. With INT8 quantization, the footprint shrinks to under 200 KB and requires roughly 35 MFLOPS. The modern ESP32-S3 chip features vector instructions and 512 KB of SRAM, making native on-device embedded inference entirely feasible as a future optimization."
