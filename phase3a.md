# PHASE 3A — PPG-ONLY CLASSICAL BASELINE, EVALUATION & ERROR ANALYSIS

## Project

Cuffless Blood Pressure Estimation from PPG using MAX30102 + ESP32

Primary research objective:

Establish a rigorous, calibration-free, PPG-only classical machine-learning baseline on the Phase 2 dataset.

The purpose of this phase is NOT to find the final model.

The purpose is to determine:

1. How much blood-pressure information is present in PPG alone.
2. Whether handcrafted PPG morphology features provide useful predictive power.
3. Whether nonlinear models substantially outperform simple baselines.
4. Where the models fail.
5. Whether the failures reveal a promising research direction for a later novel model.

STOP after Phase 3A.

Do NOT implement CNN, LSTM, GRU, Transformer, attention, or hybrid neural architectures.

---

# 1. FROZEN DATASET STATE

Project root:

/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/

Raw dataset:

BloodPressureDataset/

Contains:

part_1.mat
part_2.mat
...
part_12.mat

Raw dataset size:

approximately 4.58 GB

Sampling frequency:

125 Hz

---

## Phase 1

Phase 1 established:

- raw dataset validation
- PPG QC
- ABP QC
- beat-wise ABP target extraction
- PPG filtering reference
- dataset statistics
- persistent record IDs

Phase 1 must NOT be reimplemented unless required by an actual dependency.

---

## Phase 2

Phase 2 established:

- 10-second windows
- 1250 samples/window
- 0% overlap
- window-level PPG QC
- window-level ABP QC
- beat-wise SBP/DBP/MAP targets
- persistent window IDs
- record-level train/validation/test assignment
- leakage assertions

Primary manifest:

code/outputs/windows/window_manifest.csv

Record split:

code/outputs/splits/record_split.csv

Phase 2 report:

code/outputs/statistics/PHASE2_WINDOW_REPORT.md

---

# 2. FROZEN PARTITION

The existing Phase 2 partition must be reused exactly.

Do NOT create a new random split.

Do NOT reshuffle records.

Do NOT split windows randomly.

Do NOT optimize the split according to BP performance.

Partition is based on unique record_id with seed 42.

Expected approximate counts:

TRAIN:
8,400 records
~183,517 eligible windows

VALIDATION:
1,800 records
~39,461 eligible windows

TEST:
1,800 records
~38,361 eligible windows

Read the actual values from the manifest.

Before doing anything else, verify:

intersection(train_record_ids, validation_record_ids) == 0

intersection(train_record_ids, test_record_ids) == 0

intersection(validation_record_ids, test_record_ids) == 0

If any overlap exists:

STOP.

---

# 3. PRIMARY SUPERVISED DATASET

Use ONLY:

modeling_eligible == True

Therefore the primary supervised dataset is:

PPG_VALID_ABP_VALID

Do not use:

PPG_VALID_ABP_INVALID
PPG_INVALID_ABP_VALID
PPG_INVALID_ABP_INVALID

for supervised BP regression.

Preserve those windows in the manifest for future research.

---

# 4. MODEL INPUT RULE

The model input must contain PPG-derived information ONLY.

Allowed:

- filtered PPG
- PPG morphology
- PPG pulse timing
- PPG derivatives
- PPG spectral statistics
- PPG amplitude statistics

Forbidden:

- ECG
- ABP
- SBP as an input feature
- DBP as an input feature
- MAP as an input feature
- PTT
- PAT
- ECG-derived timing
- subject-specific calibration BP
- record-level true BP baseline
- any target-derived feature

Add an automated forbidden-feature check.

Forbidden terms:

abp
ecg
sbp
dbp
map
ptt
pat
bp_calibration
calibration

If any forbidden feature enters the model matrix:

STOP.

---

# 5. MODELING OBJECTIVES

Primary targets:

1. SBP
2. DBP

MAP can be calculated and analyzed, but do not make MAP the primary prediction target in Phase 3A.

Use separate regressors for:

SBP

DBP

Do not build multi-output models yet.

This keeps the first baseline easy to interpret.

---

# 6. CODE STRUCTURE

Create:

code/modeling/

with:

code/modeling/__init__.py
code/modeling/data_loader.py
code/modeling/preprocessing_pipeline.py
code/modeling/features.py
code/modeling/classical_models.py
code/modeling/evaluation.py
code/modeling/error_analysis.py

Create:

code/visualization/model_plots.py

Create:

code/scripts/run_classical_baselines.py

Create:

code/notebooks/03A_ppg_classical_baselines.ipynb

Create outputs:

code/outputs/features/
code/outputs/models/
code/outputs/metrics/
code/outputs/predictions/
code/outputs/figures/
code/outputs/reports/

---

# 7. DATA LOADER

Implement:

code/modeling/data_loader.py

The loader must reconstruct each 10-second PPG window from:

window_manifest.csv

and the original MAT file.

For each manifest row:

1. determine part file
2. determine record index
3. load required record
4. slice start_sample:end_sample
5. extract ONLY Channel 0 = PPG
6. verify exactly 1250 samples
7. return signal and metadata

Return:

ppg
sbp
dbp
map
record_id
window_id
split

Do NOT return ECG as an input feature.

Do NOT load all 261k raw windows into RAM.

Process incrementally or in batches.

---

# 8. SIGNAL PREPROCESSING

Use the established Phase 1/2 offline research reference:

Sampling frequency:

125 Hz

Bandpass:

0.5–8.0 Hz

Butterworth:

order 3

Filtering:

scipy.signal.filtfilt

Explicitly label:

OFFLINE RESEARCH REFERENCE FILTER

Do NOT claim that this is the final ESP32 filter.

Do NOT implement causal filtering in Phase 3A.

Hardware-domain compatibility will be addressed later.

---

# 9. THREE INPUT REPRESENTATIONS

Implement these candidate preprocessing representations:

## Representation A — FILTERED PPG

Bandpassed PPG only.

No amplitude normalization.

## Representation B — PER-WINDOW Z-SCORE

For every window independently:

z = (x - mean(x)) / std(x)

## Representation C — PER-WINDOW ROBUST NORMALIZATION

For every window independently:

x_norm = (x - median(x)) / IQR(x)

Important:

These per-window transformations do not require a global fit.

Do not calculate normalization statistics over the complete dataset.

Do not allow test data to influence the choice of representation.

---

# 10. FEATURE ENGINEERING

Create:

code/modeling/features.py

The first baseline must use a compact, interpretable PPG feature set.

Target:

approximately 20–40 features.

Do NOT generate hundreds of automatically constructed features.

Every feature must have:

- name
- mathematical definition
- implementation
- physiological motivation
- missing-value behavior

---

# 11. FEATURE GROUP A — BASIC WAVEFORM FEATURES

Calculate:

- mean
- median
- standard deviation
- variance
- RMS
- peak-to-peak amplitude
- minimum
- maximum
- IQR
- skewness
- kurtosis

These features are purely diagnostic/baseline features.

---

# 12. FEATURE GROUP B — PULSE RATE FEATURES

Detect PPG pulse peaks using a documented and reproducible method.

Calculate:

- pulse count
- estimated heart rate
- mean pulse interval
- median pulse interval
- pulse interval standard deviation
- coefficient of variation of pulse intervals

Do not use ECG.

HR is PPG-derived only.

Make the peak-detection parameters configurable.

Do not tune them on the test set.

---

# 13. FEATURE GROUP C — PULSE MORPHOLOGY

For each detected pulse, calculate robust statistics across pulses for:

- pulse amplitude
- pulse width
- systolic/rise time
- decay time
- maximum upstroke slope
- maximum downstroke slope
- pulse area

Then aggregate using robust statistics such as:

median
mean
standard deviation
IQR

Do not retain unstable individual pulse measurements as separate features.

---

# 14. FEATURE GROUP D — FIRST DERIVATIVE

Calculate:

VPG = dPPG/dt

Use appropriate numerical differentiation.

Extract:

- maximum VPG
- minimum VPG
- VPG RMS
- maximum upstroke slope
- median maximum upstroke slope
- derivative variability

Make sure the differentiation is numerically stable.

---

# 15. FEATURE GROUP E — SECOND DERIVATIVE

Calculate:

APG = d²PPG/dt²

Explore conventional second-derivative morphology only when reliably detectable.

Potential descriptors:

a
b
c
d
e

and ratios such as:

b/a
c/a
d/a
e/a

IMPORTANT:

Do not force APG morphology onto every signal.

If the waveform does not support reliable detection:

return NaN / missing indicator

and handle it consistently.

Document the detection rule.

---

# 16. FEATURE GROUP F — OPTIONAL SPECTRAL FEATURES

Calculate a very small number of signal-level frequency-domain diagnostics, for example:

- dominant cardiac frequency
- power in physiologically relevant pulse-frequency band
- spectral entropy

Do not generate a huge FFT feature vector.

The purpose is interpretability.

---

# 17. FEATURE INTEGRITY CHECK

Before training any model:

Verify that every feature originates from PPG only.

Print:

total_features

feature_names

feature_groups

missingness percentage

mean
std
min
max

for every feature.

Generate:

code/outputs/features/feature_summary.csv

Also generate:

code/outputs/reports/FEATURE_DEFINITIONS.md

with the mathematical and physiological description of every feature.

---

# 18. MISSING FEATURE HANDLING

Some morphology features may fail.

Do NOT simply replace all missing values with zero without justification.

Evaluate:

1. feature missingness
2. cause of missingness

If the feature is rarely missing:

use a training-set-derived imputation method.

If a feature is frequently missing:

consider excluding it from the primary baseline and document why.

Any learned imputation must be fitted ONLY on TRAIN.

Never fit an imputer on the entire dataset.

---

# 19. LEAKAGE-SAFE PREPROCESSING

If a transformation learns parameters from data, fit it ONLY on TRAIN.

Examples:

- StandardScaler
- RobustScaler
- imputer
- learned feature selection
- PCA

Training:

fit → TRAIN

Validation:

transform using TRAIN parameters

Test:

transform using TRAIN parameters

Never:

fit on train + validation + test.

Never:

fit separately on test.

---

# 20. BASELINE MODELS

Implement exactly these initial models:

## MODEL 0 — Dummy Mean

sklearn DummyRegressor(strategy="mean")

This is essential.

## MODEL 1 — Linear Regression

sklearn LinearRegression

## MODEL 2 — Ridge Regression

sklearn Ridge

Use a small predefined alpha search using VALIDATION only.

Do not use test performance for selecting alpha.

## MODEL 3 — Random Forest

sklearn RandomForestRegressor

Use conservative computational settings.

Set:

random_state = 42

## MODEL 4 — Histogram Gradient Boosting

Use:

sklearn.ensemble.HistGradientBoostingRegressor

Prefer this over adding XGBoost unless XGBoost is already installed.

Do not unnecessarily modify the environment.

---

# 21. COMPUTATIONAL REQUIREMENT

The dataset contains approximately 261k eligible windows.

Do not repeatedly reconstruct raw MAT records for every experiment.

Create an efficient deterministic feature cache.

Recommended:

Parquet

or:

HDF5

or:

NumPy + metadata

The cache must retain:

record_id
window_id
split
sbp
dbp
map

and all PPG-only features.

Because the features are deterministic functions of each individual PPG window, generating the cache once is acceptable.

But any learned preprocessing must remain split-specific.

---

# 22. SAMPLE RUN FIRST

Before the full dataset:

run:

python code/scripts/run_classical_baselines.py --sample-run

Use a deterministic small subset.

Recommended:

1,000 windows

Ensure the sample contains examples from:

train
validation
test

but preserves record-level separation.

The sample run must verify:

- MAT loading
- PPG extraction
- filtering
- normalization
- peak detection
- feature extraction
- missing feature handling
- model fitting
- metric calculation
- prediction generation

If sample run fails:

STOP.

Do not run full dataset.

---

# 23. FULL RUN

Only after successful sample run:

python code/scripts/run_classical_baselines.py --full-run

Process:

- all eligible train windows
- all eligible validation windows
- all eligible test windows

Do not subsample the final experiment.

---

# 24. TRAINING PROTOCOL

Train models only on TRAIN.

Use VALIDATION for:

- comparing models
- selecting normalization
- selecting Ridge alpha
- selecting the primary feature representation

Do not inspect test performance while choosing the model.

Once the best configuration is chosen:

FREEZE IT.

Then evaluate exactly once on TEST.

---

# 25. TEST SET POLICY

The test set is a final holdout.

Do not:

- choose model based on test MAE
- choose normalization based on test MAE
- tune hyperparameters based on test
- remove difficult test records
- alter feature definitions based on test
- rerun experiments until test performance improves

If an experiment is changed after seeing test results:

label it as a new experiment and do not treat the previous test result as final.

---

# 26. PRIMARY METRICS

For SBP and DBP calculate:

### Required

MAE

RMSE

R²

Mean Error / Bias

Error Standard Deviation

Pearson correlation

Spearman correlation

### Supplementary

MAPE

Clearly state:

MAPE is supplementary because percentage error is not necessarily the most appropriate blood-pressure metric.

---

# 27. CLINICALLY RELEVANT DESCRIPTIVE METRICS

Calculate:

bias

standard deviation of error

and optionally report the number of predictions satisfying:

|error| <= 5 mmHg

|error| <= 10 mmHg

|error| <= 15 mmHg

These are descriptive only.

Do NOT call this clinical validation.

Do NOT claim AAMI/ISO compliance.

Do NOT claim medical-device accuracy.

---

# 28. PREDICTION TABLE

Create:

code/outputs/predictions/validation_predictions.csv

and after configuration is frozen:

code/outputs/predictions/test_predictions.csv

Required columns:

record_id
window_id
split

sbp_true
sbp_pred
sbp_error
sbp_abs_error

dbp_true
dbp_pred
dbp_error
dbp_abs_error

map_true

estimated_hr

plus useful PPG quality diagnostics.

Do not duplicate the raw PPG signal into these files.

---

# 29. ERROR ANALYSIS — VERY IMPORTANT

The purpose of Phase 3A is not only model accuracy.

We want to understand WHY the model fails.

Implement:

code/modeling/error_analysis.py

Analyze:

1. error vs SBP
2. error vs DBP
3. error vs heart rate
4. error vs PPG amplitude
5. error vs pulse variability
6. error vs quality status
7. error by BP range
8. error by record

---

# 30. BP RANGE ANALYSIS

For SBP:

<90
90–119
120–139
140–159
>=160

For DBP:

<60
60–79
80–89
90–99
>=100

For each range report:

sample count
MAE
RMSE
bias
error SD

Do not remove small groups.

If a group has few examples:

report it clearly.

---

# 31. QUALITY-STRATIFIED ANALYSIS

Phase 2 already contains quality diagnostics.

Use them.

Compare model errors for:

PPG PASS

PPG WARN

high amplitude variability

mild clipping

pulse morphology warnings

etc., where available.

This is particularly important for research discovery.

We want to know whether:

good PPG → low error

poor/borderline PPG → high error

If this relationship exists, document it.

---

# 32. RECORD-LEVEL ANALYSIS

For the test set calculate:

MAE per record_id

and:

number of windows per record

for every record.

Then calculate:

median record-level MAE
mean record-level MAE
standard deviation
worst 10 records
best 10 records

Do NOT delete bad records.

The purpose is to understand generalization.

---

# 33. MODEL-ERROR CORRELATIONS

Calculate correlations between absolute BP error and:

- PPG amplitude
- PPG variance
- pulse amplitude variability
- pulse interval variability
- estimated HR
- target SBP
- target DBP
- quality flags

Create a summary table:

feature
correlation_with_abs_sbp_error
correlation_with_abs_dbp_error
p_value_if_appropriate

Use p-values carefully and do not imply causality from correlation.

---

# 34. BASELINE COMPARISON

Create:

code/outputs/metrics/classical_baseline_results.csv

Columns:

target
model
representation
feature_count
train_windows
validation_windows
test_windows
MAE
RMSE
R2
bias
error_std
pearson_r
spearman_r

Include:

Dummy
Linear Regression
Ridge
Random Forest
HistGradientBoosting

for both:

SBP
DBP

and relevant normalization representations.

---

# 35. VISUALIZATIONS

Create:

code/visualization/model_plots.py

Required plots:

1. SBP true vs predicted
2. DBP true vs predicted
3. SBP residual distribution
4. DBP residual distribution
5. SBP Bland–Altman
6. DBP Bland–Altman
7. SBP error vs reference BP
8. DBP error vs reference BP
9. SBP MAE by BP range
10. DBP MAE by BP range
11. Record-level MAE distribution
12. Model comparison
13. Error vs HR
14. Error vs PPG quality
15. Error vs pulse-amplitude variability

Save under:

code/outputs/figures/

---

# 36. BLAND–ALTMAN

For SBP:

x-axis:

mean(predicted, reference)

y-axis:

predicted - reference

Draw:

mean bias
bias ± 1.96 * SD

Repeat for DBP.

Do NOT interpret the limits as proof of clinical agreement.

They are descriptive error analysis.

---

# 37. NORMALIZATION COMPARISON

Compare:

Filtered PPG
Z-score
Robust normalization

using the SAME feature extraction definitions where mathematically appropriate.

Do not compare methods using different train/test partitions.

Do not select the winner using test results.

The primary question is:

Does removing amplitude scale improve generalization?

---

# 38. FEATURE IMPORTANCE

For tree models:

calculate model feature importance.

For the best tree-based baseline:

report:

feature
importance/rank

Also calculate permutation importance on VALIDATION only.

Do not use test data for feature importance during model selection.

Create:

code/outputs/metrics/feature_importance.csv

This is important for understanding what information the baseline is using.

---

# 39. AVOID AUTOMATIC FEATURE SELECTION

Do NOT automatically run:

RFE
recursive feature elimination
genetic feature selection
large feature search
automated feature elimination

in Phase 3A.

We want a transparent baseline.

Feature selection can be investigated later as a separate experiment.

---

# 40. BASELINE INTERPRETATION

After all models are evaluated, answer these questions:

### Question 1

Does PPG-only information outperform the Dummy Mean baseline?

### Question 2

Does nonlinear modeling outperform Linear/Ridge substantially?

### Question 3

Does normalization matter?

### Question 4

Are SBP and DBP equally predictable?

### Question 5

Does the model regress toward the mean?

### Question 6

Are extreme BP ranges significantly harder?

### Question 7

Do WARN/low-quality PPG windows have higher errors?

### Question 8

Are errors concentrated in particular records?

### Question 9

Which PPG features appear most informative?

### Question 10

Does the evidence suggest that handcrafted features are insufficient?

DO NOT answer these questions by speculation.

Use the actual results.

---

# 41. RESEARCH-DISCOVERY SECTION

Create:

code/outputs/reports/PHASE3A_RESEARCH_DISCOVERY.md

This is extremely important.

Do NOT invent novelty.

Instead, identify observed limitations.

Use this structure:

## Observation

What did the experiments actually show?

## Evidence

Which metric/plot supports it?

## Limitation

What limitation does this reveal?

## Hypothesis

What might address that limitation?

## Future Experiment

What experiment should be performed in Phase 3B?

Example format:

Observation:
SBP error increases strongly at high-pressure ranges.

Evidence:
SBP MAE is X in 120–139 and Y in >=160.

Limitation:
The model appears to regress toward the population mean.

Hypothesis:
A representation emphasizing waveform morphology may improve extreme-BP discrimination.

Future experiment:
Compare raw waveform deep representation against handcrafted morphology baseline.

IMPORTANT:

This section is a hypothesis-generation document.

It is NOT allowed to claim that a proposed idea is already novel.

---

# 42. RESEARCH-PAPER ORIENTATION

Create:

code/outputs/reports/PHASE3A_PAPER_NOTES.md

Include:

### Problem

Calibration-free PPG-only BP estimation.

### Dataset

Public Kachuee/MIMIC-II-derived dataset.

### Experimental protocol

10-second non-overlapping windows.

### Leakage control

Record-level split.

### Baseline methods

Dummy
Linear
Ridge
Random Forest
HistGradientBoosting

### Main findings

Based strictly on actual results.

### Limitations

Based strictly on actual error analysis.

### Potential research questions

Do not yet commit to a new architecture.

Possible questions should be derived from observed weaknesses.

---

# 43. REPRODUCIBILITY

Use:

RANDOM_STATE = 42

Store:

Python version

package versions

feature list

filter parameters

normalization method

model parameters

random seed

manifest path

split file path

experiment timestamp

inside:

code/outputs/reports/PHASE3A_EXPERIMENT_METADATA.json

---

# 44. SAMPLE RUN

Implement:

python code/scripts/run_classical_baselines.py --sample-run

Recommended sample:

1000 windows.

The sample must be deterministic.

The sample run must verify:

- all imports
- MAT loading
- window reconstruction
- filtering
- normalization
- peak detection
- feature extraction
- feature integrity
- model fitting
- evaluation
- figure generation
- prediction saving

If sample run fails:

STOP.

---

# 45. FULL RUN

Only after successful sample run:

python code/scripts/run_classical_baselines.py --full-run

The full run must process the complete modeling-eligible dataset.

Use progress indicators.

Do not load the entire raw dataset into RAM.

---

# 46. PERFORMANCE SAFETY

The project machine may have limited RAM.

Therefore:

- process MAT files incrementally
- process records incrementally
- batch feature generation
- persist feature cache incrementally
- avoid building giant Python lists containing all raw signals
- explicitly release large arrays where necessary

Do not create another copy of the raw 4.58 GB dataset.

---

# 47. REQUIRED NOTEBOOK

Create:

code/notebooks/03A_ppg_classical_baselines.ipynb

Sections:

1. Objective
2. Frozen dataset verification
3. Split verification
4. Example PPG loading
5. Filtering
6. Normalization representations
7. Feature extraction
8. Feature definitions
9. Feature integrity
10. Missingness analysis
11. Dummy baseline
12. Linear Regression
13. Ridge
14. Random Forest
15. HistGradientBoosting
16. Validation comparison
17. Configuration freeze
18. Final test evaluation
19. Error distribution
20. BP-range analysis
21. Signal-quality analysis
22. Record-level analysis
23. Feature importance
24. Research-discovery analysis
25. Conclusions
26. Phase 3B recommendation placeholder

Notebook must execute successfully.

---

# 48. REQUIRED REPORT

Create:

code/outputs/reports/PHASE3A_BASELINE_REPORT.md

Include:

## 1. Dataset

actual counts

## 2. Split

record counts and window counts

## 3. Signal preprocessing

filter

normalization

## 4. Feature set

complete feature list

## 5. Models

all models tested

## 6. Validation results

full table

## 7. Frozen configuration

selected model
selected representation
selected feature configuration

## 8. Final test results

SBP:

MAE
RMSE
R2
bias
error_std
Pearson
Spearman

DBP:

MAE
RMSE
R2
bias
error_std
Pearson
Spearman

## 9. Error analysis

BP ranges

quality strata

record-level errors

HR relationship

## 10. Feature importance

top features

## 11. Limitations

## 12. Research observations

## 13. Phase 3A conclusion

---

# 49. FINAL MODEL-SELECTION RULE

The best model must be selected using VALIDATION.

Correct:

TRAIN
  ↓
fit models
  ↓
VALIDATION
  ↓
select configuration
  ↓
FREEZE
  ↓
TEST once
  ↓
final report

Incorrect:

TRAIN
  ↓
TEST all models
  ↓
pick lowest test MAE

Do NOT do the incorrect procedure.

---

# 50. TEST SET FREEZE

Once the final configuration is selected using validation:

record exactly:

- model
- hyperparameters
- feature representation
- normalization
- imputation strategy
- feature count

Then freeze it.

Evaluate on TEST once.

Do not make any change based on the test result.

If a change is made afterward:

label it a new experiment.

---

# 51. NO CALIBRATION

Absolutely no:

- subject calibration
- record calibration
- initial BP offset
- true BP baseline
- delta prediction
- calibration using test data
- calibration using validation data

The baseline must be:

CALIBRATION-FREE.

---

# 52. NO DEEP LEARNING

Do NOT implement:

CNN
LSTM
GRU
Transformer
Attention
CNN-LSTM
ResNet
Temporal convolutional network
Autoencoder

Those belong to later phases.

---

# 53. NO HARDWARE INTEGRATION

Do not change the ESP32 pipeline in this phase.

MAX30102 hardware data collection will be handled separately.

For Phase 3A:

use the validated research dataset only.

---

# 54. ACCEPTANCE CRITERIA

Phase 3A is successful only if:

1. Phase 1 is untouched.
2. Phase 2 is untouched.
3. Original MAT files are untouched.
4. Original sample notebooks are untouched.
5. Record splits remain identical to Phase 2.
6. All partition-overlap checks pass.
7. Only PPG is used as model input.
8. No ABP/ECG/PPG-derived target leakage exists.
9. Feature extraction is reproducible.
10. Learned preprocessing uses TRAIN only.
11. Dummy baseline exists.
12. Linear Regression exists.
13. Ridge exists.
14. Random Forest exists.
15. HistGradientBoosting exists.
16. Validation is used for configuration selection.
17. Test is used only after freezing.
18. SBP and DBP are evaluated separately.
19. Error analysis is performed.
20. Feature importance is analyzed.
21. Research observations are documented.
22. No deep-learning model is trained.
23. No calibration is used.

If any scientific leakage or integrity assertion fails:

STOP.

---

# 55. FINAL RESPONSE TO USER

When Phase 3A finishes, report ONLY the completed work and actual evidence.

Include:

## Dataset
- train records
- validation records
- test records
- train windows
- validation windows
- test windows

## Features
- feature count
- feature groups
- missingness

## Models
- Dummy
- Linear
- Ridge
- Random Forest
- HistGradientBoosting

## Validation results

Complete comparison table.

## Frozen configuration

Explicitly identify:

model
representation
normalization
feature configuration
hyperparameters

## Final test results

SBP:

MAE
RMSE
R2
bias
error SD

DBP:

MAE
RMSE
R2
bias
error SD

## Error analysis

Report:

- BP-range performance
- quality-stratified performance
- record-level performance
- HR relationship
- strongest error correlations

## Top features

List top important PPG-only features.

## Research discoveries

List observed limitations and evidence.

Do NOT invent novelty.

## Leakage verification

Report:

train/validation overlap
train/test overlap
validation/test overlap

All must be zero.

## Files created

List important artifacts.

## Problems encountered

Be honest.

## Ready for Phase 3B

Answer:

YES / NO

Then STOP.

Do not automatically implement Phase 3B.

