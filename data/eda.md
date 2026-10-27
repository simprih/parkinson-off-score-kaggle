# EDA: Parkinson's OFF motor score (bobathon-esilv)

> **Day 1 analysis (8 October), made on the first version of the files, before the update of 9 October.** There was no `rater_id` column yet, and the correlation of `off` with the target was 0.886 (0.871 on the current files). The up-to-date figures and conclusions are in the README, sections 2 and 4; the open questions are answered at the end of this file.

Interactive report: [data/eda_visits.html](eda_visits.html)

## Dataset at a glance

| | Train | Test |
|---|---|---|
| Rows (visits) | 44 590 | 11 013 |
| Columns | 12 (11 features + target) | 11 (no target) |
| Distinct patients | 5 576 | 1 395 |
| Visits per patient | median 7, max 12 | — |

Each row is one clinical visit. The `patient_id` is a pure holdout: no
train patient appears in test and no test patient appears in train.

The target is `target`: the debiased true OFF MDS-UPDRS motor score (a
continuous value; theoretical scale 0–132).

## Per-column findings

| Column | dtype | null % | n_unique | Notes |
|---|---|---|---|---|
| `patient_id` | string | 0 % | 5 576 | Patient identifier. 12.5 % unique-ratio → one per patient, not one per visit. **Not a model feature.** |
| `cohort` | string | 0 % | 2 | Binary — two cohorts (A / B). |
| `sexM` | int | 0 % | 2 | Binary, fully observed. |
| `gene` | string | 32.4 % | 4 | Mutation label; ~1/3 missing. |
| `age_at_diagnosis` | float | 5.2 % | 563 | Small missingness. |
| `age` | float | 0 % | 1 150 | Fully observed. |
| `ledd` | float | 36.6 % | 1 320 | Levodopa equivalent daily dose; substantial missingness — possibly cohort-related. |
| `time_since_intake_on` | float | 46.4 % | 64 | Hours since last ON intake; nearly half missing. |
| `time_since_intake_off` | float | 78.8 % | 178 | Hours since last OFF intake; most often missing — the uncomfortable test is frequently skipped. |
| `on` | float | 29.6 % | 80 | Measured ON motor score; ~30 % missing. |
| `off` | float | 42.4 % | 93 | Measured OFF motor score; ~42 % missing. |
| `target` | float | 0 % | 867 | Fully observed (train only). |

**Missingness is structural, not random.** The `off` score is the most
uncomfortable to collect; `time_since_intake_off` tracks when it was taken
and is missing at the same high rate. `ledd` missingness may correlate with
cohort or disease stage.

## Target

`target` covers roughly 0–110 in practice (p0 = 0.0, p100 = 109.5), well
below the theoretical 132 ceiling. Distribution:

- mean = 37.5, median = 37.3
- std = 16.5, IQR = 23.7 (25th percentile 25.6, 75th 49.3)
- histogram is roughly bell-shaped around the mid-30s with a mild right tail

The distribution is **continuous and approximately unimodal**. No extreme
class imbalance. Regression task; RMSE is the natural metric (and the Kaggle
evaluation metric). Stratification is not required.

## Structure

No datetime columns were detected by dtype. Age and `age_at_diagnosis` encode
disease timeline implicitly — `age - age_at_diagnosis` gives approximate years
since diagnosis. There is no explicit timestamp column.

**Critical: `patient_id` repeats across visits** (median 7 visits per
patient). The Kaggle holdout is by patient. Any K-Fold that ignores this will
leak patients across folds and inflate CV scores relative to test scores.

## Associations

Ranked by Cramér V (captures both linear and non-linear association, works
with skrub regardless of dtype):

| Features | Cramér V | Pearson r | Notes |
|---|---|---|---|
| `off` ↔ `target` | 0.495 | **0.886** | Strongest predictor. `off` is the clinic OFF score — highly correlated with true OFF. But 42 % is missing. |
| `on` ↔ `target` | 0.403 | 0.688 | Second strongest. ON score also tracks disease severity. |
| `ledd` ↔ `target` | 0.236 | 0.298 | Moderate. Higher dose → higher severity. |
| `age` ↔ `target` | 0.114 | 0.310 | Weak but real. |
| `time_since_intake_on` ↔ `target` | 0.177 | ≈ 0 | Non-linear association (Cramér V) with near-zero linear correlation — suggests interaction, not direct effect. |
| `time_since_intake_off` ↔ `target` | 0.092 | ≈ 0 | Same pattern. |
| `cohort` ↔ `target` | 0.090 | — | Small but present cohort effect. |

**Notable cross-feature link:**
- `age_at_diagnosis` ↔ `age`: Pearson r = 0.94 — almost collinear (older
  patients diagnosed older). Only one is needed; a derived
  `time_since_diagnosis = age - age_at_diagnosis` captures both.
- `on` ↔ `off`: Pearson r = 0.87 — strongly correlated, as expected from the
  clinical literature.

**Leakage check on `off`:** Pearson r = 0.886 with `target` is high but
clinically expected — `off` *is* the biased version of `target`. It is not
a copy (42 % missing, and the spread around the 1:1 line reflects drug-timing
bias). Keeping `off` as a feature is appropriate; the model must learn to
debias it and fill in the large fraction of missing values. No leakage flag.

## Modelling implications

| Finding | Implication |
|---|---|
| `patient_id` groups, no patient overlap train/test | Use `GroupKFold` (or `GroupShuffleSplit`) with `groups=patient_id`; **do not use plain KFold** |
| Continuous, roughly unimodal target | RMSE primary metric; no need to stratify |
| High missingness in `off`, `time_since_intake_off`, `ledd`, `on` | An imputer is mandatory in the pipeline; these columns carry real signal (not MCAR) so drop is unsafe |
| `age_at_diagnosis` + `age` nearly collinear | Consider replacing both with `time_since_diagnosis = age - age_at_diagnosis` |
| `off` ↔ `target` r = 0.886 | `off` is the strongest single predictor; handle its missingness carefully |
| No datetime column detected | `TimeSeriesSplit` is not directly indicated; use `GroupKFold` for patient holdout |
| Two categorical columns (`gene`, `cohort`) + binary `sexM` | A tabular encoder (e.g. `skrub.tabular_pipeline`) will handle them; `gene` has ~32 % missing |

## Open questions

1. **Cohort structure**: what distinguishes cohorts A and B clinically? The
   cohort effect on target is small but real — it may warrant a cohort
   interaction feature or a stratified split check.

   **Answer:** the data does not say what separates them clinically. Cohort B patients have half as many visits
   (4.3 against 8.9 per patient), a lower mean target (33.7 against 37.9) and more often no OFF measurement at all
   (3.1 % of the patients against 0.7 %); our error is higher on B (2.96 against 2.72, README 7.3).
2. **Missing `off` mechanism**: is `off` missing more often for one cohort or
   gene? If so, a missing-indicator feature for `off` may help.

   **Answer:** hardly. OFF is missing on 43 % of the visits of cohort A and 38 % of cohort B, and on 38 to 43 %
   of the visits of each gene group. Instead of one indicator, the final model gives each visit a type (OFF
   measured, ON with a known intake time, ON without it) with its own noise model (README 5.4 and 5.5).
3. **Test column `time_since_diagnosis`**: the day-1 guide mentions test may
   carry `time_since_diagnosis` while train does not. Verify once the latest
   `X_test.csv` is fully profiled.

   **Answer:** no. Train and test have the same columns (plus `rater_id` since 9 October) and none is
   `time_since_diagnosis`; we compute `age - age_at_diagnosis` ourselves, after imputing the missing ages at
   diagnosis (README 4.5).
4. **Score ceiling**: a few test visits may come from patients with very short
   or zero disease history — cold-start edge case to watch for.

   **Answer:** every patient has 4 to 12 visits, in train as in test, so none comes without history. The hard
   case is a patient with no OFF measurement (1.2 % of the training patients, 1.4 % of the test patients): the
   error is 5.36 for them, against 2.29 with six OFF measurements or more (README 7.3).
