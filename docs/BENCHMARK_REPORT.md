# AutoML-Lens Empirical Research Benchmark Report

> **Generated:** 2026-10-04 13:25:57
> **Git Revision:** `b8d45be`
> **Protocol:** Strict fold-safe CV on training split; holdout set reserved exclusively for final evaluation.
> **Statistical Scope:** Descriptive multi-seed comparison only (N=3 seeds per condition). No inferential p-values claimed.

---

## 1. Executive Summary & Research Findings

- **Empirical Reality:** Real performance measurements on real datasets (UCI Adult Census & UCI Wine Quality Red) across 3 reproducible seeds (`[42, 123, 456]`).
- **Holdout Discipline:** Model/pipeline selection was performed using training-side cross-validation only; the holdout set was reserved exclusively for final evaluation.
- **Ablation Findings:** Evaluated deterministic baseline, LLM model selection alone, and full LLM-guided pipeline planning.

---

## 2. Dataset: `adult_census_income` (CLASSIFICATION)

- **Primary Metric:** `f1_weighted` (Direction: `maximize`)
- **Total Samples:** 1000 (800 train / 200 holdout)
- **Dataset SHA256:** `5b00264637dbfec3`

### Multi-Seed Aggregate Results (Descriptive Statistics)

| Condition | Seeds | CV Score (Mean ± Std) | Holdout Score (Mean ± Std) | Holdout [Min, Max] | Mean Runtime |
|---|---|---|---|---|---|
| `deterministic` | 3 | 0.8376 ± 0.0035 | **0.8148 ± 0.0181** | [0.7937, 0.8379] | 14.97s |
| `llm_model_only` | 3 | 0.8395 ± 0.0035 | **0.8220 ± 0.0173** | [0.7980, 0.8379] | 15.69s |
| `llm_guided` | 3 | 0.8395 ± 0.0035 | **0.8220 ± 0.0173** | [0.7980, 0.8379] | 16.90s |

> **Comparison vs Deterministic Baseline:**
> - Deterministic Mean: **0.8148**
> - LLM-Guided Mean: **0.822**
> - **Direction-Adjusted $\Delta$:** **+0.0072** (LLM showed +0.0072 improvement)
> - Raw Difference: `+0.0072`

### Per-Seed Execution Records

| Seed | Condition | CV Score | Holdout Score | Selected Winning Pipeline | Wall Time |
|---|---|---|---|---|---|
| `42` | `deterministic` | 0.8423 | 0.7937 | `ensemble_soft_voting` | 11.91s |
| `123` | `deterministic` | 0.8363 | 0.8379 | `ensemble_stacking` | 18.52s |
| `456` | `deterministic` | 0.8342 | 0.8127 | `ensemble_soft_voting` | 14.49s |
| `42` | `llm_model_only` | 0.8430 | 0.7980 | `ensemble_soft_voting` | 13.69s |
| `123` | `llm_model_only` | 0.8347 | 0.8379 | `ensemble_stacking` | 18.69s |
| `456` | `llm_model_only` | 0.8408 | 0.8300 | `svm_clf` | 14.70s |
| `42` | `llm_guided` | 0.8430 | 0.7980 | `ensemble_soft_voting` | 15.79s |
| `123` | `llm_guided` | 0.8347 | 0.8379 | `ensemble_stacking` | 19.11s |
| `456` | `llm_guided` | 0.8408 | 0.8300 | `svm_clf` | 15.79s |

### Pipeline Behavior & Registry Validation Analysis

- **Candidate Models Proposed:** `gradient_boosting_clf, hist_gradient_boosting_clf, logistic_regression, random_forest_clf, svm_clf`
- **Candidate Models Validated:** `gradient_boosting_clf, hist_gradient_boosting_clf, logistic_regression, random_forest_clf, svm_clf`
- **Feature Operations Proposed:** 0 (None)
- **Feature Operations Accepted:** 0
- **Feature Operations Rejected:** 0
- **Fallback Rate:** 0% (0 of 6 runs)

---

## 2. Dataset: `winequality_red` (REGRESSION)

- **Primary Metric:** `rmse` (Direction: `minimize`)
- **Total Samples:** 800 (640 train / 160 holdout)
- **Dataset SHA256:** `4a402cf041b025d4`

### Multi-Seed Aggregate Results (Descriptive Statistics)

| Condition | Seeds | CV Score (Mean ± Std) | Holdout Score (Mean ± Std) | Holdout [Min, Max] | Mean Runtime |
|---|---|---|---|---|---|
| `deterministic` | 3 | -0.6051 ± 0.0103 | **0.5544 ± 0.0249** | [0.5289, 0.5881] | 6.35s |
| `llm_model_only` | 3 | -0.6077 ± 0.0068 | **0.5556 ± 0.0268** | [0.5285, 0.5921] | 6.29s |
| `llm_guided` | 3 | -0.6077 ± 0.0068 | **0.5556 ± 0.0268** | [0.5285, 0.5921] | 6.21s |

> **Comparison vs Deterministic Baseline:**
> - Deterministic Mean: **0.5544**
> - LLM-Guided Mean: **0.5556**
> - **Direction-Adjusted $\Delta$:** **-0.0012** (Deterministic showed +0.0012 advantage)
> - Raw Difference: `+0.0012`

### Per-Seed Execution Records

| Seed | Condition | CV Score | Holdout Score | Selected Winning Pipeline | Wall Time |
|---|---|---|---|---|---|
| `42` | `deterministic` | -0.6152 | 0.5461 | `ensemble_weighted` | 6.33s |
| `123` | `deterministic` | -0.6091 | 0.5289 | `ensemble_weighted` | 7.02s |
| `456` | `deterministic` | -0.5909 | 0.5881 | `ensemble_weighted` | 5.69s |
| `42` | `llm_model_only` | -0.6152 | 0.5461 | `ensemble_weighted` | 6.13s |
| `123` | `llm_model_only` | -0.6092 | 0.5285 | `ensemble_weighted` | 7.08s |
| `456` | `llm_model_only` | -0.5988 | 0.5921 | `ensemble_weighted` | 5.65s |
| `42` | `llm_guided` | -0.6152 | 0.5461 | `ensemble_weighted` | 6.07s |
| `123` | `llm_guided` | -0.6092 | 0.5285 | `ensemble_weighted` | 6.92s |
| `456` | `llm_guided` | -0.5988 | 0.5921 | `ensemble_weighted` | 5.62s |

### Pipeline Behavior & Registry Validation Analysis

- **Candidate Models Proposed:** `gradient_boosting_reg, random_forest_reg, ridge`
- **Candidate Models Validated:** `gradient_boosting_reg, random_forest_reg, ridge`
- **Feature Operations Proposed:** 0 (None)
- **Feature Operations Accepted:** 0
- **Feature Operations Rejected:** 0
- **Fallback Rate:** 0% (0 of 6 runs)

---

## 3. Scientific Integrity & Limitations

1. **Holdout Isolation Guarantee:** At no point did the LLM, feature engineering, Optuna HPO, or ensemble stacking consult holdout test rows. Holdout evaluation was executed exactly once post-selection.
2. **Safety & Zero Arbitrary Code:** The LLM functioned strictly as a structured decision layer constrained by `ModelRegistry` and `OPERATION_REGISTRY`. No dynamically generated Python code was executed.
3. **Claim Boundary:** This research makes descriptive multi-seed comparisons. It does not claim general superiority or statistical significance without larger sample sizes.

---
*Report automatically generated from measured benchmark records by AutoML-Lens.*