# LLM vs. Deterministic Fallback Empirical Evaluation

> **Protocol:** Pipeline execution in fast mode (`fast_demo=True`, 3-fold CV, 5 Optuna trials per model).
> **Isolation:** Run against isolated temporary database and storage (`DATABASE_URL`, `STORAGE_PATH`).
> **Note:** All figures and text in this document are computed directly from measured execution records.

## Real System Limits & Observed Variance

1. **Fixed Models in Fast Mode:** In fast mode (`fast_demo=True`), the pipeline fixes candidate models to `logistic_regression, random_forest_clf, hist_gradient_boosting_clf`. Therefore, candidate model recommendations from the LLM or fallback do not change which models are trained in this mode.
2. **Advisory Metric:** The LLM's recommended metric is advisory and stored in `llm_analysis_json`. The API does not override `exp.primary_metric` (`f1_weighted`), so Optuna strictly optimizes `f1_weighted` in both fallback and LLM runs (as evidenced by `optimization.metric` in the stored training records: `f1_weighted`).
3. **Post-Hoc Injections:** In the ablation experiments, conditions (iii), (iv) and the per-operation variations modify `exp.llm_analysis_json` after the `/analyze` step. These are post-hoc experimental injections to isolate feature transformations, not native product behaviour.

## Seed Variation Experiment (Seeds 42, 43, 44)

To evaluate stability across train/test splits and Optuna sampler seeds, runs were evaluated on seeds 42, 43, and 44 for both datasets.

### Dataset: `classification` (Total: 300 rows | Train: 240 rows | Holdout: 60 rows)

| Seed | Fallback F1 | Gemini F1 | Paired Diff (Gemini - FB) | Fallback ROC-AUC | Gemini ROC-AUC | Paired Diff ROC-AUC | Fallback Winner | Gemini Winner |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.881778 | 0.814222 | -0.067556 | 0.912037 | 0.884259 | -0.027778 | HistGradientBoosting | HistGradientBoosting |
| 43 | 0.949811 | 0.932785 | -0.017026 | 0.981481 | 0.960648 | -0.020833 | Random Forest | Random Forest |
| 44 | 0.849432 | 0.881778 | +0.032346 | 0.896991 | 0.913194 | +0.016203 | Random Forest | Random Forest |

**Aggregate Metrics for `classification` (mean ± std across seeds 42, 43, 44):**
- **Fallback F1-weighted:** 0.893674 ± 0.051236
- **Gemini F1-weighted:** 0.876262 ± 0.059474
- **Fallback ROC-AUC:** 0.930170 ± 0.045069
- **Gemini ROC-AUC:** 0.919367 ± 0.038567

### Dataset: `winequality-red` (Total: 1599 rows | Train: 1279 rows | Holdout: 320 rows)

| Seed | Fallback F1 | Gemini F1 | Paired Diff (Gemini - FB) | Fallback ROC-AUC | Gemini ROC-AUC | Paired Diff ROC-AUC | Fallback Winner | Gemini Winner |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.661949 | 0.661949 | +0.000000 | 0.814471 | 0.814471 | +0.000000 | HistGradientBoosting | HistGradientBoosting |
| 43 | 0.681542 | 0.686802 | +0.005260 | 0.877213 | 0.872774 | -0.004439 | Random Forest | Random Forest |
| 44 | 0.687144 | 0.687144 | +0.000000 | 0.853558 | 0.853558 | +0.000000 | HistGradientBoosting | HistGradientBoosting |

**Aggregate Metrics for `winequality-red` (mean ± std across seeds 42, 43, 44):**
- **Fallback F1-weighted:** 0.676878 ± 0.013229
- **Gemini F1-weighted:** 0.678632 ± 0.014449
- **Fallback ROC-AUC:** 0.848414 ± 0.031686
- **Gemini ROC-AUC:** 0.846934 ± 0.029711

### Operation Consistency Across Seeds & Batches

- On `classification` with provider `fallback`, proposed operations were identical across all seeds: `none`.
- On `classification` with provider `fallback`, accepted operations were identical across all seeds: `none`.
- On `classification` with provider `gemini`, proposed operations varied across seeds: [('Age_zscore', 'Balance_log1p', 'Gender_freq', 'Geography_freq', 'Income_zscore'), ('Balance_log1p', 'Gender_freq', 'Geography_freq'), ('Age_zscore', 'Balance_log1p', 'CreditScore_zscore', 'Gender_freq', 'Geography_freq', 'Income_zscore')].
- On `classification` with provider `gemini`, accepted operations varied across seeds: [('Age_zscore', 'Balance_log1p', 'Gender_freq', 'Geography_freq', 'Income_zscore'), ('Balance_log1p', 'Gender_freq', 'Geography_freq'), ('Age_zscore', 'Balance_log1p', 'CreditScore_zscore', 'Gender_freq', 'Geography_freq', 'Income_zscore')].
- On `winequality-red` with provider `fallback`, proposed operations were identical across all seeds: `none`.
- On `winequality-red` with provider `fallback`, accepted operations were identical across all seeds: `none`.
- On `winequality-red` with provider `gemini`, proposed operations varied across seeds: [('alcohol_zscore', 'chlorides_log1p', 'residual sugar_log1p', 'sulphates_log1p'), ('alcohol_zscore', 'chlorides_log1p', 'residual sugar_log1p', 'sulphates_log1p'), ('alcohol_zscore', 'chlorides_log1p', 'residual sugar_log1p', 'sulphates_log1p', 'volatile acidity_zscore')].
- On `winequality-red` with provider `gemini`, accepted operations varied across seeds: [('alcohol_zscore', 'chlorides_log1p', 'residual sugar_log1p', 'sulphates_log1p'), ('alcohol_zscore', 'chlorides_log1p', 'residual sugar_log1p', 'sulphates_log1p'), ('alcohol_zscore', 'chlorides_log1p', 'residual sugar_log1p', 'sulphates_log1p', 'volatile acidity_zscore')].

## All Measured Run Details

| Dataset | Seed | Provider | Model | Proposed Operations | Accepted Operations | Rejected Operations | Models Trained | Winner | F1-weighted | ROC-AUC | Wall Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classification | 42 | Fallback (Deterministic) | rule-based | None (0 ops) | None (0 ops) | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.881778 | 0.912037 | 23.9574 |
| classification | 42 | Google Gemini | gemini-flash-lite-latest | Balance_log1p, Gender_freq, Geography_freq, Age_zscore, Income_zscore | Balance_log1p, Gender_freq, Geography_freq, Age_zscore, Income_zscore | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.814222 | 0.884259 | 27.4896 |
| classification | 43 | Fallback (Deterministic) | rule-based | None (0 ops) | None (0 ops) | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | Random Forest | 0.949811 | 0.981481 | 8.8691 |
| classification | 43 | Google Gemini | gemini-flash-lite-latest | Balance_log1p, Gender_freq, Geography_freq | Balance_log1p, Gender_freq, Geography_freq | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | Random Forest | 0.932785 | 0.960648 | 13.9965 |
| classification | 44 | Fallback (Deterministic) | rule-based | None (0 ops) | None (0 ops) | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | Random Forest | 0.849432 | 0.896991 | 10.8185 |
| classification | 44 | Google Gemini | gemini-flash-lite-latest | Balance_log1p, Income_zscore, CreditScore_zscore, Age_zscore, Gender_freq, Geography_freq | Balance_log1p, Income_zscore, CreditScore_zscore, Age_zscore, Gender_freq, Geography_freq | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | Random Forest | 0.881778 | 0.913194 | 17.1625 |
| winequality-red | 42 | Fallback (Deterministic) | rule-based | None (0 ops) | None (0 ops) | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 | 0.814471 | 34.5296 |
| winequality-red | 42 | Google Gemini | gemini-flash-lite-latest | residual sugar_log1p, chlorides_log1p, sulphates_log1p, alcohol_zscore | residual sugar_log1p, chlorides_log1p, sulphates_log1p, alcohol_zscore | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 | 0.814471 | 46.7911 |
| winequality-red | 43 | Fallback (Deterministic) | rule-based | None (0 ops) | None (0 ops) | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | Random Forest | 0.681542 | 0.877213 | 23.6706 |
| winequality-red | 43 | Google Gemini | gemini-flash-lite-latest | residual sugar_log1p, chlorides_log1p, sulphates_log1p, alcohol_zscore | residual sugar_log1p, chlorides_log1p, sulphates_log1p, alcohol_zscore | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | Random Forest | 0.686802 | 0.872774 | 33.0467 |
| winequality-red | 44 | Fallback (Deterministic) | rule-based | None (0 ops) | None (0 ops) | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.687144 | 0.853558 | 55.7188 |
| winequality-red | 44 | Google Gemini | gemini-flash-lite-latest | residual sugar_log1p, chlorides_log1p, sulphates_log1p, alcohol_zscore, volatile acidity_zscore | residual sugar_log1p, chlorides_log1p, sulphates_log1p, alcohol_zscore, volatile acidity_zscore | None (0 rejected) | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.687144 | 0.853558 | 75.8543 |

## Mean Execution Times by Provider & Dataset

| Dataset | Provider | Mean Wall Time (s) | Std Wall Time (s) |
|---|---|---|---|
| classification | fallback | 14.55 | 8.21 |
| classification | gemini | 19.55 | 7.06 |
| winequality-red | fallback | 37.97 | 16.30 |
| winequality-red | gemini | 51.90 | 21.86 |

## 4-Condition Ablation on `classification.csv` (Seed 42)

To evaluate the impact of LLM-generated feature operations versus rule-based defaults, four conditions were evaluated:

| Condition | Provider | Feature Operations Applied | Winner | Holdout F1 | Delta F1 vs (i) | Holdout ROC-AUC | Delta ROC-AUC vs (i) | Optuna Metric |
|---|---|---|---|---|---|---|---|---|
| (i) Fallback as is | Fallback (Deterministic) | None (0 ops) | HistGradientBoosting | 0.881778 | +0.000000 | 0.912037 | +0.000000 | f1_weighted |
| (ii) Gemini as is | Google Gemini | Balance_log1p, Gender_freq, Geography_freq, Age_zscore, Income_zscore | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | f1_weighted |
| (iii) Gemini with FE disabled (post-hoc injection) | Google Gemini | None (0 ops) | HistGradientBoosting | 0.881778 | +0.000000 | 0.912037 | +0.000000 | f1_weighted |
| (iv) Fallback with Gemini FE applied (post-hoc injection) | Fallback (Deterministic) | Balance_log1p, Gender_freq, Geography_freq, Age_zscore, Income_zscore | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | f1_weighted |

## Per-Operation Ablation on `classification.csv` (Seed 42)

To test which specific feature operations cause changes in performance, each Gemini operation was applied individually to Fallback analysis, and in leave-one-out combinations:

| Ablation Variant | Feature Operations Applied | Winner | Holdout F1 | Delta F1 vs Baseline | Holdout ROC-AUC | Delta ROC-AUC vs Baseline | Wall Time (s) |
|---|---|---|---|---|---|---|---|
| Fallback + [Balance_log1p] alone | Balance_log1p | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | 9.3875 |
| Fallback + [Gender_freq] alone | Gender_freq | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | 9.5304 |
| Fallback + [Geography_freq] alone | Geography_freq | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | 8.9068 |
| Fallback + [Age_zscore] alone | Age_zscore | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | 10.0967 |
| Fallback + [Income_zscore] alone | Income_zscore | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | 9.245 |
| Fallback + All except [Balance_log1p] | Gender_freq, Geography_freq, Age_zscore, Income_zscore | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | 8.8247 |
| Fallback + All except [Gender_freq] | Balance_log1p, Geography_freq, Age_zscore, Income_zscore | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | 19.0949 |
| Fallback + All except [Geography_freq] | Balance_log1p, Gender_freq, Age_zscore, Income_zscore | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | 20.1135 |
| Fallback + All except [Age_zscore] | Balance_log1p, Gender_freq, Geography_freq, Income_zscore | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | 21.8797 |
| Fallback + All except [Income_zscore] | Balance_log1p, Gender_freq, Geography_freq, Age_zscore | HistGradientBoosting | 0.814222 | -0.067556 | 0.884259 | -0.027778 | 20.3476 |

## Summary of Findings (Computed Directly from Results)

1. **Impact of 4-Condition Ablation:**
   - Condition (i) Fallback as is yielded F1 0.881778.
   - Condition (ii) Gemini as is (with 5 applied operations: `Balance_log1p, Gender_freq, Geography_freq, Age_zscore, Income_zscore`) yielded F1 0.814222 (delta vs (i): -0.067556).
   - Condition (iii) Gemini with feature operations disabled yielded F1 0.881778 (delta vs (i): +0.000000, matching Condition (i) exactly).
   - Condition (iv) Fallback with Gemini feature operations applied yielded F1 0.814222 (delta vs (i): -0.067556, matching Condition (ii) exactly).
   - The four conditions show that the score difference between Fallback and Gemini on `classification.csv` is mediated by the applied feature operations (`Balance_log1p, Gender_freq, Geography_freq, Age_zscore, Income_zscore`).
2. **Per-Operation Impact:** Testing each operation individually against the Fallback baseline (F1 0.881778) yielded: `Balance_log1p` alone (F1 0.814222, delta -0.067556); `Gender_freq` alone (F1 0.814222, delta -0.067556); `Geography_freq` alone (F1 0.814222, delta -0.067556); `Age_zscore` alone (F1 0.814222, delta -0.067556); `Income_zscore` alone (F1 0.814222, delta -0.067556).
3. **Metric Alignment:** In all runs, `exp.primary_metric` remained `f1_weighted`, and the stored `optimization.metric` records confirm that Optuna optimized `f1_weighted` throughout.
4. **Dataset Specificity:** On `winequality-red.csv`, Provider `fallback` applied no operations (0 ops); Provider `gemini` applied operations: `alcohol_zscore, chlorides_log1p, residual sugar_log1p, sulphates_log1p, volatile acidity_zscore`.

## Reproducibility

To reproduce all measurements and regenerate this report against an isolated database:
```bash
python scripts/llm_vs_fallback.py
```
