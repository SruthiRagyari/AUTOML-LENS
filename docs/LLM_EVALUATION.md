# LLM vs. Deterministic Fallback Empirical Evaluation

> **Protocol:** Pipeline execution in fast mode (`fast_demo=True`, 3-fold CV, 5 Optuna trials per model), fixed seed 42.
> **Effective Sample:** 2 datasets (classification.csv, winequality-red.csv). Across 3 repeats with a fixed seed, measured score variance was 0.0.
> **Note:** All figures in this document are computed strictly from measured execution records.

## Real System Limits & Observed Variance

1. **Fixed Models in Fast Mode:** In fast mode (`fast_demo=True`), the pipeline fixes the candidate model set to `logistic_regression, random_forest_clf, hist_gradient_boosting_clf`. Therefore, candidate model recommendations from the LLM or fallback do not alter which models are trained.
2. **Zero Seed Variance Across Repeats:** Because the pipeline uses fixed seeds (`random_state=42`, `OPTUNA_SEED=42`), the 3 repeats within each configuration produced identical holdout scores (variance = 0.0: clf fallback 0.000000, clf gemini 0.000000, wine fallback 0.000000, wine gemini 0.000000). The effective sample size is therefore **2 datasets**, not 12 independent trials.

## Measured Run Details

| Dataset | Repeat | Provider | Model | Recommended Models | Rec Metric | Models Trained | Winner | Holdout F1 | Holdout ROC-AUC | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| classification | 1 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, knn_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.881778 | 0.912037 | 11.0897 |
| classification | 1 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, logistic_regression, gradient_boosting_clf | roc_auc | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.814222 | 0.884259 | 16.8924 |
| classification | 2 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, knn_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.881778 | 0.912037 | 9.2563 |
| classification | 2 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, gradient_boosting_clf, logistic_regression, svm_clf | roc_auc | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.814222 | 0.884259 | 16.9816 |
| classification | 3 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, knn_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.881778 | 0.912037 | 8.8725 |
| classification | 3 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, logistic_regression, gradient_boosting_clf | roc_auc | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.814222 | 0.884259 | 16.872 |
| winequality-red | 1 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, hist_gradient_boosting_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 | 0.814471 | 36.5919 |
| winequality-red | 1 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, hist_gradient_boosting_clf, gradient_boosting_clf, svm_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 | 0.814471 | 48.2625 |
| winequality-red | 2 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, hist_gradient_boosting_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 | 0.814471 | 38.6994 |
| winequality-red | 2 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, hist_gradient_boosting_clf, gradient_boosting_clf, logistic_regression | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 | 0.814471 | 52.8925 |
| winequality-red | 3 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, hist_gradient_boosting_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 | 0.814471 | 43.4454 |
| winequality-red | 3 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, hist_gradient_boosting_clf, gradient_boosting_clf, svm_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 | 0.814471 | 52.4872 |

## Mean Execution Times by Provider & Dataset

| Dataset | Provider | Mean Wall Time (s) | Std Wall Time (s) |
|---|---|---|---|
| classification | fallback | 9.74 | 1.18 |
| classification | gemini | 16.92 | 0.06 |
| winequality-red | fallback | 39.58 | 3.51 |
| winequality-red | gemini | 51.21 | 2.56 |

## Ablation Study on `classification.csv`

To determine whether the score difference on `classification.csv` was caused by feature operations or model/metric choices, four conditions were evaluated:

| Condition | Provider | Feature Operations Applied | Winner | Holdout F1-weighted | Holdout ROC-AUC | Optuna Objective |
|---|---|---|---|---|---|---|
| (i) Fallback as is | Fallback (Deterministic) | None (0 ops) | HistGradientBoosting | 0.881778 | 0.912037 | f1_weighted |
| (ii) Gemini as is | Google Gemini | Balance_log1p, Gender_freq, Geography_freq, Income_abs | HistGradientBoosting | 0.814222 | 0.884259 | f1_weighted |
| (iii) Gemini with FE disabled | Google Gemini | None (0 ops) | HistGradientBoosting | 0.881778 | 0.912037 | f1_weighted |
| (iv) Fallback with Gemini FE applied | Fallback (Deterministic) | Balance_log1p, Gender_freq, Geography_freq, Income_abs | HistGradientBoosting | 0.814222 | 0.884259 | f1_weighted |

### Metric Analysis: Recommended vs. Optimized

From the stored experiment and training records:
- **`exp.primary_metric` assignment:** In `analyze_experiment`, `exp.primary_metric` defaults to `f1_weighted` for classification datasets before LLM analysis is called.
- **Advisory LLM metric:** Google Gemini recommended `roc_auc`, which was saved into `llm_analysis_json` under `recommended_metric`. However, the API does not overwrite `exp.primary_metric` with the LLM recommendation.
- **Optuna Objective:** When `train_experiment` invoked `OptunaOptimizer`, it passed `metric_name=exp.primary_metric` (`f1_weighted`). Therefore, **Optuna actually optimized `f1_weighted`** in all runs.
- **Ablation Insight:**
  - Comparing Condition (i) and (iii): When Gemini's feature operations are disabled, Gemini produces the exact same F1 score (0.881778) and ROC-AUC (0.912037) as Fallback.
  - Comparing Condition (ii) and (iv): When Gemini's feature operations are applied to Fallback, Fallback produces the exact same F1 score (0.814222) and ROC-AUC (0.884259) as Gemini.
  - The score difference on `classification.csv` is completely isolated to the feature transformations (`Balance_log1p`, `Gender_freq`, `Geography_freq`, `Income_abs`, `CreditScore_zscore`), and did not stem from metric configuration or model selection.

## Summary of Results

- On `winequality-red.csv`, neither provider proposed feature operations; both Fallback and Gemini produced the exact same winner (`HistGradientBoosting`) and holdout F1 score (`0.661949`).
- On `classification.csv`, Gemini proposed 5 valid feature operations while Fallback proposed none; the resulting feature transformations altered the feature space, leading to F1 `0.814222` vs `0.881778` for fallback (observed on these runs only).
- There is no evidence from these runs that the LLM improved accuracy over fallback.

## Reproducibility

To reproduce all measurements and regenerate this report directly from execution records:
```bash
python scripts/llm_vs_fallback.py
```
