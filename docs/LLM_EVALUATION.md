# LLM vs. Deterministic Fallback Evaluation

> **Protocol:** Fast mode (`fast_demo=True`, 3-fold CV, 5 Optuna trials per model), fixed seed 42.
> **Sample Size:** 2 datasets x 3 repeats = 6 runs per provider (12 runs total).
> **Note:** Differences are not statistically tested; findings are observed on these runs only.

## Summary of Findings

- **Recommendations:** In 6/6 runs, Google Gemini recommended different candidate models or primary metrics than the rule-based fallback.
- **Feature Engineering:** For `classification.csv`, Gemini proposed 5 valid feature operations (`Balance_log1p`, `Income_zscore`, `CreditScore_zscore`, `Gender_freq`, `Geography_freq`), all of which were accepted and executed. Fallback proposed 0 feature operations. For `winequality-red.csv`, neither provider proposed feature operations.
- **Holdout Score & Winner:**
  - `classification.csv`: Winning model was `HistGradientBoosting` across all runs. However, holdout F1-weighted score differed: Fallback achieved 0.881778 whereas Gemini achieved 0.814222 across all 3 repeats (observed on these runs only; Gemini's engineered features and metric recommendation altered the feature space and Optuna tuning).
  - `winequality-red.csv`: LLM changed nothing in the outcome. Both Fallback and Gemini selected `HistGradientBoosting` with the exact same holdout F1-weighted score of 0.661949 across all 3 repeats.
- **Execution Time:** Fallback runs averaged ~7.6s on classification and ~51.7s on wine quality. Gemini runs averaged ~13.8s on classification and ~60.2s on wine quality (additional time spent on Gemini API call and feature transformation).

## Measured Run Details

| Dataset | Repeat | Provider | Model | Recommended Models | Rec Metric | Models Trained | Winner | Holdout Score | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| classification | 1 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, knn_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.881778 (f1_weighted) | 7.8769 |
| classification | 1 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, gradient_boosting_clf, logistic_regression | roc_auc | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.814222 (f1_weighted) | 13.6121 |
| classification | 2 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, knn_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.881778 (f1_weighted) | 7.3933 |
| classification | 2 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, gradient_boosting_clf, logistic_regression | roc_auc | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.814222 (f1_weighted) | 13.2572 |
| classification | 3 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, knn_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.881778 (f1_weighted) | 7.6438 |
| classification | 3 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, gradient_boosting_clf, logistic_regression | roc_auc | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.814222 (f1_weighted) | 14.3845 |
| winequality-red | 1 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, hist_gradient_boosting_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 (f1_weighted) | 48.6256 |
| winequality-red | 1 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, hist_gradient_boosting_clf, svm_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 (f1_weighted) | 58.8379 |
| winequality-red | 2 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, hist_gradient_boosting_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 (f1_weighted) | 51.9047 |
| winequality-red | 2 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, hist_gradient_boosting_clf, gradient_boosting_clf, svm_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 (f1_weighted) | 59.9478 |
| winequality-red | 3 | Fallback (Deterministic) | rule-based | logistic_regression, random_forest_clf, gradient_boosting_clf, hist_gradient_boosting_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 (f1_weighted) | 54.4336 |
| winequality-red | 3 | Google Gemini | gemini-flash-lite-latest | random_forest_clf, hist_gradient_boosting_clf, gradient_boosting_clf, svm_clf | f1_weighted | logistic_regression, random_forest_clf, hist_gradient_boosting_clf | HistGradientBoosting | 0.661949 (f1_weighted) | 61.8038 |

## Comparison: Gemini vs. Fallback

| Dataset | Repeat | Fallback Winner (Score) | Gemini Winner (Score) | Recs Differed? | Outcome Changed? |
|---|---|---|---|---|---|
| classification | 1 | HistGradientBoosting (0.881778) | HistGradientBoosting (0.814222) | Yes | Yes |
| classification | 2 | HistGradientBoosting (0.881778) | HistGradientBoosting (0.814222) | Yes | Yes |
| classification | 3 | HistGradientBoosting (0.881778) | HistGradientBoosting (0.814222) | Yes | Yes |
| winequality-red | 1 | HistGradientBoosting (0.661949) | HistGradientBoosting (0.661949) | Yes | No |
| winequality-red | 2 | HistGradientBoosting (0.661949) | HistGradientBoosting (0.661949) | Yes | No |
| winequality-red | 3 | HistGradientBoosting (0.661949) | HistGradientBoosting (0.661949) | Yes | No |

## Reproducibility

To reproduce these exact measurements, run:
```bash
python scripts/llm_vs_fallback.py
```
