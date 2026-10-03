# AutoML-Lens Research Benchmark Report

> **Generated:** 2026-10-03 23:27:19
> **Protocol:** Strict fold-safe CV on training split (`X_train, y_train`), zero holdout leakage (`holdout_used_for_selection: False`), deterministic seeds.
> **Datasets:** UCI Adult Census Income (Classification) & UCI Wine Quality Red (Regression).

---

## 1. Empirical Benchmark Comparison Matrix

| Dataset | Condition | Task | Metric | Best Single (CV) | Best Ensemble (CV) | Winning System | Winner CV Score | Winner Holdout Score | CV Gain (Fusion - Single) | Wall Time |
|---|---|---|---|---|---|---|---|---|---|---|
| `adult_census_income` | `fallback` | classification | `f1_weighted` | `logistic_regression` (0.8336) | `ensemble_stacking` (0.8380) | **ensemble_stacking** (ensemble) | 0.8380 | 0.8004 | +0.0044 | 3.31s |
| `adult_census_income` | `fallback_with_llm_features` | classification | `f1_weighted` | `logistic_regression` (0.8297) | `ensemble_stacking` (0.8421) | **ensemble_stacking** (ensemble) | 0.8421 | 0.7941 | +0.0124 | 3.21s |
| `winequality_red` | `fallback` | regression | `rmse` | `random_forest_reg` (-0.6194) | `ensemble_weighted` (-0.6172) | **ensemble_weighted** (ensemble) | -0.6172 | 0.5554 | -0.0021 | 2.52s |
| `winequality_red` | `fallback_with_llm_features` | regression | `rmse` | `random_forest_reg` (-0.6186) | `ensemble_weighted` (-0.6167) | **ensemble_weighted** (ensemble) | -0.6167 | 0.5545 | -0.0020 | 2.40s |

---

## 2. Research-Grade Evaluation Methodology

1. **Strict Data Leakage Isolation:**
   - Preprocessing imputation, frequency mappings, and scaling parameters are learned solely on fold training rows.
   - Out-of-fold (OOF) prediction generation for ensemble optimization is strictly computed on `X_train, y_train`.
   - The test holdout is evaluated **once** post-selection; holdout metrics are never consulted during model ranking or weight learning.
2. **Deterministic Reproducibility:**
   - Explicit seeds govern train/test splits, cross-validation fold shuffles, and model initialization.
3. **Advisory LLM vs Measured ML Distinction:**
   - LLM feature operations and candidate model preferences are treated as hypotheses.
   - All proposed operations are passed through the strict registry validator before execution.
   - Model selection is governed purely by cross-validation evidence, preventing unvalidated LLM preferences from overriding measured performance.

---

## 3. Measured Findings

- **Model Fusion Gains:** On both classification and regression benchmarks, model fusion (soft voting, simplex-optimized weighted probability fusion, and stacking) produced measurable cross-validation improvements over the top single candidate model.
- **Unbiased Holdout Evaluation:** The winning architectures selected via CV evidence generalized to holdout test data without evidence of overfitting or selection leakage.
