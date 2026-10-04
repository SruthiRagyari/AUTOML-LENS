# AutoML-Lens: Final Empirical Benchmark & Evaluation Results

This document presents the definitive empirical results from the rigorous multi-seed evaluation of **AutoML-Lens**. All values recorded below are computed from stored experimental records ([`benchmarks/results/benchmark_results.json`](../benchmarks/results/benchmark_results.json)) and aggregated multi-seed statistics ([`benchmarks/results/benchmark_summary.json`](../benchmarks/results/benchmark_summary.json)).

---

## 1. Experimental Protocol & Controls

To prevent data contamination, optimistic bias, and false claims of superiority, all experiments conformed to the following experimental standards:

- **Datasets**:
  1. **UCI Adult Census Income (`adult_census_income`)**: Binary classification, 14 features (mixed numerical/categorical), target: `income`, evaluated on `f1_weighted` (higher is better). Subsampled to 2,000 rows under identical stratification.
  2. **UCI Wine Quality Red (`wine_quality_red`)**: Tabular regression, 1,599 rows, 11 numerical physicochemical features, target: `quality`, evaluated on `rmse` (lower is better).
- **Seeds**: Exactly matched seed list: `[42, 123, 456]` across all conditions ($N=3$ seeds per condition, 18 total experimental runs).
- **Holdout Discipline**: $20\%$ holdout test set (`test_size=0.2`) strictly isolated post-ingestion. Never seen by CV folds, Optuna trials, or ensemble weight optimizers. Evaluated exactly once post-selection.
- **Cross-Validation**: 3-fold stratified cross-validation on the $80\%$ training split.
- **Budget Matching**: Identical Optuna trial budgets and candidate evaluation protocols across conditions.
- **Registry Constraints**: Zero arbitrary code execution; all candidate models and feature operations were constrained to pre-validated system registries.

---

## 2. Experimental Conditions

1. **Condition A (`deterministic`)**: Standard heuristic AutoML baseline without LLM assistance (standard preprocessing, heuristic candidate model selection, fold-safe CV, Optuna HPO, greedy model fusion ensemble).
2. **Condition B (`llm_model_only` ablation)**: LLM selects candidate models from registry; default preprocessing and zero feature operations.
3. **Condition C (`llm_guided`)**: Full LLM pipeline planning (Gemini-generated structured plan with validated candidate models, feature operations, HPO focus, and ensemble configuration).

---

## 3. Measured Results Matrix

### A. UCI Adult Census Income (Binary Classification — `f1_weighted` $\uparrow$)

| Condition | Seed | CV Score | Holdout Score | Selected Winning Pipeline | Secondary Accuracy | Secondary ROC-AUC | Runtime |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Deterministic Baseline** | 42 | 0.8260 | 0.8148 | `ensemble_weighted` | 0.8325 | 0.8818 | 4.39s |
| | 123 | 0.8354 | 0.8335 | `hist_gradient_boosting_clf` | 0.8475 | 0.9022 | 4.09s |
| | 456 | 0.8211 | 0.7961 | `ensemble_weighted` | 0.8175 | 0.8596 | 3.65s |
| **LLM Model-Only (Ablation)** | 42 | 0.8260 | 0.8148 | `ensemble_weighted` | 0.8325 | 0.8818 | 3.99s |
| | 123 | 0.8354 | 0.8335 | `hist_gradient_boosting_clf` | 0.8475 | 0.9022 | 4.40s |
| | 456 | 0.8247 | 0.8176 | `svm_clf` | 0.8300 | 0.8710 | 5.34s |
| **Full LLM-Guided Plan** | 42 | 0.8260 | 0.8148 | `ensemble_weighted` | 0.8325 | 0.8818 | 4.31s |
| | 123 | 0.8354 | 0.8335 | `hist_gradient_boosting_clf` | 0.8475 | 0.9022 | 4.54s |
| | 456 | 0.8247 | 0.8176 | `svm_clf` | 0.8300 | 0.8710 | 5.38s |

#### Aggregate Descriptive Statistics (Adult Census):
- **Deterministic Baseline**: $0.8148 \pm 0.0187$ (Holdout Range: $[0.7961, 0.8335]$, Mean Runtime: $4.04\text{s}$)
- **LLM Model-Only**: $0.8220 \pm 0.0101$ (Holdout Range: $[0.8148, 0.8335]$, Mean Runtime: $4.58\text{s}$)
- **Full LLM-Guided**: $0.8220 \pm 0.0101$ (Holdout Range: $[0.8148, 0.8335]$, Mean Runtime: $4.74\text{s}$)
- **Direction-Adjusted $\Delta$ (LLM vs Deterministic)**: $\mathbf{+0.0072}$
- **Secondary Accuracy Mean**: Deterministic: $0.8325 \pm 0.0150$, LLM: $0.8367 \pm 0.0094$ ($\Delta = +0.0042$)

---

### B. UCI Wine Quality Red (Tabular Regression — `rmse` $\downarrow$)

| Condition | Seed | CV Score | Holdout Score | Selected Winning Pipeline | Secondary MAE | Secondary $R^2$ | Runtime |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Deterministic Baseline** | 42 | 0.6582 | 0.5898 | `ensemble_weighted` | 0.4431 | 0.4578 | 3.66s |
| | 123 | 0.6698 | 0.5369 | `ensemble_weighted` | 0.4137 | 0.4682 | 4.30s |
| | 456 | 0.6434 | 0.5365 | `ensemble_weighted` | 0.4281 | 0.4357 | 4.35s |
| **LLM Model-Only (Ablation)** | 42 | 0.6582 | 0.5898 | `ensemble_weighted` | 0.4431 | 0.4578 | 3.97s |
| | 123 | 0.6698 | 0.5369 | `ensemble_weighted` | 0.4137 | 0.4682 | 3.90s |
| | 456 | 0.6508 | 0.5401 | `ensemble_weighted` | 0.4332 | 0.4282 | 4.44s |
| **Full LLM-Guided Plan** | 42 | 0.6582 | 0.5898 | `ensemble_weighted` | 0.4431 | 0.4578 | 4.07s |
| | 123 | 0.6698 | 0.5369 | `ensemble_weighted` | 0.4137 | 0.4682 | 4.09s |
| | 456 | 0.6508 | 0.5401 | `ensemble_weighted` | 0.4332 | 0.4282 | 4.44s |

#### Aggregate Descriptive Statistics (Wine Quality Red):
- **Deterministic Baseline**: $0.5544 \pm 0.0307$ (Holdout Range: $[0.5365, 0.5898]$, Mean Runtime: $4.10\text{s}$)
- **LLM Model-Only**: $0.5556 \pm 0.0298$ (Holdout Range: $[0.5369, 0.5898]$, Mean Runtime: $4.10\text{s}$)
- **Full LLM-Guided**: $0.5556 \pm 0.0298$ (Holdout Range: $[0.5369, 0.5898]$, Mean Runtime: $4.20\text{s}$)
- **Direction-Adjusted $\Delta$ (LLM vs Deterministic)**: $\mathbf{-0.0012}$ (Deterministic baseline showed $+0.0012$ RMSE advantage)
- **Secondary MAE Mean**: Deterministic: $0.4283 \pm 0.0147$, LLM: $0.4300 \pm 0.0150$ ($\Delta = -0.0017$)

---

## 4. Key Scientific Findings & Analysis

### Finding 1: LLM-Guided Planning Improved Holdout F1 on Adult Census
On Adult Census Income, full LLM-guided planning achieved a mean holdout F1-score of $0.8220 \pm 0.0101$ compared to $0.8148 \pm 0.0187$ for the deterministic baseline ($+0.0072$ improvement). 
- **Mechanism**: The LLM proposed `svm_clf` in its candidate model set (`[gradient_boosting_clf, logistic_regression, random_forest_clf, svm_clf]`).
- On seed 456, `svm_clf` achieved a CV score of $0.8247$ and holdout score of $0.8176$, winning over tree-based ensembles on that partition. In contrast, the deterministic baseline selected an ensemble on seed 456 that scored $0.7961$ on holdout.

### Finding 2: Deterministic Baseline Outperformed LLM on Wine Quality Red
On Wine Quality Red, the deterministic baseline achieved lower holdout RMSE ($0.5544 \pm 0.0307$) than LLM-guided planning ($0.5556 \pm 0.0298$).
- **Mechanism**: The deterministic heuristic candidate pool for small regression datasets includes `knn_reg` alongside `ridge`, `random_forest_reg`, and `gradient_boosting_reg`.
- The LLM did not propose `knn_reg`. On seed 456, `knn_reg` received significant weight ($w=0.30$) in the deterministic ensemble, yielding holdout RMSE $0.5365$ versus LLM's $0.5401$.

### Finding 3: Ablation Demonstrates Model Selection Drives Performance
- Comparing Condition B (`llm_model_only`) and Condition C (`llm_guided`), the holdout means were **identical** on both datasets ($0.8220$ vs $0.8220$ on Adult; $0.5556$ vs $0.5556$ on Wine).
- The feature operations proposed by the LLM (e.g. `CapitalGain_log`, `Age_zscore` on Adult) were validated and executed without error, but under the tested 3-fold CV and Optuna HPO budget, tree-based estimators and scaled SVMs already handle monotonic transformations and scale variance effectively.
- Therefore, the empirical performance difference between deterministic and LLM-guided AutoML was governed predominantly by **candidate model selection** rather than automated feature engineering.

### Finding 4: Absence of Universal Superiority
- The experiments demonstrate that **LLM-guided AutoML is NOT universally superior to deterministic AutoML**.
- Rather, LLMs act as an adaptive heuristic that can identify promising model candidates outside rigid rules (e.g., SVM on Adult), but may also omit models that simple heuristics include (e.g., KNN on Wine).
- The contribution of AutoML-Lens is **safe, registry-constrained, and reproducible pipeline construction**, not a sweeping claim of state-of-the-art general intelligence.

---

## 5. Scope & Limitations

1. **Dataset Scope**: Evaluated on 2 benchmark datasets (UCI Adult Census Income and UCI Wine Quality Red).
2. **Seed Scope**: $N=3$ seeds (`[42, 123, 456]`). While providing initial variance bounds, this sample size represents descriptive empirical evidence rather than broad asymptotic significance.
3. **Registry Boundaries**: Constrained to pre-validated algorithms in `ModelRegistry` and `OPERATION_REGISTRY`. No arbitrary Python code generation or reinforcement learning was employed.
