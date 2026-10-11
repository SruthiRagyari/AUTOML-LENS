# AutoML-Lens: Final Empirical Benchmark & Evaluation Results

This document presents the definitive empirical results from the multi-seed evaluation of **AutoML-Lens**. All values are sourced directly from [`benchmarks/results/benchmark_results.json`](../benchmarks/results/benchmark_results.json) (per-seed runs) and [`benchmarks/results/benchmark_summary.json`](../benchmarks/results/benchmark_summary.json) (aggregated statistics). No values in this document were manually estimated.

---

## 1. Experimental Protocol & Controls

- **Datasets**:
  1. **UCI Adult Census Income (`adult_census_income`)**: Binary classification, 14 features (mixed numerical/categorical), target: `income`, evaluated on `f1_weighted` (higher is better). Used **1,000 rows total** (800 train / 200 holdout, stratified split).
  2. **UCI Wine Quality Red (`winequality_red`)**: Tabular regression, 11 numerical physicochemical features, target: `quality`, evaluated on `rmse` (lower is better). Used **800 rows total** (640 train / 160 holdout).
- **Seeds**: `[42, 123, 456]` across all conditions (N=3 seeds per condition, 18 total runs).
- **Holdout Discipline**: 20% holdout strictly isolated post-ingestion. Never seen by CV folds, Optuna trials, or ensemble weight optimizers. Evaluated exactly once post-selection.
- **Cross-Validation**: 3-fold stratified cross-validation on the 80% training split.
- **Optuna Trials**: `n_trials=0` — this benchmark measures model/pipeline **selection**, not HPO search.
- **Std Convention**: Population std (ddof=0, verified against per-seed values with NumPy).
- **Registry Constraints**: Zero arbitrary code execution; all candidate models and feature operations constrained to pre-validated system registries.

---

## 2. Experimental Conditions

1. **Condition A (`deterministic`)**: Standard heuristic AutoML baseline without LLM assistance.
2. **Condition B (`llm_model_only` ablation)**: LLM selects candidate models from registry; default preprocessing and zero feature operations.
3. **Condition C (`llm_guided`)**: Full LLM pipeline planning (validated candidate models, feature operations, HPO focus, ensemble configuration).

---

## 3. Measured Results — Per Seed

Source: `benchmarks/results/benchmark_results.json`.

### A. UCI Adult Census Income — `f1_weighted` ↑ (higher is better)

| Condition | Seed | CV Score | Holdout F1 | Winning Pipeline | Runtime |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Deterministic** | 42 | 0.8423 | 0.7937 | `ensemble_soft_voting` | 11.91s |
| | 123 | 0.8363 | 0.8379 | `ensemble_stacking` | 18.52s |
| | 456 | 0.8342 | 0.8127 | `ensemble_soft_voting` | 14.49s |
| **LLM-Guided** | 42 | 0.8430 | 0.7980 | `ensemble_soft_voting` | 15.79s |
| | 123 | 0.8347 | 0.8379 | `ensemble_stacking` | 19.11s |
| | 456 | 0.8408 | 0.8300 | `svm_clf` | 15.79s |

#### Aggregate Descriptive Statistics (Adult Census):
- **Deterministic Baseline**: $0.8148 \pm 0.0181$ (Holdout Range: $[0.7937, 0.8379]$, Mean Runtime: $14.97\text{s}$)
- **LLM-Guided**: $0.8220 \pm 0.0173$ (Holdout Range: $[0.7980, 0.8379]$, Mean Runtime: $16.90\text{s}$)
- **Direction-Adjusted $\Delta$ (LLM vs Deterministic)**: $\mathbf{+0.0072}$

> **Interpretive note**: The difference (+0.0072) is **smaller than the across-seed standard deviation** (±0.0181 for deterministic, ±0.0173 for LLM-guided) and should not be read as a significant difference. With N=3 seeds and a 200-row holdout, this is a descriptive observation only — no p-values are claimed.

---

### B. UCI Wine Quality Red — `rmse` ↓ (lower is better)

| Condition | Seed | CV Score | Holdout RMSE | Winning Pipeline | Runtime |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Deterministic** | 42 | -0.6152 | 0.5461 | `ensemble_weighted` | 6.33s |
| | 123 | -0.6091 | 0.5289 | `ensemble_weighted` | 7.02s |
| | 456 | -0.5909 | 0.5881 | `ensemble_weighted` | 5.69s |
| **LLM-Guided** | 42 | -0.6152 | 0.5461 | `ensemble_weighted` | 6.07s |
| | 123 | -0.6092 | 0.5285 | `ensemble_weighted` | 6.92s |
| | 456 | -0.5988 | 0.5921 | `ensemble_weighted` | 5.62s |

#### Aggregate Descriptive Statistics (Wine Quality Red):
- **Deterministic Baseline**: $0.5544 \pm 0.0249$ (Holdout Range: $[0.5289, 0.5881]$, Mean Runtime: $6.35\text{s}$)
- **LLM-Guided**: $0.5556 \pm 0.0268$ (Holdout Range: $[0.5285, 0.5921]$, Mean Runtime: $6.21\text{s}$)
- **Direction-Adjusted $\Delta$ (LLM vs Deterministic)**: $\mathbf{-0.0012}$ (Deterministic baseline was $+0.0012$ RMSE lower)

---

## 4. Summary Table

| Dataset | Task | Deterministic | LLM-Guided | Difference |
| :--- | :--- | :---: | :---: | :---: |
| UCI Adult Census | `f1_weighted` ↑ | $0.8148 \pm 0.0181$ | $0.8220 \pm 0.0173$ | $+0.0072$ |
| UCI Wine Quality Red | `rmse` ↓ | $0.5544 \pm 0.0249$ | $0.5556 \pm 0.0268$ | $-0.0012$ (within seed variability) |

Std = population std (ddof=0), N=3. No inferential statistics are claimed.

---

## 5. Key Scientific Findings

### Finding 1: Observed difference on Adult Census is within seed variability
On Adult Census, full LLM-guided planning achieved a mean holdout F1 of $0.8220 \pm 0.0173$ versus $0.8148 \pm 0.0181$ for the deterministic baseline (+0.0072 observed difference).

- The +0.0072 difference is smaller than the across-seed std on both conditions; it cannot be interpreted as a significant difference with N=3 seeds.
- The LLM proposed `svm_clf` on seed 456, which achieved holdout F1 of 0.8300 (vs. deterministic's 0.8127 on the same seed). This illustrates how LLM candidate selection can identify non-standard models for a specific partition, but seed-to-seed variance dominates.

### Finding 2: Deterministic baseline achieved lower holdout RMSE on Wine Quality Red
On Wine Quality Red, the deterministic baseline achieved lower holdout RMSE than LLM-guided planning (difference: +0.0012 in favour of deterministic).

- The deterministic heuristic candidate pool for small regression datasets includes `knn_reg` alongside `ridge`, `random_forest_reg`, and `gradient_boosting_reg`.
- The LLM did not propose `knn_reg`. On seed 456, `knn_reg` received weight 0.34 in the deterministic `ensemble_weighted` pipeline, contributing to its lower RMSE on that seed.

### Finding 3: Ablation shows model selection governs performance
The `llm_model_only` (Condition B) and `llm_guided` (Condition C) conditions achieved **identical** aggregate holdout scores on both datasets ($0.8220$ Adult; $0.5556$ Wine). The validated LLM-proposed feature operations did not produce measurable additional benefit at this budget (n_trials=0, 3-fold CV), indicating that performance was governed primarily by **candidate model selection** rather than automated feature engineering.

### Finding 4: LLM-guided AutoML is not universally superior
The experiments demonstrate that LLM-guided AutoML is **not universally superior** to deterministic AutoML. LLMs act as an adaptive heuristic that can identify competitive non-standard candidates but may also omit models that simple heuristics include. The contribution of AutoML-Lens is **safe, registry-constrained, and reproducible pipeline planning**, not a claim of universal superiority.

---

## 6. Scope & Limitations

1. **Dataset Scope**: 2 benchmark datasets (UCI Adult Census Income, UCI Wine Quality Red).
2. **Seed Scope**: N=3 seeds (`[42, 123, 456]`). Descriptive empirical evidence only.
3. **HPO Budget**: n_trials=0 — no Optuna hyperparameter search was performed during this benchmark. The measured differences reflect model selection, not optimised hyperparameters.
4. **Registry Boundaries**: Constrained to pre-validated algorithms in `ModelRegistry` and `OPERATION_REGISTRY`.
