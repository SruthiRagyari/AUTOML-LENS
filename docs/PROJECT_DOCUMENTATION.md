# AutoML-Lens: Comprehensive Technical & Research Documentation

## 1. Introduction
**AutoML-Lens** is an open-source, full-stack Automated Machine Learning (AutoML) system featuring structured Large Language Model (LLM) pipeline planning, registry-constrained execution, fold-safe preprocessing, Bayesian hyperparameter optimization, out-of-fold model fusion, post-selection holdout evaluation, model explainability, and reproducible experiment provenance.

Unlike black-box AutoML frameworks that rely solely on brute-force search over rigid candidate spaces, AutoML-Lens utilizes LLMs (Google Gemini, OpenAI, or a local deterministic rule-based engine) as an intelligent decision layer. Crucially, rather than generating unconstrained code, the LLM produces a declarative pipeline specification that is strictly validated against system registries before execution.

---

## 2. Problem Statement
Traditional AutoML systems suffer from three primary challenges:
1. **Computational Inefficiency**: Grid or random searches evaluate thousands of candidate combinations, requiring vast computing power and often failing to converge under realistic resource bounds.
2. **Heuristic Rigidity**: Heuristic-driven AutoML relies on fixed thresholds (e.g., row/column counts) that cannot reason about feature semantics, column relationships, or domain context.
3. **LLM Hallucination and Safety Hazards**: Directly asking LLMs to generate Python training scripts leads to runtime failures, invalid package imports, catastrophic data leakage (fitting scalers on test data), and security vulnerabilities from arbitrary code execution.

---

## 3. Objectives
1. **Safe LLM Integration**: Incorporate LLMs as structured pipeline planners producing schema-enforced declarative plans without executing arbitrary code.
2. **Registry-Constrained Execution**: Validate all LLM-proposed models and transformations against strict registries (`ModelRegistry`, `OPERATION_REGISTRY`).
3. **Zero Data Leakage**: Implement fold-safe preprocessing and an untouchable post-selection holdout partition.
4. **Reproducibility & Provenance**: Ensure every run captures git commit revisions, dataset SHA-256 hashes, random seeds, and trial parameters.
5. **Rigorous Empirical Benchmarking**: Evaluate LLM-guided AutoML against deterministic AutoML across multiple reproducible seeds on standard tabular datasets.

---

## 4. System Architecture
AutoML-Lens follows a modern, decoupled client-server architecture:
- **Client Tier**: React 18 Single-Page Application (SPA) built with Vite, styled with custom CSS variables, and rendered via modular component architecture.
- **API Tier**: High-performance FastAPI backend with asynchronous request routing, Pydantic v2 validation, and real-time training progress polling.
- **Storage Tier**: SQLite database managed via SQLAlchemy ORM for experiment tracking; local disk storage for trained model binaries (`.joblib`), dataset uploads, and SHAP artifacts.
- **Core Engine Tier**: Scikit-learn, Optuna, and SHAP integrated via decoupled service modules.

---

## 5. Backend Architecture
The backend is structured into modular layers under `backend/app/`:
- `api/`: REST routing (`datasets.py`, `experiments.py`, `health.py`).
- `core/`: Application settings (`config.py`) and database schemas (`database.py`).
- `llm/`: Provider abstractions (`manager.py`, `gemini.py`, `openai.py`, `fallback.py`).
- `services/`: Core ML operations:
  - `profiler.py`: Tabular statistics and type inference.
  - `pipeline_planner.py`: LLM pipeline planner and `PipelinePlanValidator`.
  - `preprocessor.py` & `fold_safe.py`: Leakage-free column transformations.
  - `feature_engineer.py` & `feature_operations.py`: Controlled feature operations.
  - `model_registry.py`: Candidate model catalog and hyperparameter spaces.
  - `trainer.py`: Cross-validation loops and estimator fitting.
  - `optimizer.py`: Optuna Bayesian optimization studies.
  - `ensemble.py`: Out-of-fold (OOF) stacking and greedy weighted averaging.
  - `evaluator.py`: Sklearn metrics and contract verification.
  - `explainer.py`: Model explainability (SHAP).
  - `predictor.py`: Single and batch inference.
  - `reporter.py`: Self-contained HTML and Markdown report generation.
  - `benchmarker.py`: Multi-seed research benchmarking and statistical aggregation.

---

## 6. Frontend Architecture
The frontend is built in React 18 under `frontend/src/`:
- `pages/`: Primary application views:
  - `Dashboard.jsx`: Experiment history, metrics overview, and quick-action links.
  - `NewExperiment.jsx`: Drag-and-drop file upload, instant profiling preview, and experiment configuration.
  - `Experiment.jsx`: 7-step interactive workflow (Dataset, AI Analysis, Preprocessing, Models, Optimization, Results, Explainability).
- `components/`: Modular UI widgets (Optuna trial history charts, confusion matrices, pipeline plan inspectors, multi-seed benchmark tables).
- `api/`: Native fetch-based API client for REST interaction.

---

## 7. Dataset Processing & Profiling
The `DatasetProfiler` computes comprehensive descriptive statistics across uploaded CSV and Excel datasets:
- **Shape & Memory**: Row count, column count, memory footprint.
- **Inferred Types**: Numerical (`int64`, `float64`), Categorical (`object`, low-cardinality strings), Datetime, and Text.
- **Missing Value Audit**: Null counts and missing percentages per feature.
- **Cardinality & Distribution**: Unique value counts, min, max, mean, standard deviation, and quantiles.
- **Readiness Checks**: Identifies constant columns, ID-like columns, and target suitability.

---

## 8. LLM Pipeline Planner
The LLM Pipeline Planner (`backend/app/services/pipeline_planner.py`) receives a serialized statistical summary of the training partition and outputs an `AutoMLPipelinePlan` object containing:
- `problem_type`: Inferred task (`classification` or `regression`).
- `primary_metric`: Selection metric (e.g., `f1_weighted`, `neg_root_mean_squared_error`).
- `candidate_models`: Recommended algorithm IDs.
- `preprocessing_strategy`: Imputation methods, scaling techniques, encoding rules.
- `feature_operations`: Declarative feature engineering transformations.
- `hyperparameter_strategy`: Suggested focus parameters and trial budgets.
- `ensemble_strategy`: Ensembling methods (`weighted_average`, `soft_voting`, `stacking`).
- `reasoning`: Technical justification for the chosen design.

The planner supports three provider backends: Google Gemini (`gemini-flash-lite-latest`), OpenAI (`gpt-4o-mini`), and a deterministic rule-based fallback that generates reproducible plans without network calls or API keys.

---

## 9. Pipeline Validation & Safety Constraints
To guarantee safety and execution stability, `PipelinePlanValidator.validate_and_sanitize()` inspects every plan:
1. **Registry Adherence**: Only model IDs registered in `ModelRegistry` are permitted.
2. **Operation Verification**: Feature operations must exist in `OPERATION_REGISTRY`.
3. **Target Leakage Prevention**: Operations targeting the label column are rejected immediately.
4. **Task Compatibility**: Classification metrics and algorithms cannot be assigned to regression tasks, and vice versa.
5. **Autonomous Repair**: Malformed parameters are repaired using deterministic defaults, preventing pipeline crashes.

---

## 10. Controlled Feature Engineering
Feature engineering is managed by `FeatureEngineer` and `OPERATION_REGISTRY`:
- **Mathematical Transforms**: `log`, `sqrt`, `square`, `abs`, `reciprocal`.
- **Scaling Transforms**: `zscore`, `minmax`, `robust_scale`.
- **Interaction Transforms**: Pairwise products, ratios, and differences between numerical columns.
- **Categorical Frequency Encoding**: Frequency count representations of high-cardinality categories.
- **Safety**: Division-by-zero guards, NaN clipping, and variance validation ensure numeric stability.

---

## 11. Fold-Safe Preprocessing
A primary cause of optimistic evaluation bias in academic ML is data snooping. AutoML-Lens enforces **fold-safe execution**:
- Preprocessing pipelines (`ColumnTransformer`) and feature transformers are instantiated per fold.
- Statistics (mean, median, variance, category vocabularies) are fit **exclusively** on the $K-1$ training folds.
- Validation folds and the holdout split are transformed using parameters learned solely on the training partition.

---

## 12. Model Registry
The `ModelRegistry` maintains a catalog of 17 vetted scikit-learn algorithms:
- **Classifiers (8)**: Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting, Support Vector Machine (SVC), K-Nearest Neighbors, Decision Tree, Gaussian Naive Bayes.
- **Regressors (9)**: Ridge Regression, Lasso, ElasticNet, Random Forest, Gradient Boosting, HistGradientBoosting, Support Vector Regression (SVR), K-Nearest Neighbors, Decision Tree Regressor.
Each algorithm entry defines default parameters, search spaces for Optuna, and task suitability tags.

---

## 13. Hyperparameter Optimization (Optuna)
Hyperparameter tuning is powered by Optuna (`backend/app/services/optimizer.py`):
- **Sampler**: Tree-structured Parzen Estimator (`TPESampler`) seeded with deterministic random states.
- **Objective**: Evaluates cross-validation performance of candidate models across defined parameter ranges.
- **Pruning**: Median pruner to terminate unpromising trials early.
- **Isolation**: Each study runs inside an isolated trial context to prevent memory leaks.

---

## 14. Model Fusion & Ensemble Optimization
The ensemble subsystem (`backend/app/services/ensemble.py`) combines diverse candidate models:
- **Out-of-Fold (OOF) Prediction**: Predictions on cross-validation validation folds are recorded without test leakage.
- **Greedy Weighted Averaging**: Sequential forward selection of model weights optimizing CV score via bounded Nelder-Mead or coordinate descent.
- **Stacking Meta-Learner**: Fits a regularized Ridge or Logistic Regression meta-model on OOF prediction matrices.
- **Selection Decision**: An ensemble is selected as the overall winner only if its CV performance strictly outperforms the best individual candidate model.

---

## 15. Model Explainability
Model interpretability is implemented via SHAP (`backend/app/services/explainer.py`):
- **TreeSHAP**: Fast, exact Shapley value computation for tree ensembles (Random Forest, Gradient Boosting).
- **KernelSHAP**: Model-agnostic approximation for linear models and kernel estimators.
- **Outputs**: Mean absolute SHAP values, feature importance bar charts, beeswarm distributions, and natural language summary interpretations.

---

## 16. Prediction & Inference Engine
The `Predictor` service provides production-ready inference:
- **Input Validation**: Schema validation checking data types, required features, and missing column handling.
- **Transformation Pipeline**: Automatically applies the winning pipeline's fitted preprocessor and feature transformers.
- **Inference Modes**:
  - Single sample real-time prediction with class probability distributions.
  - High-throughput batch prediction from uploaded CSV files.

---

## 17. Scientific Provenance & Reproducibility
Every experiment records complete provenance metadata stored in the database:
- **Git Commit Hash**: Exact repository commit revision.
- **Dataset Hash**: SHA-256 cryptographic digest of raw ingested data.
- **Random Seed**: Master seed propagated to splitters, samplers, and estimators.
- **Metric Contracts**: Formal declaration of metric names and optimization directions (`maximize` vs `minimize`).
- **Plan Provenance**: Distinguishes LLM-proposed vs validated vs executed pipelines.

---

## 18. Benchmark Methodology
The benchmark suite (`scripts/run_benchmarks.py`, `backend/app/services/benchmarker.py`) evaluates three conditions:
1. `deterministic`: Rule-based baseline.
2. `llm_model_only`: LLM model selection ablation (no feature operations).
3. `llm_guided`: Full LLM pipeline planning.
Each condition is evaluated over identical seeds (`[42, 123, 456]`) on UCI Adult Census and UCI Wine Quality Red datasets under identical 3-fold CV and Optuna trial budgets.

---

## 19. Experimental Results Summary
- **UCI Adult Census (`f1_weighted` $\uparrow$)**:
  - Deterministic: $0.8148 \pm 0.0187$
  - LLM Model-Only: $0.8220 \pm 0.0101$
  - Full LLM-Guided: $0.8220 \pm 0.0101$
  - Delta: $+0.0072$ (LLM advantage driven by `svm_clf` selection on seed 456).
- **UCI Wine Quality Red (`rmse` $\downarrow$)**:
  - Deterministic: $0.5544 \pm 0.0307$
  - LLM Model-Only: $0.5556 \pm 0.0298$
  - Full LLM-Guided: $0.5556 \pm 0.0298$
  - Delta: $-0.0012$ (Deterministic advantage driven by `knn_reg` in ensemble on seed 456).
- **Ablation Finding**: Identical holdout means between Model-Only and Full LLM conditions demonstrate that model candidate selection, not feature transforms, drove performance variation under the evaluated budget.

---

## 20. Scope & Limitations
- **Empirical Scale**: 2 benchmark datasets across $N=3$ seeds; provides descriptive evidence rather than universal statistical significance.
- **Registry Constraints**: Restricted to scikit-learn models and mathematical transforms.
- **No Reinforcement Learning / NAS**: Does not implement dynamic RL agents or neural architecture search.
- **No Arbitrary Code Execution**: Code generation is deliberately disbarred for safety and reliability.

---

## 21. Future Work
- Expansion to time-series and multi-table relational benchmarks.
- Automated GPU-accelerated gradient boosting backends (LightGBM, XGBoost, CatBoost).
- Multi-objective Pareto frontier optimization (accuracy vs latency vs memory footprint).
- Cross-dataset meta-learning memory to initialize LLM prompts from prior runs.
