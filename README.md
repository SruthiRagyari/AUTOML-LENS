# AutoML-Lens

> **An LLM-guided structured AutoML framework with registry-constrained pipeline planning, fold-safe training, hyperparameter optimization, ensemble selection, explainability, and reproducible evaluation.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org)
[![Optuna](https://img.shields.io/badge/Optuna-4.2.0-purple.svg)](https://optuna.org)
[![Tests](https://img.shields.io/badge/Tests-see%20docs%2FTEST__RUN__OUTPUT.txt-blue.svg)](docs/TEST_RUN_OUTPUT.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Problem Statement

Standard Automated Machine Learning (AutoML) tools face fundamental structural bottlenecks:
1. **Combinatorial Inefficiency**: Exhaustive search over high-dimensional algorithm and hyperparameter spaces is computationally prohibitive on constrained hardware.
2. **Rigid Heuristics**: Deterministic rule sets rely on static row/column thresholds that fail to adapt to nuanced data semantics, feature interactions, or domain properties.
3. **Safety & Hallucination Risks in LLMs**: While Large Language Models exhibit strong contextual reasoning, directly prompting an LLM to generate executable Python code introduces critical failure modes: syntax errors, silent data leakage, invalid library calls, and arbitrary code execution vulnerabilities.

---

## 2. Research Motivation: LLMs as a Structured Decision Layer

AutoML-Lens addresses this dilemma by separating **intelligent planning** from **deterministic execution**:
- The LLM (Google Gemini, OpenAI, or a local deterministic fallback) acts strictly as a **structured planner**, outputting a declarative JSON pipeline specification (candidate model recommendations, mathematical feature operations, hyperparameter search priorities, and ensemble strategies).
- The pipeline plan is passed to a deterministic **`PipelinePlanValidator`** that constrains all requests to validated system registries (`ModelRegistry`, `OPERATION_REGISTRY`). Any unsupported model, hazardous transformation, or target-leakage operation is rejected or safely repaired.
- The validated plan is then executed by a **fold-safe AutoML engine**, ensuring that feature scalers, imputers, and transformers are fit strictly inside cross-validation folds.

---

## 3. End-to-End System Architecture

```mermaid
flowchart TD
    User(["👤 User / Client"]) --> Upload["📁 Raw Dataset Upload (.csv, .xlsx)"]
    Upload --> Profiler["📊 Dataset Profiler (types, cardinality, missingness)"]
    Profiler --> Split{"🔀 Deterministic Split"}
    Split --> TrainData["🟢 Training Split (80%)"]
    Split --> HoldoutData["🔴 100% Isolated Holdout (20%)"]

    TrainData --> Planner["🤖 LLM / Fallback Pipeline Planner"]
    Planner --> Validator{"🛡️ PipelinePlanValidator (Registry Constraints)"}
    Validator --> FoldSafe["⚙️ Fold-Safe Preprocessing & Feature Engineering"]
    FoldSafe --> HPO["🎯 Optuna Bayesian HPO & Model Training"]
    HPO --> Ensemble["🤝 Model Fusion & Stacking Optimizer"]
    Ensemble --> CVSelection{"🏆 CV-Only Winner Selection"}
    CVSelection --> BestModel["✨ Selected Best Pipeline / Ensemble"]
    
    BestModel --> HoldoutEval["🎯 Final Holdout Evaluation (Evaluated Once)"]
    HoldoutData --> HoldoutEval

    BestModel --> SHAP["🔍 SHAP Feature Explainability"]
    BestModel --> Inference["🔮 Interactive & Batch Prediction Engine"]
    HoldoutEval --> Reports["📑 Interactive HTML & Markdown Reports"]
```

> **Strict Holdout Discipline**: The 20% holdout test partition is split immediately upon ingestion and remains completely untouched during feature engineering, cross-validation, hyperparameter tuning, and ensemble optimization. It is evaluated strictly once after final model selection.

---

## 4. Key Capabilities

- **Real Ingestion & Automatic Profiling**: Supports CSV and Excel spreadsheets with automatic delimiter detection, type inference (numerical, categorical, datetime, text), missingness auditing, and dataset suitability checks.
- **LLM Pipeline Planning**: Generates structured, schema-validated pipeline plans with provider support for Google Gemini, OpenAI, and a self-contained deterministic fallback (requiring zero API keys or external calls).
- **Registry-Constrained Safety**: Zero arbitrary code execution. Every model is verified against `ModelRegistry` (17 scikit-learn estimators across classification and regression) and feature operations against `OPERATION_REGISTRY`.
- **Fold-Safe Preprocessing**: Imputation, scaling, one-hot encoding, and feature transforms are fit strictly on training folds to completely prevent target leakage and data snooping.
- **Optuna Hyperparameter Optimization**: Bayesian optimization with Tree-structured Parzen Estimator (TPE) sampler over bounded parameter spaces.
- **Model Fusion & Stacking**: Greedy ensemble weight optimization and ridge-stacked meta-learners computed exclusively over out-of-fold (OOF) cross-validation predictions.
- **Model Explainability**: Model-agnostic TreeSHAP and KernelSHAP explanation summaries, beeswarm plots, and feature importance rankings.
- **Real-Time & Batch Prediction**: REST endpoints and interactive UI forms for single-sample inference and batch CSV scoring.
- **Provenance & Reproducibility**: End-to-end provenance tracking including dataset SHA-256 hash, Git commit revision, experiment seed, timestamp, and runtime configurations.

---

## 5. Empirical Evaluation

AutoML-Lens was evaluated under a rigorous multi-seed benchmark comparing three experimental conditions across an identical seed sequence (`[42, 123, 456]`):
1. **Condition A (`deterministic`)**: Deterministic heuristic AutoML baseline.
2. **Condition B (`llm_model_only` ablation)**: LLM candidate model selection only; default preprocessing and zero feature operations.
3. **Condition C (`llm_guided`)**: Full LLM pipeline planning (candidate models, feature operations, HPO focus, ensemble configuration).

### Benchmark Protocol

| Parameter | Value |
| :--- | :--- |
| Seeds | `[42, 123, 456]` |
| Cross-validation folds | 3 |
| Optuna trials (`n_trials`) | **0** (model/pipeline selection only; no HPO search) |
| Adult Census — total rows used | 1,000 (800 train / 200 holdout) |
| Wine Quality Red — total rows used | 800 (640 train / 160 holdout) |
| Holdout policy | Isolated post-ingestion; evaluated exactly once post-selection |
| Statistical statement | Descriptive multi-seed comparison only (N=3 seeds per condition). No inferential p-values claimed. |
| Std convention | Population std (ddof=0) |

> **Note**: `n_trials=0` means this benchmark measures LLM-guided model and pipeline selection, **not** Optuna hyperparameter search. HPO is a separately implemented capability. For a separate fast-mode comparison between Gemini and the deterministic fallback, see [`docs/LLM_EVALUATION.md`](docs/LLM_EVALUATION.md), where Gemini did not outperform the fallback on average.

### Measured Multi-Seed Results

All values from `benchmarks/results/benchmark_summary.json`. Std = population std (ddof=0, N=3).

| Dataset | Task | Primary Metric | Deterministic Baseline | LLM Model-Only (Ablation) | Full LLM-Guided Plan | Difference (within seed variability) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **UCI Adult Census** | Classification | `f1_weighted` ↑ | $0.8148 \pm 0.0181$ | $0.8220 \pm 0.0173$ | $0.8220 \pm 0.0173$ | $+0.0072$ |
| **UCI Wine Quality Red** | Regression | `rmse` ↓ | $0.5544 \pm 0.0249$ | $0.5556 \pm 0.0268$ | $0.5556 \pm 0.0268$ | $-0.0012$ (det. better) |

**Per-seed holdout scores (source: `benchmark_summary.json`):**

| Dataset | Condition | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Adult Census | Deterministic | 0.7937 | 0.8379 | 0.8127 | 0.8148 ± 0.0181 |
| Adult Census | LLM-Guided | 0.7980 | 0.8379 | 0.8300 | 0.8220 ± 0.0173 |
| Wine Quality | Deterministic | 0.5461 | 0.5289 | 0.5881 | 0.5544 ± 0.0249 |
| Wine Quality | LLM-Guided | 0.5461 | 0.5285 | 0.5921 | 0.5556 ± 0.0268 |

### Key Takeaways

1. **Observed difference on Adult Census**: The difference (+0.0072 F1) is smaller than the across-seed standard deviation (±0.0181 for deterministic, ±0.0173 for LLM-guided) and should **not** be interpreted as significant evidence that LLM-guided planning is better given N=3 seeds and a 200-row holdout set.
2. **Deterministic baseline is competitive on Wine Quality**: The deterministic baseline achieved slightly lower holdout RMSE (−0.0012). On seed 456 the deterministic run scored 0.5881 vs 0.5921 for the LLM-guided run (a 0.004 difference on one seed); the deterministic candidate set included `knn_reg`, which received weight in the winning `ensemble_weighted`, while the LLM-guided candidate set did not include `knn_reg`.
3. **Ablation Findings**: Conditions B and C achieved identical aggregate holdout scores ($0.8220$ on Adult, $0.5556$ on Wine), showing that under this budget, performance deltas were governed primarily by **model candidate selection** rather than automated feature engineering.
4. **Empirical Honesty**: LLM-guided AutoML is **not** universally superior to deterministic AutoML. LLMs act as an adaptive heuristic that can identify competitive non-standard candidates while occasionally omitting useful heuristics. With N=3 seeds and no p-values, these are descriptive observations only.

For full details, see [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md) and [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md).

---

## 6. Limitations & Scope Boundaries

- **Benchmark Scope**: Evaluated on 2 datasets across N=3 seeds; findings are descriptive empirical evidence, not asymptotically universal claims.
- **Model Registry Boundaries**: Constrained strictly to algorithms supported in `ModelRegistry` (scikit-learn linear models, tree ensembles, SVMs, KNN, Naive Bayes).
- **Safety Boundaries**: The system executes zero arbitrary Python code. All transformations are predefined declarative functions.
- **Explicit Exclusions**: This framework does **not** implement reinforcement learning (RL), automated neural architecture search (NAS), multimodal AutoML, or autonomous code generation.

---

## 7. Quick Start Guide

### Prerequisites
- **Python**: 3.11+
- **Node.js**: 18+
- *(Optional)* Google Gemini or OpenAI API key (the system runs out of the box with the deterministic fallback without keys).

### Installation & Execution

#### 1. Clone the Repository
```bash
git clone https://github.com/SruthiRagyari/AUTOML-LENS.git
cd AUTOML-LENS
```

#### 2. Backend Setup
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

# Run FastAPI backend:
python -m uvicorn app.main:app --reload --port 8000
```

#### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```

The application is available at:
- **Frontend UI**: [http://localhost:5173](http://localhost:5173)
- **FastAPI Backend Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Endpoint**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 8. Verification & Test Suite

The system includes a test suite covering holdout isolation, provenance, schema validation, metric direction contracts, Optuna HPO, ensembles, and API routes.
See [`docs/TEST_RUN_OUTPUT.txt`](docs/TEST_RUN_OUTPUT.txt) for the latest verified run result.

```bash
# Run backend test suite:
python -m pytest tests -q

# Run frontend production build:
cd frontend
npm run build
```

---

## 9. Documentation Index

- [System Architecture & Flow Diagram](docs/ARCHITECTURE_DIAGRAM.md)
- [Comprehensive Project Documentation](docs/PROJECT_DOCUMENTATION.md)
- [Research Write-up](docs/RESEARCH_PAPER_DRAFT.md)
- [Final Multi-Seed Empirical Results](docs/FINAL_RESULTS.md)
- [Presentation Outline](docs/PRESENTATION_OUTLINE.md)
- [Demo Script](docs/DEMO_SCRIPT.md)
- [Standalone HTML Research Report](docs/BENCHMARK_REPORT.html)

---

## 10. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
