# AutoML-Lens

> **LLM-guided AutoML platform**

AutoML-Lens is a full-stack, LLM-guided AutoML platform: upload a raw tabular dataset and it profiles the data, runs LLM-guided problem analysis with Google Gemini, OpenAI, or a deterministic fallback, selects and tunes models with leakage-safe cross-validated selection, explains the winning model with SHAP, and exports a self-contained HTML report - from a React UI backed by FastAPI.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)](https://scikit-learn.org)
[![Optuna](https://img.shields.io/badge/Optuna-4.2.0-purple)](https://optuna.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

AutoML-Lens is a full-stack web application that implements an LLM-integrated AutoML pipeline. It takes a raw tabular dataset and produces a trained, evaluated, and explainable machine learning model — fully automated with optional AI guidance from Google Gemini or OpenAI.

### Key Capabilities

| Feature | Details |
|---------|---------|
| **Dataset Profiling** | Per-column statistics, type inference, missing value analysis, duplicate detection |
| **LLM Analysis** | Gemini/OpenAI-powered dataset understanding with deterministic fallback |
| **Preprocessing** | sklearn ColumnTransformer: imputation, scaling, one-hot encoding, TF-IDF |
| **Model Selection** | 8 classifiers + 9 regressors in configurable registry |
| **Optimization** | Optuna Bayesian hyperparameter search with cross-validation |
| **Evaluation** | Real sklearn metrics: accuracy, F1, ROC-AUC, RMSE, R², etc. |
| **Explainability** | SHAP values, feature importances, coefficients |
| **Predictions** | Single + batch prediction with confidence scores |
| **Reports** | Self-contained HTML experiment reports |
| **LLM Assistant** | AI chatbot with experiment context awareness |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- (Optional) Google Gemini API key or OpenAI API key

### 1. Clone & Setup

```bash
git clone https://github.com/SruthiRagyari/AUTOML-LENS.git
cd automl-lens
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env to add your API key (optional — fallback mode works without it)
python -m uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 4. Open App

Visit **http://localhost:5173** in your browser.

---

## Demo

1. Go to **Dashboard** → **New Experiment**
2. Upload `demo_data/classification.csv` (customer churn) or `demo_data/regression.csv` (house prices)
3. Target is auto-suggested (`Churn` or `Price`)
4. Select **Fast Demo** mode (3 models, 5 Optuna trials, 3-fold CV)
5. Click **Create Experiment** → **Profile** → **AI Analysis** → **Start Training**
6. View model comparison, SHAP feature importance, make predictions, download report

---

## Project Structure

```
automl-lens/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routes (datasets, experiments, chat)
│   │   ├── core/           # Config, database, security
│   │   ├── llm/            # LLM providers (Gemini, OpenAI, Fallback)
│   │   ├── models/         # Pydantic schemas
│   │   ├── services/       # ML pipeline services (fold_safe, trainer, optimizer, etc.)
│   │   └── utils/          # File utilities, validators
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/     # Layout, Charts, Common UI
│       ├── pages/          # Landing, Dashboard, Experiment, History, etc.
│       └── services/       # API client
├── benchmarks/             # Benchmark runner and datasets (adult, wine-quality)
├── demo_data/              # Sample CSV datasets (classification, regression)
├── docs/                   # Documentation and audit reports
├── scripts/                # Utility scripts
├── storage/                # Datasets, models, reports, predictions
├── tests/                  # pytest test suite (223 tests)
└── README.md
```

---

## Hyperparameter Optimization (Optuna)

Optuna runs on the models that survive LLM-recommendation validation, using
each model's own search space. Key guarantees:

**Explicit metric contract.** `metric → sklearn scoring → Optuna direction` is
resolved by `evaluator.get_metric_spec()` and persisted. sklearn negates loss
scorers, so the *study* maximizes the negated value while the *reported* metric
keeps its natural meaning (`rmse` → study `maximize`, reported `minimize`).

**No holdout leakage.** The train/test split, feature engineering and
preprocessing all happen *before* Optuna is called. `OptunaOptimizer` only ever
receives `X_train`/`y_train`, so CV folds are drawn exclusively from training
rows. The winner is refit on the training split with the real best parameters
and the holdout is scored exactly once.

**Deterministic.** The TPE sampler is seeded (`OPTUNA_SEED = 42`), so re-running
an experiment replays the same search.

**Honest provenance.** Every trained model persists an `optimization` block:

| Field | Meaning |
|---|---|
| `metric`, `scoring`, `direction`, `raw_direction` | the metric contract |
| `n_trials_requested` / `n_trials_completed` / `n_trials_failed` | real trial counts |
| `n_folds_requested` / `n_folds_used` / `cv_strategy` | folds actually used |
| `seed` | sampler seed for reproducibility |
| `best_cv_score`, `best_params` | best CV result (`None` if all trials failed) |
| `status` | `COMPLETED` / `PARTIAL` / `FAILED` / `SKIPPED` |
| `error` | the real exception message, verbatim |

Failures are never masked: a trial that raises is recorded with its real error
and pruned; if *every* trial fails, `best_cv_score` stays `None` (never `0.0`)
and the model keeps its baseline. A search that raises is caught per model, so
one broken model never aborts the run. Fold counts are clamped to what the data
supports (with a recorded `fold_note`) instead of crashing.

`fast_demo` deliberately shrinks the budget to 5 trials × 3 folds; the API
reports this as `optimization_budget.budget_reduced = true` rather than
presenting a short run as a full one.

---

## Model Selection & Evaluation Protocol

The winner is chosen from **training/CV evidence only**. The holdout never
decides which model wins — otherwise the test score leaks into selection and
the reported result is optimistically biased.

```
upload → profile → LLM analysis → feature engineering → preprocessing
      → [train/test split FIRST]
      → Optuna CV tuning (train rows only)
      → SELECT WINNER by CV score
      → refit winner on the training split with best params
      → evaluate ONCE on the untouched holdout
      → report final holdout metrics
```

**Selection evidence**, in priority order per model:
1. `optimization.best_cv_score` — Optuna CV mean for the tuned model
2. `cv_scores` mean — CV of the final refit model
3. no CV evidence → the model is **not selectable** (never given a fake score)

**Persisted `selection` block** (on `/train`, `/results`, and
`experiment.model_selection_json`):

| Field | Meaning |
|---|---|
| `selection_metric`, `selection_scoring`, `selection_direction` | the ranking contract |
| `selection_raw_direction` | direction of the reported metric (`minimize` for RMSE) |
| `evidence_source` | `cross_validation_on_training_split` |
| `selected_model`, `selected_model_display_name` | the winner |
| `selected_model_cv_score`, `selected_model_cv_folds` | the CV score that decided it |
| `final_holdout_metrics` | holdout metrics, computed once **after** selection |
| `holdout_used_for_selection` | always `false` |
| `candidates_considered[]` | every model with its status and CV score |

`best_score` / `best_metric` remain the **final holdout** evaluation of the
already-selected winner — an unbiased estimate, not a selection artifact.
`selection_cv_score` is reported alongside so the two are never conflated.

### Deterministic tie-breaking

Equal CV means are common (small datasets often produce identical scores), and
resolving them by candidate order would make the winner an accident of the
registry rather than a decision. Ties are therefore broken by an explicit,
persisted cascade:

| # | Criterion | Rationale |
|---|---|---|
| 1 | **Best CV score according to metric direction** | primary criterion, unchanged |
| 2 | Lowest CV fold **standard deviation** | same mean, but a tighter spread across folds is the less fold-dependent result |
| 3 | **Most CV folds** evaluated | more evidence behind the same mean |
| 4 | Lexicographically smallest `model_name` | fully deterministic, independent of input order |

The ranking always operates on the **sklearn scorer value**, which is
uniformly "higher is better" because `neg_*` losses are already pre-negated.
So for a raw loss such as `root_mean_squared_error`, *lower is better* — and
the stored score `neg_root_mean_squared_error = -0.61` correctly outranks
`-0.66`. Selecting "the highest score" is therefore correct for every metric,
but the persisted wording says **best score according to metric direction** so
it cannot be misread as favouring large losses.

Scores are treated as **tied** only when they agree to within
`rel_tol=1e-9` / `abs_tol=1e-12` — the numerical resolution of the CV
arithmetic. This is *not* a statistical-significance claim: no confidence
interval or significance test is computed. A difference above the tolerance is
simply ranked first; a difference inside it is declared indistinguishable rather
than resolved by pretending to more precision than the scores support.

Every criterion uses training-split evidence only. **Holdout metrics are never
consulted, not even to break a tie.** A model with fewer than two CV folds has
no spread and is *not* given a fabricated `0.0`.

The outcome is persisted under `selection.tie_break`:

```json
{
  "rule": "1) highest CV score; 2) ...",
  "tolerance": {"rel_tol": 1e-9, "abs_tol": 1e-12},
  "tie_detected": true,
  "tied_models": ["hist_gradient_boosting_clf", "random_forest_clf"],
  "tied_cv_scores": {"hist_gradient_boosting_clf": 0.8675, ...},
  "resolved_by": "cv_fold_std",
  "winner": "hist_gradient_boosting_clf",
  "tie_broken_by_holdout": false
}
```

`candidates_considered[]` also carries each model's real `cv_fold_std` and
`cv_fold_count`.

---

## Training Runtime & Real Progress

Training is genuinely CPU-bound: a single SVM fit on ~39k rows can take longer
than every other model combined. The pipeline therefore runs **off the asyncio
event loop** (`asyncio.to_thread`), so the API keeps serving requests while a
long job is in flight.

`GET /api/experiments/{id}/progress` reports **observed state only**:

| Field | Meaning |
|---|---|
| `status` | `queued` / `running` / `completed` / `failed` |
| `phase` | `preparing`, `training`, `optimizing`, `refining`, `scoring`, `selecting`, `persisting` |
| `current_model`, `current_model_display_name`, `current_model_index` | the model actually running |
| `current_trial` | the Optuna trial actually being evaluated |
| `models_total` / `models_completed` / `models_failed` / `models_remaining` | exact counts |
| `models[]` | per-model `status`, `phase`, `error`, trial counts, `elapsed_seconds` |
| `elapsed_seconds` | measured, not estimated |

What is deliberately **absent**:

* **no percentage complete** — the work is not linearly divisible, so any
  percentage would be invented;
* **no ETA / seconds remaining** — that would require an unvalidated cost model;
* **no synthetic stages** — a phase is reported only once it genuinely begins.

Failures are preserved verbatim (`error_message` per model and on the run), and
per-model states `queued → running → completed | failed` are mirrored into the
`training_runs` table. The last snapshot is persisted to
`experiments.progress_json`, so progress survives a page refresh and a restart.

---

## Configuration

Copy `backend/.env.example` to `backend/.env`:

```env
# LLM Provider: 'gemini' | 'openai' | 'fallback'
LLM_PROVIDER=fallback

# Google Gemini (optional)
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-flash-lite-latest

# OpenAI (optional)
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

> ⚠️ **Never commit `.env` files.** They are in `.gitignore`.

---

## Running Tests

```bash
cd automl-lens
python -m pytest tests/ -v
```

The full suite (223 passing) covers dataset profiling, ingestion and
suitability checks, feature engineering, the model registry, evaluation metrics,
model selection, training progress, and the full end-to-end pipeline.

---

## Architecture

```
User Browser
    │
    ▼
React 18 (Vite) ──── /api proxy ────▶ FastAPI (Uvicorn)
                                            │
                                     ┌──────┴──────┐
                                     │             │
                                  LLMManager   ML Pipeline
                                     │             │
                          ┌──────────┤             ├─────────────┐
                          │          │             │             │
                       Gemini    OpenAI       Fallback    sklearn/Optuna
                          │          │             │             │
                          └──────────┴─────────────┴─────────────┘
                                            │
                                       SQLite DB
```

---

## ML Pipeline

1. **Dataset Upload** — CSV/XLSX up to 100MB
2. **Profiling** — Column statistics, type inference, quality warnings
3. **LLM Analysis** — Dataset context → problem type + model recommendations
4. **Preprocessing** — sklearn ColumnTransformer pipeline (imputation, scaling, encoding)
5. **Feature Engineering** — Datetime extraction, interaction features, frequency encoding
6. **Model Training** — 8 classifiers or 9 regressors with baseline training
7. **Optuna Optimization** — Bayesian HPO with StratifiedKFold/KFold CV
8. **Evaluation** — Real sklearn metrics computed from actual predictions
9. **Explainability** — SHAP TreeExplainer → feature_importances_ → permutation
10. **Prediction & Report** — Single/batch prediction, HTML report, model download

---

## References

- Feurer et al. (2015). *Auto-sklearn: Efficient and Robust Automated Machine Learning*. NeurIPS.
- He et al. (2021). *AutoML: A Survey of the State-of-the-Art*. IEEE TKDE.
- Brown et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS.

---

## License

MIT License — see [LICENSE](LICENSE) file.
