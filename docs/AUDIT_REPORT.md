# AutoML-Lens — Repo Audit Report

**Date:** 2026-09-30 · **Commit:** `e4025e9` (main, 5 modified files uncommitted) · **Auditor:** Cline
**Goal:** find what blocks *real user-uploaded data* from working end-to-end, and separate "real computation" from anything simulated.

## A. Method & evidence base

| Check | Result |
|---|---|
| Unit/integration tests | `python -m pytest tests/ -q` → **15 passed, EXIT=0** (3.9s) |
| End-to-end API probe | `TestClient` script replicating the exact frontend call sequence (upload → profile → create → train → analyze → predict → input-schema), run in an isolated CWD |
| Static review | all backend modules (api, core, models, llm, services, utils) + all frontend pages/charts |
| Not done | live uvicorn/browser smoke test; LLM calls with a real API key (blocked on your confirmation) |

Everything below marked **[proven]** was reproduced, not inferred.

## B. Repository & stack inventory

```
backend/  FastAPI + Pydantic v2 + SQLAlchemy(SQLite) + Optuna + scikit-learn + SHAP
  app/api/        datasets.py, experiments.py, chat.py
  app/core/       config.py, database.py, security.py
  app/llm/        base, manager, fallback, gemini, openai_compat
  app/services/   profiler, feature_engineer, preprocessor, model_registry,
                  trainer, optimizer, evaluator, explainer, predictor, reporter
  app/utils/      file_utils.py, validators.py
  storage/        datasets/ models/ reports/ predictions/  (+ committed artifacts)
frontend/ React 18 + Vite 6 + Plotly + axios + react-dropzone + react-markdown
tests/    6 pytest modules (profiler, evaluator, registry, preprocessor, pipeline, optimizer)
```

**Verdict:** the architecture is sound and genuinely functional — real sklearn training, real Optuna studies, real SHAP, real metrics. No hardcoded metrics/predictions/charts exist in the backend. The problems are in **plumbing**, not in the ML core.

## C. Startup, configuration, environment

| Item | Status | Evidence |
|---|---|---|
| Backend boots | ✅ | lifespan logs `Database initialized` → `LLM Provider: Fallback (Deterministic)` → `AutoML-Lens ready!` |
| Tests | ✅ | 15/15, exit 0 |
| Frontend toolchain | ✅ present | `frontend/node_modules/.bin/vite.cmd` exists |
| `.env` location | ⚠️ **broken by design** | real file is `backend/.env`; `config.py:9` does `load_dotenv(<repo_root>/.env)` which **does not exist**. Keys only load because pydantic-settings `env_file=".env"` is *CWD-relative* |
| `DATABASE_URL` | ❌ ignored | `database.py:107` hardcodes `Path("automl_lens.db")` |
| Storage path | ❌ CWD-relative | `STORAGE_PATH="./storage"`; `run_project.bat:36` writes `STORAGE_BASE_PATH` (a name nothing reads) |
| API keys | ❌ absent | `backend/.env`: `LLM_PROVIDER=fallback`, `GEMINI_API_KEY=`, `OPENAI_API_KEY=` |

**Consequence [proven by code path]:** if the server is started from the repo root instead of `backend/`, the `.env` is not read at all (silently falling back to the deterministic provider) **and** a second DB/storage pair is created at the root — "my experiments disappeared" and "my API key does nothing".

## D. Real-data ingestion (upload → profile → target)

Working: extension allow-list (`.csv/.xlsx/.xls`), 100 MB cap, sanitised unique filenames, CSV parse, cached profile in `Dataset.profile_json`, target suggestion (`suggested_target='Churn'`, `suggested_problem_type='classification'` **[proven]**).

### 🔴 C1 — `.xlsx` is accepted on upload but breaks everything downstream **[proven]**
```
xlsx upload                    -> 200 OK
GET /datasets/{id}/columns      -> {"columns": []}                     (silent)
POST /experiments/{id}/analyze  -> UnicodeDecodeError
      'utf-8' codec can't decode byte 0xa0 in position 11             (HTTP 500)
```
Root cause: `datasets.py:135` uses `pd.read_csv(..., nrows=0)`; `experiments.py` uses bare `pd.read_csv(ds.file_path)` at **lines 94, 159, 479, 550, 666**. Only `datasets.py` (upload + profile) has the `read_excel` branch. The upload dropzone advertises `.xls/.xlsx`, so this is the most likely first failure for a real user.

### 🔴 C2 — "auto" problem type trains nothing and reports success **[proven]**
`create_experiment` maps `"auto"` → `problem_type=None` (`experiments.py:52`). `/analyze` auto-detects it (lines 100–106) but **`/train` never does**:
```
POST /experiments   {problem_type:"auto", primary_metric:null} -> 200
POST /train         -> 200 {"status":"completed","models":[],"best_model":null,"total_training_time":0}
```
The user gets a green "completed" experiment with **zero models**, a blank best model, and a dead Prediction tab. With a string target the failure is silent because every model error is swallowed per-model by `manager.train_all()`.


## E. LLM layer — provider honesty

The fallback is genuinely transparent in chat (`_Note: Running in Fallback Mode._` appended everywhere) and `exp.llm_provider` records the real provider. Two honesty defects remain:

### 🟠 C5 — AI Analysis tab mislabels the provider and shows a fabricated confidence **[proven by code]**
`Experiment.jsx:270`
```jsx
Provider: {llmAnalysis.provider_used || 'Fallback'} · Confidence: {((llmAnalysis.confidence || 0.8)*100).toFixed(0)}%
```
* `provider_used` is returned at the *top level* of `/analyze` (`manager.py:55`), but the UI does `setLlmAnalysis(la.data?.result || la.data)` (line 79) → the key is **always** `undefined` → a real Gemini analysis is displayed as **"Provider: Fallback"**.
* `DatasetAnalysisResult.confidence` defaults to a hardcoded **0.8** (`base.py:42`) and the fallback returns a fabricated **0.75** (`fallback.py:140`); `|| 0.8` makes it 80% in the UI regardless.
* This is the only violation I found of your "every UI value must come from real computation" rule.

## F. Preprocessing & feature engineering

Preprocessing itself is correct and real: median+scale numeric, mode+one-hot categorical (`max_categories=20` → `infrequent_sklearn`), TF-IDF text, id-like/constant dropped, target excluded, `ColumnTransformer` + `remainder="drop"`.

### 🟠 C4 — engineered features are computed, logged, reported … and then thrown away **[proven]**
```
[c] engineered columns created by FE: ['TicketCode_freq']
[c] transformed feature names:        numerical__Age, … categorical__City_NYC, …   (no *_freq)
[c] engineered cols present in matrix?: []
[c] preprocessor summary: total_features_in: 5, total_features_out: 26
```
`build_and_fit` iterates `profile["column_profiles"]` (computed on the **raw** frame before FE), so new columns (`*_freq`, `A_x_B` interactions) are never attached to a transformer and hit `remainder="drop"`. README/Methodology advertise feature engineering; the model never sees it. Consistency of `transform()` at prediction time is preserved only because both sides drop the same columns.

## G. Training, optimization, evaluation — provenance of numbers

All real: `train_test_split(random_state=42, stratify)` → per-model baseline → Optuna study (TPE) with `StratifiedKFold/KFold` CV → refit with best params → `Evaluator` on the held-out test set (accuracy/F1/precision/recall/balanced/ROC-AUC/PR-AUC/confusion matrix/classification report; MAE/MSE/RMSE/R²/MAPE for regression), timings measured for real. `best_score` extraction was **fixed in the uncommitted diff** (metric-key mapping for `neg_*` metrics) — a genuine improvement I verified by reading the patch. ✔ No simulated metrics anywhere.

### 🟠 C6 — training blocks the event loop **[proven by code path]**
`/train` is `async def` but runs profiling, FE, Optuna and sklearn **synchronously**; only the LLM call is awaited. With one uvicorn worker every other request (including `/api/health` and the UI refresh) is stalled for the whole run — and the browser `fetch` may time out with no feedback.


## H. Explainability, prediction, reports, persistence

* **Explainability** ✅ real and well-guarded: `_select_method` → SHAP `TreeExplainer` → `feature_importances_` → `coef_` → `permutation_importance`; the uncommitted patch adds `_clean_feature_name` stripping `numerical__`/`x0_` prefixes (a real UX improvement — committed metadata still shows raw `numerical__CustomerID`).
* **Model persistence** ✅ `model_{id}.joblib` + `model_{id}_metadata.json` (features, params, metrics, lib versions, seed).
* **Restore-from-disk prediction** ✅ newly added in the working diff, and it correctly re-derives the preprocessor from the original dataset (verified by reading the patch).

### 🔴 C3 — reloading a completed experiment destroys it **[proven]**
```
GET  /experiments/2              -> status "completed"
POST /experiments/2/analyze      -> 200             (the frontend does this on every page load)
GET  /experiments/2              -> status "analyzed"    <- downgraded
POST /experiments/2/predict      -> 400 {"detail":"Experiment not completed"}
GET  /experiments/2/input-schema -> 400 {"detail":"Experiment not completed"}
```
Two defects compound:
1. `_exp_to_dict` (`experiments.py:690`) omits `llm_analysis_json`, so the frontend guard `if (r.data.llm_analysis_json || r.data.status !== 'created')` (`Experiment.jsx:54`) is **always true** → `/analyze` is re-invoked on *every* view/refresh (burning LLM quota) even when an analysis already exists.
2. `/analyze` unconditionally sets `exp.status = "analyzed"` (`experiments.py:145`), so a **completed** experiment is downgraded, and every `status != "completed"` guard (predict / batch-predict / input-schema) then rejects the user. There is no way back except retraining.

### 🟡 Secondary (not blocking, worth knowing)
* `_experiment_cache[exp_id]["df_original"]` keeps the **entire uploaded dataset in RAM forever** (never evicted) — a real memory problem at the 100 MB upload limit.
* The per-model table and `ModelComparison` chart pick the *first* numeric metric (`Object.values(met).find(...)`) while labelling it with `results.primary_metric`; for classification the first key is `accuracy`, so a chart titled `f1_weighted` can plot accuracy. `best_score` itself is mapped correctly (working diff) — the frontend needs the same per-model value.
* `predict_batch` writes a fixed `batch_input_{exp_id}.csv` (concurrent users overwrite each other); predictions are never recorded in the `Prediction` table; no delete/cleanup endpoints; report `metrics: {}` is left empty.

## I. Frontend — value provenance & state machine

* **No mock/demo data in the UI.** Every number comes from an API response; charts render real arrays; `EmptyState` / `No models to compare yet.` correctly handle absence. `DEFAULT_METRICS` is only a metric-name list (fine).
* Stepper state is derived from `exp.status` (`statusToStep`) — real, not faked; the "Fast Demo" button genuinely switches to 3 models / 5 trials (and the backend enforces `n_folds=3, n_trials=5`).
* The only fabricated value chain is **C5** (provider/confidence), plus **C3**'s state-machine break where the UI silently degrades its own experiment.

## J. Consolidated register

| # | Sev | Issue | Where | Real-data impact | Resolution |
|---|---|---|---|---|---|
| C1 | 🔴 | `.xlsx/.xls` accepted, then `pd.read_csv` → 500 / empty columns | `datasets.py:135`, `experiments.py:94,159,479,550,666` | Spreadsheet uploads unusable | ✅ Already fixed at `9b3d650` — `load_dataframe` dispatches on suffix; verified by `test_c1_xlsx_is_read_not_merely_accepted` |
| C2 | 🔴 | `problem_type="auto"` → `/train` yields 0 models, status "completed" | `experiments.py:52,151-213` | False success, no model to predict with | ✅ Fixed `31855e9` — `/train` now detects the task and defaults the metric from the target profile before model selection; verified by `test_c2_train_detects_problem_type_when_auto` |
| C3 | 🔴 | Page load re-runs `/analyze`, downgrading `completed` → `analyzed` | `experiments.py:145,690`, `Experiment.jsx:54` | Predictions/schema 400 after any refresh; wasted LLM calls | ✅ Fixed `c49fb09` — `/analyze` is idempotent for stored analyses (`?force=true` to re-run), the detail payload exposes `has_analysis`, and the UI hydrates instead of calling the LLM; verified by both `test_c3_*` tests |
| C4 | 🟠 | FE features dropped by `remainder="drop"` | `preprocessor.build_and_fit` | Advertised FE never affects results | ✅ Already fixed at `9b3d650` — `extra_numeric`/`feature_engineer` feed engineered columns into the numeric transformer; verified by `test_c4_engineered_columns_reach_the_model_matrix` |
| C5 | 🟠 | Provider mislabelled + fabricated 0.75/0.8 confidence | `Experiment.jsx:270`, `base.py:42`, `fallback.py:140` | Misleading AI panel | ✅ Already fixed at `9b3d650` — provider provenance is returned by the API and nothing is invented in the UI; verified by both `test_c5_*` tests |
| C6 | 🟠 | Async endpoint does blocking CPU work | `experiments.py:151` | Server frozen during training | ✅ Already fixed at `9b3d650` — training runs in `asyncio.to_thread`; verified by `test_c6_training_is_not_run_inline_on_the_event_loop` and `test_c6_health_responds_while_training_is_in_flight` |
| C7 | 🟠 | `.env`/DB/storage paths CWD-dependent; `DATABASE_URL` ignored; `run_project.bat` wrong var names | `config.py:9`, `database.py:107`, `run_project.bat:32-38` | Keys silently ignored, "lost" experiments | ✅ Fixed `ef869bd` (launcher writes `STORAGE_PATH`); config/DB/storage anchoring was already fixed at `9b3d650`; verified by all three `test_c7_*` tests |
| C8 | 🟡 | Full dataset kept in memory per experiment | `experiments.py:367-375` | RAM blow-up on real data | ✅ Fixed `ea880fe` — `_experiment_cache` is an LRU (`maxsize=8`, evicted entries reload from disk) and `df_original` is no longer stored; verified by both `test_c8_*` tests |
| C9 | 🟡 | Per-model metric taken as "first numeric value" | `Experiment.jsx:144,366` | Chart/table can show accuracy under an `f1_weighted` label | ✅ Fixed `353621b` (report table + LLM prompt use the resolved primary metric); `holdout_primary`/`holdout_primary_key` were already returned at `9b3d650`; verified by both `test_c9_*` tests |
| C10 | 🟡 | Fixed `batch_input_{exp_id}.csv`, no `Prediction` rows, empty report `metrics` | `experiments.py:571,583`, `reporter` | Overwrites on concurrency; no prediction history | ✅ Fixed `0f6dec6` (single + batch predictions recorded, uuid-suffixed batch input) and `353621b` (report labelled with the primary metric); verified by all four `test_c10_*` tests |

Every row above is covered by `tests/test_audit_regressions.py` (20 tests) plus the
pre-existing suite; `python -m pytest tests -q` was run after each change.

## K. Minimal fix plan for the blockers (no rewrite, no deletions)

Ordered by risk-of-doing-nothing; every fix is local and will be covered by the existing suite plus new regression tests.

1. **C1** — add `load_dataframe(path)` to `utils/file_utils.py` (suffix-dispatch `read_csv`/`read_excel`) and use it in `datasets.py` + all five `experiments.py` call sites; batch-predict accepts `.xlsx` too. → CSV and spreadsheets behave identically.
2. **C2** — in `/train`, when `problem_type` is falsy: detect it from the target column profile (`DatasetProfiler.detect_problem_type`, the same helper `/analyze` already uses), set a default `primary_metric`, persist both, and return **400** if the target is missing or **zero models succeed** (never "completed" with an empty model list).
3. **C3** — expose `has_analysis` (+ `has_results`, `report_path`) in `_exp_to_dict`; call `/analyze` from the UI only when `!has_analysis`; make `/analyze` refuse to downgrade `completed`/`training` (separate analysis flag, `?force=true` for a deliberate re-run); hydrate the AI panel from the stored analysis so a reload shows the saved result instead of re-calling the LLM.
4. **C5** — surface `provider_used` in the payload the UI actually consumes (or read `exp.llm_provider`), label the deterministic provider as "Rule-based (no LLM)", drop the `|| 0.8` and show confidence only when the provider returned one.
5. **C7** — anchor `config.py` to `backend/.env`, honour `DATABASE_URL` in `database.py`, resolve storage relative to the backend package, fix `run_project.bat`'s variable names.
6. **C6** — offload the training body to a worker thread (or make the endpoint a sync `def`, which FastAPI already runs in a threadpool) so `/api/health` and the UI stay responsive.
7. **C8** — stop caching `df_original` (rebuild on demand) and bound `_experiment_cache` with a small LRU.
8. **C9/C10** — cheap correctness wins: return the primary-metric value per model so the chart/table label matches the number; unique batch filename; fill the report's `metrics`.
9. **C4** — needs your decision (below), because it changes model inputs and therefore invalidates existing results.

Validation after each fix: `python -m pytest tests/ -q` (must stay 15/15) + re-run the end-to-end probe script; new regression tests for C1 (xlsx → profile → train), C2 (auto problem type → models trained), C3 (completed experiment stays completed after a reload).

## L. Open questions (blocking the fix pass)

1. **LLM provider** — `backend/.env` is `LLM_PROVIDER=fallback` with empty keys. Should I (a) keep the deterministic fallback as the default and only make its labelling honest, (b) wire your Gemini key (needs your explicit go-ahead to use that key/quota), or (c) wire an OpenAI-compatible endpoint? Nothing in the fix list above needs a key — only this choice depends on your answer.
2. **C4 scope** — include engineered features in the model (results change; requires storing the training-time frequency map so `transform()` stays consistent at prediction time) **or** keep the pipeline as-is and correct the README/Methodology/UI wording so nothing is over-claimed? The former changes results; the latter changes only text.

*End of report. The audit itself modified no files outside `docs/`; the fixes that followed are recorded (with commit hashes) in the Resolution column of section J.*

