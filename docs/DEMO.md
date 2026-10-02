# AutoML-Lens — Live Demo Runbook

Everything below was executed in this environment; nothing is guessed.

## 1. Start

From the repo root (`automl-lens\`):

```powershell
cmd /c run_project.bat
```

The script installs backend/frontend dependencies if missing, writes
`backend\.env` if it does not exist (defaults to the deterministic fallback
LLM provider — no API key needed), then opens two windows:

- **Backend** — `python -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0`
- **Frontend** — `npm run dev`

### URLs (verified HTTP 200)

| | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/health → `{"status":"healthy","version":"1.0.0",...}` |

`run_project.bat` ran without modification in this environment (it auto-creates
`backend\.env`, so never open or edit that file during the demo).

## 2. Dataset to upload

**Primary: `demo_data\classification.csv`** — 300 rows × 12 columns,
target column `Churn` (the app auto-suggests it).

Alternate: `benchmarks\data\wine+quality\winequality-red.csv` — 1599 rows ×
12 columns, semicolon-delimited (parsed automatically), target `quality`.

## 3. Click order

1. Open http://localhost:5173 → **Dashboard**.
2. Click **New Experiment**.
3. Drop `demo_data\classification.csv` into the dropzone. It uploads,
   auto-profiles, fills target `Churn`, and shows the **Dataset Readiness**
   card.
4. Leave Problem Type = **Auto-detect** and Metric = **Auto (based on problem
   type)** → click **Create Experiment**.
5. On the Experiment page (step **Dataset**) click **Profile Dataset**.
6. The page jumps to step **AI Analysis** → click **Run AI Analysis**
   (runs on the fallback provider — no key required; recommended metric comes
   back as `f1_weighted`).
7. Steps **Preprocess / Models / Optimize** show the training buttons →
   click **Fast Demo** (3 models). A live progress panel streams while it runs.
8. Landing on step **Evaluate**, you get: best model + score + metric, the
   selection protocol (CV on the training split, holdout untouched), model
   comparison chart, per-model table, real Optuna trial history, confusion
   matrix.
9. Continue through **Explain** (feature importance), **Predict** (form is
   pre-filled with schema sample values → click **Predict**), and **Report**
   → **Generate HTML Report** → **Open Report** (served at
   `/reports/report_<id>.html`).

## 4. What each screen shows

- **Dashboard** — stats grid (total / completed / running / best score) and
  the recent-experiments table (name, target, type, mode, best model, score,
  status, date).
- **Experiment** — stepper with one panel per stage, the AI analysis output,
  evaluation tables/charts, prediction form, report buttons, and the chat
  button (bottom-right).
- **History** — filterable table (all / completed / training / failed /
  created) including the **Metric** column for every experiment.

## 5. Verified API sequence (both datasets)

Every step of the demo flow was called directly against
`http://localhost:8000/api` — all returned HTTP 200:

| Step | Endpoint | classification.csv | winequality-red.csv |
|---|---|---|---|
| Upload | `POST /datasets/upload` | 300×12 | 1599×12 |
| Profile | `GET /datasets/{id}/profile` | target `Churn`, classification | target `quality`, classification |
| Readiness | `GET /datasets/{id}/suitability` | suitable, 0 blocking issues | suitable, 0 blocking issues |
| Create | `POST /experiments` | ok | ok |
| Analyze | `POST /experiments/{id}/analyze` | ok | ok |
| Train | `POST /experiments/{id}/train?fast_demo=true` | completed, 3 models | completed, 3 models |
| Results | `GET /experiments/{id}/results` | best `HistGradientBoosting`, score 0.8818, metric `f1_weighted` | best `HistGradientBoosting`, score 0.6619, metric `f1_weighted` |
| Input schema | `GET /experiments/{id}/input-schema` | 11 fields | 11 fields |
| Predict | `POST /experiments/{id}/predict` | prediction 1, confidence 0.9779 | prediction 5, confidence 0.9823 |
| Report | `GET /experiments/{id}/report` | `/reports/report_4.html` | `/reports/report_5.html` |

Measured wall time for the whole sequence (upload → report):
**55 s** for `classification.csv`, **189 s** for `winequality-red.csv`
(fast demo, 3 folds).

A prediction was also re-sent with every field as a **string** (exactly what
the browser form produces) — both datasets returned HTTP 200.

## 6. Frontend build

```powershell
cd frontend
npm run build
```

exited 0 (vite build; only warning: chunk size > 500 kB).
