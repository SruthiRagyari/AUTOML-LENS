# 🔬 AutoML-Lens

> **LLM-Powered Automated Machine Learning Framework**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)](https://scikit-learn.org)
[![Optuna](https://img.shields.io/badge/Optuna-5.0-purple)](https://optuna.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

**B.Tech Final Year Project — Computer Science Engineering (AI/ML Specialization)**

*"Construction of Automated Machine Learning Framework Based on Large Language Models"*

---

## 📋 Overview

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

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- (Optional) Google Gemini API key or OpenAI API key

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/automl-lens.git
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

## 🎮 Demo

1. Go to **Dashboard** → **New Experiment**
2. Upload `demo_data/classification.csv` (customer churn) or `demo_data/regression.csv` (house prices)
3. Target is auto-suggested (`Churn` or `Price`)
4. Select **Fast Demo** mode (3 models, 5 Optuna trials, 3-fold CV)
5. Click **Create Experiment** → **Profile** → **AI Analysis** → **Start Training**
6. View model comparison, SHAP feature importance, make predictions, download report

---

## 📁 Project Structure

```
automl-lens/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routes (datasets, experiments, chat)
│   │   ├── core/           # Config, database, security
│   │   ├── llm/            # LLM providers (Gemini, OpenAI, Fallback)
│   │   ├── models/         # Pydantic schemas
│   │   ├── services/       # ML pipeline services
│   │   └── utils/          # File utilities, validators
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/     # Layout, Charts, Common UI
│       ├── pages/          # Landing, Dashboard, Experiment, History, etc.
│       └── services/       # API client
├── demo_data/              # Sample CSV datasets
├── storage/                # Datasets, models, reports, predictions
├── tests/                  # pytest test suite (15 tests)
└── README.md
```

---

## 🔧 Configuration

Copy `backend/.env.example` to `backend/.env`:

```env
# LLM Provider: 'gemini' | 'openai' | 'fallback'
LLM_PROVIDER=fallback

# Google Gemini (optional)
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-2.0-flash

# OpenAI (optional)
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

> ⚠️ **Never commit `.env` files.** They are in `.gitignore`.

---

## 🧪 Running Tests

```bash
cd automl-lens
python -m pytest tests/ -v
```

All 15 tests cover: profiling, model registry, evaluation metrics, and full end-to-end pipeline.

---

## 🏗️ Architecture

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

## 📊 ML Pipeline

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

## 📚 References

- Feurer et al. (2015). *Auto-sklearn: Efficient and Robust Automated Machine Learning*. NeurIPS.
- He et al. (2021). *AutoML: A Survey of the State-of-the-Art*. IEEE TKDE.
- Brown et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.
