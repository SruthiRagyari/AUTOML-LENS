import React from 'react'

export default function Methodology() {
  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Methodology</h1>
          <p>LLM-guided AutoML platform</p>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h2>1. Background</h2>
        <div className="research-section">
          <p>Traditional AutoML systems automate the ML pipeline through exhaustive search or evolutionary strategies but lack semantic understanding of the data. AutoML-Lens adds a reasoning layer: the language model analyzes dataset context, infers problem semantics, and guides model selection — while the classical ML pipeline handles preprocessing, training, and optimization with mathematical rigor.</p>
          <p>The approach combines three fields: <strong>Neural Architecture Search (NAS)</strong>, <strong>Hyperparameter Optimization (HPO)</strong>, and <strong>Large Language Model (LLM) reasoning</strong>.</p>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h2>2. Three Operating Modes</h2>
        <div className="comparison-grid">
          <div className="comparison-col">
            <h4>Mode A: Baseline</h4>
            <ul><li>Default hyperparameters only</li><li>No optimization</li><li>Reference performance</li><li>Fast execution</li><li>sklearn defaults</li></ul>
          </div>
          <div className="comparison-col">
            <h4>Mode B: AutoML</h4>
            <ul><li>Optuna Bayesian optimization</li><li>Configurable n_trials</li><li>Cross-validation scoring</li><li>Model registry search</li><li>Best-model selection</li></ul>
          </div>
          <div className="comparison-col">
            <h4>Mode C: LLM Assisted</h4>
            <ul><li>LLM dataset analysis</li><li>Semantic model recommendation</li><li>Optuna optimization</li><li>AI-guided preprocessing</li><li>Natural language explanation</li></ul>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h2>3. Pipeline Architecture</h2>
        <div className="research-section">
          <p><strong>Phase 1 — Dataset Understanding:</strong> The DatasetProfiler computes per-column statistics (dtype, missing %, cardinality, sample values) and infers semantic types (numerical, categorical, text, datetime, id-like, constant, boolean). Target suggestion uses keyword heuristics and cardinality analysis.</p>
          <p><strong>Phase 2 — LLM Analysis:</strong> The LLMManager dispatches a structured prompt (dataset context + JSON schema) to the configured provider (Gemini/OpenAI/Fallback). The FallbackLLMProvider uses deterministic rule-based heuristics ensuring the system works without any API key.</p>
          <p><strong>Phase 3 — Preprocessing:</strong> A sklearn ColumnTransformer pipeline applies: (a) median imputation + StandardScaler for numerical features, (b) mode imputation + OneHotEncoder (max 20 categories) for categorical features, (c) TF-IDF (max 100 features) for text columns. ID-like and constant columns are dropped automatically.</p>
          <p><strong>Phase 4 — Training & Optimization:</strong> TrainingManager iterates over ModelRegistry entries. For each model: (1) baseline training with default params, (2) Optuna study with StratifiedKFold/KFold cross-validation, (3) final retraining with best params. Per-model exceptions are caught so one failure doesn't abort the run.</p>
          <p><strong>Phase 5 — Evaluation & Explainability:</strong> The Evaluator computes comprehensive classification (accuracy, F1-weighted, precision, recall, balanced accuracy, ROC-AUC, confusion matrix) or regression (MAE, MSE, RMSE, R², MAPE) metrics from real predictions. The Explainer uses SHAP TreeExplainer → feature_importances_ → coef_ → permutation_importance in priority order.</p>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h2>4. Model Registry</h2>
        <div style={{ display: 'flex', gap: 24 }}>
          <div style={{ flex: 1 }}>
            <div className="section-title">Classification (8 models)</div>
            <ul style={{ fontSize: 14, paddingLeft: 18, color: 'var(--text-secondary)' }}>
              <li>Logistic Regression</li><li>Decision Tree</li><li>Random Forest</li>
              <li>Gradient Boosting</li><li>HistGradientBoosting</li>
              <li>K-Nearest Neighbors</li><li>Support Vector Machine</li><li>Naive Bayes</li>
            </ul>
          </div>
          <div style={{ flex: 1 }}>
            <div className="section-title">Regression (9 models)</div>
            <ul style={{ fontSize: 14, paddingLeft: 18, color: 'var(--text-secondary)' }}>
              <li>Linear Regression</li><li>Ridge</li><li>Lasso</li><li>Decision Tree</li>
              <li>Random Forest</li><li>Gradient Boosting</li><li>HistGradientBoosting</li>
              <li>K-Nearest Neighbors</li><li>Support Vector Regressor</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>5. Evaluation Methodology</h2>
        <div className="research-section">
          <p>All metrics are computed from real model predictions using sklearn.metrics — no values are hardcoded or simulated. The best model is selected by comparing actual test-set metric values, with higher-is-better semantics applied correctly (accuracy, R², F1 → higher; RMSE, MAE → lower).</p>
          <p><strong>References:</strong> Feurer et al. (2015) Auto-sklearn: Efficient and Robust Automated Machine Learning. NeurIPS. · He et al. (2021) AutoML: A Survey of the State-of-the-Art. TKDE. · Brown et al. (2020) Language Models are Few-Shot Learners. NeurIPS.</p>
        </div>
      </div>
    </div>
  )
}
