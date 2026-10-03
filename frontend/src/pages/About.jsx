import React from 'react'

export default function About() {
  const techStack = [
    { cat: 'Backend', items: ['Python 3.11', 'FastAPI 0.115.6', 'SQLAlchemy', 'Pydantic v2', 'Uvicorn'] },
    { cat: 'ML/AI', items: ['scikit-learn 1.6.1', 'Optuna 4.2.0', 'SHAP', 'NumPy', 'pandas'] },
    { cat: 'LLM', items: ['Google Gemini', 'OpenAI API', 'Fallback (Deterministic)', 'httpx async'] },
    { cat: 'Frontend', items: ['React 18', 'Vite 6', 'react-router-dom 6', 'react-plotly.js', 'axios'] },
  ]
  const limitations = [
    'Tabular data only — no image, audio, or NLP datasets in current version',
    'LLM-dependent features require valid API keys (fallback mode available)',
    'Large datasets (>50MB) may experience slow profiling',
    'No time-series or multi-label classification support yet',
  ]
  const futureWork = [
    'Async/background training with WebSocket progress updates',
    'Deep learning models (TensorFlow/PyTorch integration)',
    'Time-series forecasting support',
    'Multi-objective optimization (accuracy + inference speed)',
    'Model versioning and A/B testing support',
    'Cloud deployment with Docker & Kubernetes',
    'LLM fine-tuning on domain-specific datasets',
  ]
  return (
    <div className="page">
      <div className="page-header"><h1>About AutoML-Lens</h1></div>
      <div className="card" style={{ marginBottom: 20 }}>
        <h2>Product Overview</h2>
        <p style={{ fontSize: 15, lineHeight: 1.8, color: 'var(--text-secondary)', marginBottom: 16 }}>
          AutoML-Lens is an LLM-guided AutoML platform. It takes a raw tabular dataset (CSV/Excel) and walks it through a complete machine-learning workflow: profiling and type inference, LLM-guided problem analysis, preprocessing, cross-validated and leakage-safe model selection with hyperparameter tuning, explainability, and a shareable HTML report — from a React UI backed by FastAPI.
        </p>
        <p style={{ fontSize: 15, lineHeight: 1.8, color: 'var(--text-secondary)', marginBottom: 16 }}>
          LLM output is advisory: recommendations are validated against the dataset before training, every metric is computed from real model predictions, and a deterministic fallback keeps the whole pipeline working without any API key.
        </p>
        <div className="alert alert-info">
          <strong>How it works:</strong> upload a dataset → profile it → run LLM-guided analysis (Google Gemini, OpenAI, or the built-in fallback) → train candidate models with Optuna search → evaluate on a held-out test split → explain the winner → predict or export the report.
        </div>
      </div>
      <div className="card" style={{ marginBottom: 20 }}>
        <h2>Technology Stack</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
          {techStack.map(ts => (
            <div key={ts.cat} style={{ background: 'var(--bg-main)', borderRadius: 8, padding: 16 }}>
              <div className="section-title" style={{ marginTop: 0 }}>{ts.cat}</div>
              <ul style={{ listStyle: 'none', padding: 0 }}>
                {ts.items.map(item => <li key={item} style={{ padding: '3px 0', fontSize: 13.5, color: 'var(--text-secondary)' }}>• {item}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <div className="card">
          <h2>Current Limitations</h2>
          <ul style={{ paddingLeft: 18, fontSize: 14, lineHeight: 2, color: 'var(--text-secondary)' }}>
            {limitations.map((l, i) => <li key={i}>{l}</li>)}
          </ul>
        </div>
        <div className="card">
          <h2>Future Scope</h2>
          <ul style={{ paddingLeft: 18, fontSize: 14, lineHeight: 2, color: 'var(--text-secondary)' }}>
            {futureWork.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      </div>
      <div className="card">
        <h2>References</h2>
        <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
          Feurer, M., Klein, A., Eggensperger, K., Springenberg, J. T., Blum, M., & Hutter, F. (2015). <em>Efficient and Robust Automated Machine Learning</em>. Advances in Neural Information Processing Systems (NeurIPS).
        </p>
        <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.8, marginTop: 10 }}>
          He, X., Zhao, K., & Chu, X. (2021). <em>AutoML: A Survey of the State-of-the-Art</em>. IEEE Transactions on Knowledge and Data Engineering.
        </p>
      </div>
    </div>
  )
}
