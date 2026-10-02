import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

const FEATURES = [
  { icon: '🤖', title: 'LLM-Assisted Understanding', desc: 'Integrates Google Gemini or OpenAI to analyze your dataset, infer problem type, and recommend ML strategy — with a deterministic fallback for offline use.' },
  { icon: '⚡', title: 'Automated ML Pipeline', desc: 'End-to-end automation from dataset upload to trained model — profiling, preprocessing, training, and evaluation all handled automatically.' },
  { icon: '🔧', title: 'Hyperparameter Optimization', desc: 'Bayesian optimization via Optuna with configurable trials and cross-validation, selecting the best parameters for every model.' },
  { icon: '🔍', title: 'Explainable AI', desc: 'SHAP values, feature importances, and coefficient analysis reveal which features drive your model\'s decisions.' },
]

const STEPS = [
  { num: 1, title: 'Upload', desc: 'CSV/Excel' },
  { num: 2, title: 'Profile', desc: 'Auto-analyze' },
  { num: 3, title: 'LLM Analysis', desc: 'AI strategy' },
  { num: 4, title: 'Preprocess', desc: 'Sklearn pipeline' },
  { num: 5, title: 'Select Models', desc: 'Registry' },
  { num: 6, title: 'Optimize', desc: 'Optuna' },
  { num: 7, title: 'Evaluate', desc: 'Real metrics' },
  { num: 8, title: 'Explain', desc: 'SHAP / FI' },
  { num: 9, title: 'Predict', desc: 'Single & batch' },
  { num: 10, title: 'Report', desc: 'HTML export' },
]

const TECH = [
  { icon: '🐍', name: 'Python 3.11' }, { icon: '⚡', name: 'FastAPI' }, { icon: '⚛️', name: 'React 18' },
  { icon: '🤖', name: 'Google Gemini' }, { icon: '📊', name: 'scikit-learn' }, { icon: '🔧', name: 'Optuna' },
  { icon: '🔍', name: 'SHAP' }, { icon: '📈', name: 'Plotly' }, { icon: '🗄️', name: 'SQLAlchemy' },
]

export default function Landing() {
  // The top nav is transparent over the hero and turns solid once the page
  // scrolls, so the wordmark stays readable on every section.
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <div className="landing">
      <nav className={`landing-nav${scrolled ? ' scrolled' : ''}`}>
        <span className="landing-nav-logo">AutoML-Lens</span>
        <div className="landing-nav-links">
          <Link to="/methodology">Methodology</Link>
          <Link to="/about">About</Link>
          <Link to="/history">History</Link>
          <Link to="/dashboard" style={{ background: 'var(--primary)', color: '#fff', padding: '8px 20px', borderRadius: 8 }}>Open App</Link>
        </div>
      </nav>

      <div className="landing-hero">
        
        <h1 className="hero-title">
          <span className="gradient-text">AutoML-Lens</span>
        </h1>
        <p className="hero-subtitle">
          LLM-guided AutoML platform
        </p>
        <div className="hero-buttons">
          <Link to="/dashboard" className="btn-hero-primary">Start experiment</Link>
          <Link to="/methodology" className="btn-hero-secondary">How it works</Link>
        </div>
      </div>

      <div>
        <div className="landing-section">
          <div className="section-header">
            <h2>Why AutoML-Lens?</h2>
            <p>AutoML-Lens combines classical AutoML with large language model guidance.</p>
          </div>
          <div className="features-grid">
            {FEATURES.map(f => (
              <div key={f.title} className="feature-card">
                <div className="feature-icon">{f.icon}</div>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="landing-section">
        <div className="section-header">
          <h2>10-Step Automated Pipeline</h2>
          <p>Every step automated, every result explainable.</p>
        </div>
        <div className="pipeline-grid">
          {STEPS.map((s, i) => (
            <div key={s.num} className="pipeline-step" style={i === STEPS.length - 1 ? {} : {}}>
              <div className="pipeline-num">{s.num}</div>
              <h4>{s.title}</h4>
              <p>{s.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div>
        <div className="landing-section">
          <div className="section-header">
            <h2>Technology Stack</h2>
            <p>Built on industry-standard open-source tools.</p>
          </div>
          <div className="tech-grid">
            {TECH.map(t => (
              <div key={t.name} className="tech-badge">
                <span>{t.icon}</span> {t.name}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="landing-footer">
        <p><strong>AutoML-Lens</strong> — LLM-guided AutoML platform</p>
        <p style={{ marginTop: 8 }}>Profiling · LLM-guided analysis · leakage-safe model selection · explainability · reports</p>
      </div>
    </div>
  )
}
