import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  FiZap,
  FiCpu,
  FiSliders,
  FiLayers,
  FiTarget,
  FiFileText,
  FiCheckCircle,
  FiArrowRight,
  FiUploadCloud,
  FiMessageSquare
} from 'react-icons/fi'

const PILLARS = [
  {
    icon: <FiCpu style={{ color: 'var(--netflix-red)' }} />,
    title: 'LLM-Guided Pipeline Planning',
    desc: 'Structured JSON pipeline planning using Gemini or OpenAI, validated against a strict model registry with instant deterministic fallback.'
  },
  {
    icon: <FiSliders style={{ color: 'var(--netflix-red)' }} />,
    title: 'Optuna Bayesian Optimization',
    desc: 'Automated hyperparameter tuning with Tree-structured Parzen Estimators (TPE) and median pruning across 17 model families.'
  },
  {
    icon: <FiLayers style={{ color: 'var(--netflix-red)' }} />,
    title: 'Model Fusion & Ensembles',
    desc: 'Leakage-safe voting classifiers and ridge-regularized stacking ensembles constructed strictly using out-of-fold cross-validation.'
  },
  {
    icon: <FiTarget style={{ color: 'var(--netflix-red)' }} />,
    title: 'Strict Holdout Discipline',
    desc: 'Zero data leakage guarantee. Holdout test sets remain completely untouched during feature engineering, CV selection, and HPO.'
  },
  {
    icon: <FiZap style={{ color: 'var(--netflix-red)' }} />,
    title: 'Explainable AI Integration',
    desc: 'Model-agnostic SHAP values, feature importance distributions, and permutation importance for transparent machine learning decisions.'
  },
  {
    icon: <FiFileText style={{ color: 'var(--netflix-red)' }} />,
    title: 'Reproducible Benchmarks',
    desc: 'Scientific multi-seed empirical evaluation framework comparing deterministic and LLM-assisted AutoML with persisted metrics.'
  }
]

const PIPELINE_STEPS = [
  { num: '01', title: 'Data Ingestion', desc: 'CSV / XLSX upload & structure validation' },
  { num: '02', title: 'Profiling', desc: 'Column distributions & suitability check' },
  { num: '03', title: 'LLM Planning', desc: 'Structured plan & registry validation' },
  { num: '04', title: 'Preprocessing', desc: 'Fold-safe categorical & numeric transformers' },
  { num: '05', title: 'Model Selection', desc: '17 candidate algorithms across tasks' },
  { num: '06', title: 'Bayesian HPO', desc: 'Optuna search space exploration' },
  { num: '07', title: 'Model Fusion', desc: 'Leakage-free voting & stacking ensembles' },
  { num: '08', title: 'Explainability', desc: 'SHAP summary plots & feature drivers' },
  { num: '09', title: 'Model Serving', desc: 'Single record and batch CSV predictions' },
  { num: '10', title: 'Benchmark Report', desc: 'Standalone HTML benchmark report export' },
]

export default function Landing() {
  const [scrolled, setScrolled] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <div className="landing-page" style={{ background: 'var(--bg-black)', minHeight: '100vh', color: '#fff' }}>
      {/* Top Navbar */}
      <header className={`netflix-navbar ${scrolled ? 'scrolled' : ''}`}>
        <div className="nav-left">
          <Link to="/" className="netflix-brand">
            <span className="netflix-brand-logo">AUTOML-LENS</span>
          </Link>
          <nav className="nav-links">
            <Link to="/dashboard" className="nav-link-item">Experiments</Link>
            <Link to="/datasets" className="nav-link-item">Datasets</Link>
            <Link to="/models" className="nav-link-item">Models</Link>
            <Link to="/predictions" className="nav-link-item">Predictions</Link>
            <Link to="/reports" className="nav-link-item">Reports</Link>
            <Link to="/assistant" className="nav-link-item">AI Assistant</Link>
            <Link to="/methodology" className="nav-link-item">Methodology</Link>
          </nav>
        </div>

        <div className="nav-right">
          <Link to="/dashboard" className="btn-netflix-primary">
            Open Platform
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '60px 32px 40px' }}>
        <div className="netflix-hero-banner" style={{ padding: '64px 48px', position: 'relative' }}>
          <div className="hero-pill-badge">
            <FiZap /> LLM-Guided Automated Machine Learning
          </div>
          <h1 className="hero-title" style={{ fontSize: '48px', fontWeight: '900', letterSpacing: '-0.5px' }}>
            AI-POWERED AUTOML PLATFORM
          </h1>
          <p className="hero-desc" style={{ fontSize: '18px', maxWidth: '720px', lineHeight: '1.6' }}>
            Transform tabular data into production-grade, explainable machine learning models.
            Featuring structured LLM pipeline planning, fold-safe preprocessing, Optuna Bayesian optimization,
            and out-of-fold model fusion.
          </p>
          <div className="hero-actions">
            <Link to="/dashboard?new=1" className="btn-netflix-primary" style={{ padding: '14px 28px', fontSize: '16px' }}>
              Start Experiment <FiArrowRight />
            </Link>
            <Link to="/datasets" className="btn-netflix-secondary" style={{ padding: '14px 24px', fontSize: '15px' }}>
              <FiUploadCloud /> Upload Dataset
            </Link>
            <Link to="/assistant" className="btn-netflix-secondary" style={{ padding: '14px 24px', fontSize: '15px' }}>
              <FiMessageSquare /> Ask AI Assistant
            </Link>
          </div>

          <div style={{ display: 'flex', gap: '24px', marginTop: '48px', flexWrap: 'wrap', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '24px' }}>
            <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              <strong style={{ color: '#fff' }}>17</strong> Supported ML Algorithms
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              <strong style={{ color: 'var(--success)' }}>100%</strong> Fold-Safe Isolation
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              <strong style={{ color: 'var(--netflix-red)' }}>Zero</strong> Holdout Leakage
            </div>
          </div>
        </div>

        {/* Feature Highlights Grid */}
        <div style={{ marginBottom: '64px' }}>
          <div className="netflix-row-header">
            <h2 className="netflix-row-title">Core Capabilities</h2>
          </div>
          <div className="netflix-grid">
            {PILLARS.map((p, idx) => (
              <div key={idx} className="netflix-card netflix-card-featured">
                <div style={{ fontSize: '28px', marginBottom: '14px' }}>{p.icon}</div>
                <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '8px', color: '#fff' }}>
                  {p.title}
                </h3>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                  {p.desc}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* 10-Step Pipeline */}
        <div style={{ marginBottom: '64px' }}>
          <div className="netflix-row-header">
            <h2 className="netflix-row-title">10-Step Automated Pipeline Architecture</h2>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '16px' }}>
            {PIPELINE_STEPS.map((s) => (
              <div key={s.num} className="netflix-card" style={{ padding: '20px' }}>
                <div style={{ fontSize: '13px', fontWeight: '900', color: 'var(--netflix-red)', marginBottom: '6px' }}>
                  STEP {s.num}
                </div>
                <h4 style={{ fontSize: '15px', fontWeight: '800', marginBottom: '4px', color: '#fff' }}>
                  {s.title}
                </h4>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                  {s.desc}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom CTA Card */}
        <div className="netflix-card" style={{ textAlign: 'center', padding: '56px 32px', background: 'linear-gradient(180deg, #181818 0%, #111111 100%)', border: '1px solid var(--border-highlight)' }}>
          <h2 style={{ fontSize: '32px', fontWeight: '900', marginBottom: '12px' }}>
            Ready to Run Machine Learning?
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '16px', maxWidth: '600px', margin: '0 auto 28px' }}>
            Upload your CSV or Excel dataset, let the LLM-guided pipeline optimize the best model,
            and inspect comprehensive SHAP explainability.
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '16px' }}>
            <Link to="/dashboard?new=1" className="btn-netflix-primary" style={{ padding: '12px 28px', fontSize: '15px' }}>
              Launch Experiment
            </Link>
            <Link to="/reports" className="btn-netflix-secondary" style={{ padding: '12px 24px', fontSize: '15px' }}>
              View Benchmark Report
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
