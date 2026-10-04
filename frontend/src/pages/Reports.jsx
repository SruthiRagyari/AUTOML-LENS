import React, { useState, useEffect } from 'react'
import {
  FiFileText,
  FiDownload,
  FiExternalLink,
  FiAward,
  FiBarChart2,
  FiDatabase
} from 'react-icons/fi'
import { api } from '../services/api'
import LoadingSpinner from '../components/Common/LoadingSpinner'

export default function Reports() {
  const [experiments, setExperiments] = useState([])
  const [selectedExpId, setSelectedExpId] = useState('')
  const [benchmarks, setBenchmarks] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [expRes, benchRes] = await Promise.allSettled([
        api.listExperiments(),
        api.getBenchmarksSummary(),
      ])
      if (expRes.status === 'fulfilled') {
        const completed = (expRes.value.data?.experiments || expRes.value.data || []).filter((e) => e.status === 'completed')
        setExperiments(completed)
        if (completed.length > 0) setSelectedExpId(String(completed[0].id))
      }
      if (benchRes.status === 'fulfilled') {
        setBenchmarks(benchRes.value.data)
      }
    } catch (err) {
      console.error('Failed to load report data:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-container">
      <div className="netflix-hero-banner" style={{ padding: '40px 36px', marginBottom: '32px' }}>
        <div className="hero-pill-badge">Scientific Reports & Artifacts</div>
        <h1 className="hero-title" style={{ fontSize: '36px' }}>Research Reports & Benchmarks</h1>
        <p className="hero-desc" style={{ fontSize: '16px', maxWidth: '650px' }}>
          Explore the official standalone multi-seed research evaluation report comparing deterministic
          AutoML with LLM-guided structured pipeline planning.
        </p>
      </div>

      {/* Featured Research Report Card */}
      <div className="netflix-card netflix-card-featured" style={{ marginBottom: '36px', padding: '32px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '24px', flexWrap: 'wrap' }}>
          <div>
            <div className="badge-netflix badge-completed" style={{ marginBottom: '12px' }}>
              Freeze Verified Research Artifact
            </div>
            <h2 style={{ fontSize: '24px', fontWeight: '900', color: '#ffffff', marginBottom: '8px' }}>
              Standalone Multi-Seed Benchmark Evaluation Report
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '15px', maxWidth: '700px', lineHeight: '1.6' }}>
              Comprehensive evaluation of 18 empirical runs across Adult Census Income (Classification)
              and Wine Quality (Regression). Compares CV scores, holdout scores, runtime, and pipeline
              composition between Deterministic Fallback and LLM-guided structured planning.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '12px' }}>
            <a
              href="/api/experiments/benchmarks/report"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-netflix-primary"
            >
              <FiExternalLink /> View HTML Report
            </a>
            <a
              href="/api/experiments/benchmarks/report"
              download="BENCHMARK_REPORT.html"
              className="btn-netflix-secondary"
            >
              <FiDownload /> Download HTML
            </a>
          </div>
        </div>
      </div>

      {/* Benchmark Summary Metrics */}
      {benchmarks?.datasets && (
        <div style={{ marginBottom: '36px' }}>
          <div className="netflix-row-header">
            <h2 className="netflix-row-title">Multi-Seed Benchmark Highlights</h2>
          </div>
          <div className="netflix-grid">
            {Object.entries(benchmarks.datasets).map(([datasetKey, dsInfo]) => (
              <div key={datasetKey} className="netflix-card">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                  <h3 style={{ fontSize: '18px', fontWeight: '800', color: '#ffffff' }}>
                    {datasetKey === 'adult_income' ? 'Adult Census Income' : 'Wine Quality'}
                  </h3>
                  <span className="badge-netflix badge-created">
                    {dsInfo.problem_type}
                  </span>
                </div>

                <div style={{ marginBottom: '16px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                  Primary Metric: <strong style={{ color: '#fff' }}>{dsInfo.primary_metric}</strong>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
                  <div style={{ background: '#111', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Deterministic Mean</div>
                    <div style={{ fontSize: '16px', fontWeight: '800', color: '#fff', marginTop: '4px' }}>
                      {dsInfo.conditions?.deterministic?.holdout_mean?.toFixed(4) || 'N/A'}
                    </div>
                  </div>
                  <div style={{ background: '#111', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>LLM-Guided Mean</div>
                    <div style={{ fontSize: '16px', fontWeight: '800', color: 'var(--netflix-red)', marginTop: '4px' }}>
                      {dsInfo.conditions?.llm_guided?.holdout_mean?.toFixed(4) || 'N/A'}
                    </div>
                  </div>
                </div>

                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                  Seeds Evaluated: <strong>{dsInfo.seeds_evaluated?.join(', ') || '42, 123, 999'}</strong>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-Experiment Report Generator */}
      <div className="netflix-card">
        <h2 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '16px' }}>
          Individual Experiment Artifacts
        </h2>
        {loading ? (
          <LoadingSpinner />
        ) : experiments.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>No completed experiments available.</p>
        ) : (
          <div>
            <div style={{ display: 'flex', gap: '16px', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap' }}>
              <select
                className="netflix-select"
                style={{ maxWidth: '400px' }}
                value={selectedExpId}
                onChange={(e) => setSelectedExpId(e.target.value)}
              >
                {experiments.map((exp) => (
                  <option key={exp.id} value={exp.id}>
                    #{exp.id} - {exp.name || `Experiment ${exp.id}`}
                  </option>
                ))}
              </select>

              <a
                href={api.downloadModelUrl(selectedExpId)}
                className="btn-netflix-secondary"
                download
              >
                <FiDownload /> Download Trained Model (.joblib)
              </a>
              <a
                href={api.downloadMetaUrl(selectedExpId)}
                className="btn-netflix-secondary"
                download
              >
                <FiFileText /> Download Metadata (.json)
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
