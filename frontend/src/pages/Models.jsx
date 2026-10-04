import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  FiCpu,
  FiSliders,
  FiEye,
  FiSearch,
  FiPlus,
  FiLayers,
  FiCheckCircle
} from 'react-icons/fi'
import { api } from '../services/api'
import LoadingSpinner from '../components/Common/LoadingSpinner'

export default function Models() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    loadCatalog()
  }, [])

  const loadCatalog = async () => {
    setLoading(true)
    try {
      const res = await api.getModelsCatalog()
      setModels(res.data.models || [])
    } catch (err) {
      console.error('Failed to load model catalog:', err)
    } finally {
      setLoading(false)
    }
  }

  const filteredModels = models.filter((m) => {
    const matchesFilter = filter === 'all' || m.task === filter
    const matchesSearch =
      m.display_name.toLowerCase().includes(search.toLowerCase()) ||
      m.name.toLowerCase().includes(search.toLowerCase())
    return matchesFilter && matchesSearch
  })

  return (
    <div className="page-container">
      <div className="netflix-hero-banner" style={{ padding: '40px 36px', marginBottom: '32px' }}>
        <div className="hero-pill-badge">Model Zoo & HPO Search Spaces</div>
        <h1 className="hero-title" style={{ fontSize: '36px' }}>Supported ML Algorithms</h1>
        <p className="hero-desc" style={{ fontSize: '16px', maxWidth: '650px' }}>
          AutoML-Lens registers 17 diverse algorithms across classification and regression, each with
          bounded Bayesian search spaces for Optuna hyperparameter optimization.
        </p>
      </div>

      {/* Filter and Search Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', marginBottom: '28px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className={`btn-netflix-secondary ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
            style={filter === 'all' ? { borderColor: 'var(--netflix-red)', background: 'var(--netflix-red-subtle)', color: '#fff' } : {}}
          >
            All Models ({models.length})
          </button>
          <button
            className={`btn-netflix-secondary ${filter === 'classification' ? 'active' : ''}`}
            onClick={() => setFilter('classification')}
            style={filter === 'classification' ? { borderColor: 'var(--netflix-red)', background: 'var(--netflix-red-subtle)', color: '#fff' } : {}}
          >
            Classification ({models.filter(m => m.task === 'classification').length})
          </button>
          <button
            className={`btn-netflix-secondary ${filter === 'regression' ? 'active' : ''}`}
            onClick={() => setFilter('regression')}
            style={filter === 'regression' ? { borderColor: 'var(--netflix-red)', background: 'var(--netflix-red-subtle)', color: '#fff' } : {}}
          >
            Regression ({models.filter(m => m.task === 'regression').length})
          </button>
        </div>

        <div style={{ position: 'relative', width: '280px' }}>
          <FiSearch style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
          <input
            type="text"
            className="netflix-input"
            placeholder="Search models..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ paddingLeft: '36px' }}
          />
        </div>
      </div>

      {loading ? (
        <div style={{ padding: '60px', textAlign: 'center' }}><LoadingSpinner /></div>
      ) : filteredModels.length === 0 ? (
        <div className="netflix-card" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
          No models matched your filter.
        </div>
      ) : (
        <div className="netflix-grid">
          {filteredModels.map((m) => (
            <div key={m.name} className="netflix-card netflix-card-featured" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <span className={`badge-netflix ${m.task === 'classification' ? 'badge-completed' : 'badge-training'}`}>
                    {m.task}
                  </span>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    {m.explainability_method?.toUpperCase()}
                  </span>
                </div>

                <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '6px', color: '#ffffff' }}>
                  {m.display_name}
                </h3>
                <code style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'block', marginBottom: '16px' }}>
                  {m.name}
                </code>

                <div style={{ background: '#111111', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', marginBottom: '16px' }}>
                  <div style={{ fontSize: '11px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '6px' }}>
                    Hyperparameter Optimization
                  </div>
                  <div style={{ fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', color: m.has_search_space ? 'var(--success)' : 'var(--text-muted)' }}>
                    <FiSliders />
                    <span>{m.has_search_space ? 'Optuna Bayesian Search Space' : 'Fixed Defaults'}</span>
                  </div>
                </div>

                <div style={{ background: '#111111', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '11px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '6px' }}>
                    Default Parameters
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', maxHeight: '70px', overflowY: 'auto' }}>
                    {Object.keys(m.default_params || {}).length > 0 ? (
                      Object.entries(m.default_params).map(([k, v]) => (
                        <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>{k}:</span>
                          <span style={{ fontFamily: 'monospace' }}>{String(v)}</span>
                        </div>
                      ))
                    ) : (
                      <span>scikit-learn canonical defaults</span>
                    )}
                  </div>
                </div>
              </div>

              <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid var(--border)' }}>
                <button
                  className="btn-netflix-primary"
                  style={{ width: '100%', fontSize: '13px' }}
                  onClick={() => navigate('/dashboard?new=1')}
                >
                  <FiPlus /> Train in Experiment
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
