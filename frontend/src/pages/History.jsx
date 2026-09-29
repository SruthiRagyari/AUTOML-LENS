import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../services/api'
import StatusBadge from '../components/Common/StatusBadge'
import LoadingSpinner from '../components/Common/LoadingSpinner'
import EmptyState from '../components/Common/EmptyState'

export default function History() {
  const navigate = useNavigate()
  const [experiments, setExperiments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    api.listExperiments()
      .then(r => setExperiments(r.data.experiments || []))
      .catch(() => setError('Failed to load experiments'))
      .finally(() => setLoading(false))
  }, [])

  const filtered = filter === 'all' ? experiments : experiments.filter(e => e.status === filter)

  return (
    <div className="page">
      <div className="page-header">
        <div><h1>Experiment History</h1><p>All past AutoML experiments</p></div>
        <div style={{ display: 'flex', gap: 8 }}>
          {['all', 'completed', 'training', 'failed', 'created'].map(f => (
            <button key={f} className={`btn btn-sm${filter === f ? ' btn-primary' : ''}`} onClick={() => setFilter(f)}
              style={{ textTransform: 'capitalize' }}>{f}</button>
          ))}
        </div>
      </div>
      {error && <div className="alert alert-danger">{error}</div>}
      {loading ? <LoadingSpinner /> : filtered.length === 0 ? (
        <EmptyState icon="📭" title="No experiments found" description={filter === 'all' ? 'Create your first experiment from the dashboard.' : `No ${filter} experiments.`} />
      ) : (
        <div className="card">
          <div className="table-container">
            <table>
              <thead>
                <tr><th>#</th><th>Name</th><th>Target</th><th>Type</th><th>Mode</th><th>Best Model</th><th>Score</th><th>Metric</th><th>Status</th><th>Date</th><th></th></tr>
              </thead>
              <tbody>
                {filtered.map(e => (
                  <tr key={e.id}>
                    <td style={{ color: 'var(--text-muted)' }}>{e.id}</td>
                    <td><strong>{e.name}</strong></td>
                    <td>{e.target_column}</td>
                    <td>{e.problem_type || 'auto'}</td>
                    <td>{e.mode}</td>
                    <td>{e.best_model_name || '—'}</td>
                    <td>{e.best_score?.toFixed(4) ?? '—'}</td>
                    <td style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{e.primary_metric || '—'}</td>
                    <td><StatusBadge status={e.status} /></td>
                    <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>{new Date(e.created_at).toLocaleDateString('en-IN')}</td>
                    <td><button className="btn btn-sm btn-outline" onClick={() => navigate(`/experiment/${e.id}`)}>View →</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
