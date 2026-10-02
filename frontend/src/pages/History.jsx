import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../services/api'
import StatusBadge from '../components/Common/StatusBadge'
import LoadingSpinner from '../components/Common/LoadingSpinner'
import EmptyState from '../components/Common/EmptyState'
import ScrollRow, { ExperimentTile, DatasetTile } from '../components/Common/ScrollRow'

export default function History() {
  const navigate = useNavigate()
  const [experiments, setExperiments] = useState([])
  const [datasetsById, setDatasetsById] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    api.listExperiments()
      .then(r => {
        const list = r.data.experiments || []
        setExperiments(list)
        // Dataset names are not part of the experiment payload, so ask the
        // dataset endpoint for the ids we actually see. Failures are skipped.
        const ids = [...new Set(list.map(e => e.dataset_id).filter(Boolean))]
        return Promise.all(ids.map(async id => {
          try { const d = await api.getDataset(id); return [id, d.data] } catch { return null }
        }))
      })
      .then(pairs => { if (pairs) setDatasetsById(Object.fromEntries(pairs.filter(Boolean))) })
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
      {/* Scrollable rows - every card is built from API data */}
      <ScrollRow
        title="Recent experiments"
        items={experiments.slice(0, 12)}
        loading={loading}
        error={error}
        emptyTitle="No experiments yet"
        emptyDescription="Create your first experiment from the dashboard."
        renderItem={e => <ExperimentTile experiment={e} datasetName={datasetsById[e.dataset_id]?.original_filename} />}
      />
      <ScrollRow
        title="Completed"
        items={experiments.filter(e => e.status === 'completed').slice(0, 12)}
        loading={loading}
        emptyTitle="Nothing completed yet"
        emptyDescription="Experiments appear here once training finishes successfully."
        renderItem={e => <ExperimentTile experiment={e} datasetName={datasetsById[e.dataset_id]?.original_filename} />}
      />
      <ScrollRow
        title="Datasets"
        items={Object.values(datasetsById)}
        loading={loading}
        emptyTitle="No datasets yet"
        emptyDescription="Datasets you upload appear here."
        emptyIcon="📁"
        renderItem={d => <DatasetTile dataset={d} />}
      />

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
