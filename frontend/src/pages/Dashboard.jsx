import React, { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { FiPlus, FiZap } from 'react-icons/fi'
import { api } from '../services/api'
import StatusBadge from '../components/Common/StatusBadge'
import LoadingSpinner from '../components/Common/LoadingSpinner'
import EmptyState from '../components/Common/EmptyState'
import ScrollRow, { ExperimentTile, DatasetTile } from '../components/Common/ScrollRow'

const DEFAULT_METRICS = {
  classification: ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'balanced_accuracy', 'roc_auc'],
  regression: ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],
}

function pctText(n) {
  return n == null ? '—' : `${Number(n).toFixed(1)}%`
}

/**
 * Real suitability report for the uploaded file.
 *
 * Every value here comes straight from GET /api/datasets/{id}/suitability.
 * Blocking issues mean the dataset cannot drive a meaningful experiment, so
 * the Create button is disabled and the reasons are listed. Nothing is
 * inferred client-side and nothing is invented when a field is null.
 */
function SuitabilityReport({ suit, form }) {
  if (!suit) return null
  const t = suit.target || {}
  const blocking = suit.blocking_issues || []
  const warnings = [...(suit.warnings || []), ...(suit.warnings_from_profile || [])]
  const types = suit.column_type_summary || {}

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h2>Dataset Readiness</h2>

      <div className={`alert ${suit.suitable ? 'alert-success' : 'alert-danger'}`} style={{ marginBottom: 16 }}>
        {suit.suitable
          ? <>✅ This dataset can be used for AutoML{suit.task_type ? <> — detected task: <strong>{suit.task_type}</strong></> : null}.</>
          : <>⛔ This dataset cannot be used for AutoML yet. Fix the issues below and re-upload or change the target.</>}
        {suit.task_reason && (
          <div style={{ fontSize: 12, marginTop: 6, opacity: 0.85 }}>{suit.task_reason}</div>
        )}
      </div>

      <div className="metrics-grid" style={{ marginBottom: 16 }}>
        <div className="metric-card"><div className="metric-value">{suit.rows}</div><div className="metric-label">Rows</div></div>
        <div className="metric-card"><div className="metric-value">{suit.columns}</div><div className="metric-label">Columns</div></div>
        <div className="metric-card"><div className="metric-value">{suit.usable_feature_count}</div><div className="metric-label">Usable features</div></div>
        <div className="metric-card"><div className="metric-value">{pctText(suit.total_missing_percentage)}</div><div className="metric-label">Missing cells</div></div>
      </div>

      {Object.keys(types).length > 0 && (
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
          Column types:{' '}
          {Object.entries(types).map(([k, v]) => <span key={k} className="tag" style={{ marginRight: 6 }}>{k}: {v}</span>)}
          {suit.duplicate_rows > 0 && (
            <span className="tag" style={{ marginLeft: 6 }}>duplicate rows: {suit.duplicate_rows} ({suit.duplicate_percentage}%)</span>
          )}
        </div>
      )}

      {blocking.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div className="section-title">Blocking issues</div>
          <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, lineHeight: 1.8, color: 'var(--danger)' }}>
            {blocking.map((b, i) => <li key={i}>{b}</li>)}
          </ul>
        </div>
      )}

      {t.found && (
        <div style={{ marginBottom: 16 }}>
          <div className="section-title">Target &ldquo;{t.name}&rdquo;</div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 8 }}>
            inferred type <code>{t.inferred_type ?? 'unknown'}</code>
            {' '}· {t.unique_count} distinct value(s)
            {' '}· {pctText(t.missing_percentage)} missing
            {t.summary ? <> · {t.summary}</> : null}
          </div>
          {t.distribution_kind === 'classes' && t.distribution && (
            <>
              <div className="table-container" style={{ maxHeight: 260, overflowY: 'auto' }}>
                <table>
                  <thead><tr><th>Class</th><th>Rows</th><th>Share</th></tr></thead>
                  <tbody>
                    {Object.entries(t.distribution).map(([k, v]) => (
                      <tr key={k}>
                        <td><code>{k}</code></td>
                        <td>{v}</td>
                        <td>{pctText((v / suit.rows) * 100)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {t.distribution_truncated && (
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6 }}>
                  Showing the 20 most frequent values; the rest are not listed.
                </div>
              )}
            </>
          )}
        </div>
      )}

      {warnings.length > 0 && (
        <div>
          <div className="section-title">Warnings</div>
          <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, lineHeight: 1.8, color: 'var(--text-secondary)' }}>
            {warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [experiments, setExperiments] = useState([])
  const [datasetsById, setDatasetsById] = useState({})
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [step, setStep] = useState(1) // 1=upload, 2=configure, 3=creating
  const [uploadedDataset, setUploadedDataset] = useState(null)
  const [columns, setColumns] = useState([])
  const [profileData, setProfileData] = useState(null)
  const [form, setForm] = useState({
    name: '', target_column: '', problem_type: 'auto', mode: 'automl',
    primary_metric: '', n_folds: 5, n_trials: 10, task_description: ''
  })
  const [uploading, setUploading] = useState(false)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  const [suitability, setSuitability] = useState(null)
  const [suitLoading, setSuitLoading] = useState(false)

  // The backend stays the source of truth for whether this dataset can be
  // used: re-ask it whenever the target or task choice changes.
  const refreshSuitability = useCallback(async (datasetId, target, ptype) => {
    if (!datasetId) return
    setSuitLoading(true)
    try {
      const r = await api.getSuitability(datasetId, target, ptype)
      setSuitability(r.data)
    } catch {
      setSuitability(null)
    } finally {
      setSuitLoading(false)
    }
  }, [])

  useEffect(() => {
    if (uploadedDataset?.id) {
      refreshSuitability(uploadedDataset.id, form.target_column, form.problem_type)
    } else {
      setSuitability(null)
    }
  }, [uploadedDataset?.id, form.target_column, form.problem_type, refreshSuitability])

  const fetchExperiments = async () => {
    try {
      setLoading(true)
      const r = await api.listExperiments()
      const list = r.data.experiments || []
      setExperiments(list)
      // Dataset names are not part of the experiment payload, so ask the
      // dataset endpoint for the ids we actually see. Failures are skipped.
      const ids = [...new Set(list.map(e => e.dataset_id).filter(Boolean))]
      const pairs = await Promise.all(ids.map(async id => {
        try { const d = await api.getDataset(id); return [id, d.data] } catch { return null }
      }))
      setDatasetsById(Object.fromEntries(pairs.filter(Boolean)))
    } catch { setError('Failed to load experiments') }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchExperiments() }, [])

  const stats = {
    total: experiments.length,
    completed: experiments.filter(e => e.status === 'completed').length,
    running: experiments.filter(e => ['training', 'analyzing', 'preprocessing'].includes(e.status)).length,
    bestScore: experiments.filter(e => e.best_score != null).reduce((best, e) => {
      return e.best_score > (best?.best_score ?? -Infinity) ? e : best
    }, null)?.best_score?.toFixed(4) ?? '—',
  }

  const onDrop = useCallback(async (files) => {
    if (!files.length) return
    setUploading(true); setError('')
    try {
      const r = await api.uploadDataset(files[0])
      setUploadedDataset(r.data)
      setColumns(r.data.column_names || [])
      setForm(f => ({ ...f, name: `Experiment on ${r.data.original_filename}` }))
      // Auto-profile to get suggested target
      try {
        const pr = await api.profileDataset(r.data.id)
        setProfileData(pr.data)
        if (pr.data.suggested_target) {
          setForm(f => ({
            ...f,
            target_column: pr.data.suggested_target,
            problem_type: pr.data.suggested_problem_type || 'auto',
          }))
        }
      } catch {}
      setStep(2)
    } catch (e) {
      setError(`Upload failed: ${e.response?.data?.detail || e.message}`)
    } finally { setUploading(false) }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'text/csv': ['.csv'], 'application/vnd.ms-excel': ['.xls'], 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] },
    maxFiles: 1,
  })

  const handleMetricChange = (ptype) => {
    const pt = ptype || form.problem_type
    // "Auto-detect" must leave the metric unset so the backend resolves it
    // from the detected task; injecting a concrete metric here scored
    // classification experiments with neg_RMSE.
    const defaultMetric =
      pt === 'classification' ? 'f1_weighted'
        : pt === 'regression' ? 'neg_root_mean_squared_error'
          : ''
    setForm(f => ({ ...f, problem_type: pt, primary_metric: defaultMetric }))
  }

  const handleCreate = async (e) => {
    e.preventDefault()
    if (!uploadedDataset) { setError('Please upload a dataset first'); return }
    if (!form.target_column) { setError('Target column is required'); return }
    setCreating(true); setStep(3)
    try {
      const payload = {
        dataset_id: uploadedDataset.id,
        name: form.name || `Experiment on ${uploadedDataset.original_filename}`,
        target_column: form.target_column,
        // "auto" and "Auto (based on problem type)" are real values the
        // backend understands; sending null for them fails validation (422).
        problem_type: form.problem_type,
        primary_metric: form.primary_metric || null,
        mode: form.mode,
        n_folds: parseInt(form.n_folds),
        n_trials: parseInt(form.n_trials),
        task_description: form.task_description,
      }
      const r = await api.createExperiment(payload)
      navigate(`/experiment/${r.data.id}`)
    } catch (e) {
      setError(`Failed to create experiment: ${e.response?.data?.detail || e.message}`)
      setCreating(false); setStep(2)
    }
  }

  const availableMetrics = DEFAULT_METRICS[form.problem_type] || [...DEFAULT_METRICS.classification, ...DEFAULT_METRICS.regression]

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Dashboard</h1>
          <p>Manage your AutoML experiments</p>
        </div>
        <button className="btn btn-primary" onClick={() => { setShowForm(v => !v); setStep(1); setUploadedDataset(null); setError('') }}>
          <FiPlus /> New Experiment
        </button>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <div className="card stat-card"><div className="stat-icon">🧪</div><div className="stat-value">{stats.total}</div><div className="stat-label">Total Experiments</div></div>
        <div className="card stat-card"><div className="stat-icon">✅</div><div className="stat-value">{stats.completed}</div><div className="stat-label">Completed</div></div>
        <div className="card stat-card"><div className="stat-icon">⚡</div><div className="stat-value">{stats.running}</div><div className="stat-label">Running</div></div>
        <div className="card stat-card"><div className="stat-icon">🏆</div><div className="stat-value" style={{ fontSize: 24 }}>{stats.bestScore}</div><div className="stat-label">Best Score</div></div>
      </div>

      {/* Scrollable rows - every card is built from API data */}
      <ScrollRow
        title="Recent experiments"
        items={experiments.slice(0, 12)}
        loading={loading}
        error={showForm ? '' : error}
        emptyTitle="No experiments yet"
        emptyDescription="Upload a dataset and start your first AutoML experiment."
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

      {/* New Experiment Form */}
      {showForm && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h2>New Experiment</h2>
          {error && <div className="alert alert-danger">{error}</div>}

          {step === 1 && (
            <div>
              <p style={{ color: 'var(--text-secondary)', marginBottom: 20, fontSize: 14 }}>Upload a CSV or Excel dataset to begin.</p>
              <div {...getRootProps()} className={`dropzone${isDragActive ? ' active' : ''}`}>
                <input {...getInputProps()} />
                <div className="dropzone-icon">📁</div>
                {uploading ? <p>Uploading & profiling...</p> : isDragActive ? <p>Drop it here...</p> : (
                  <p>Drag & drop your CSV/XLSX file here, or click to browse</p>
                )}
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>Max 100MB · CSV or Excel</p>
              </div>
            </div>
          )}

          {step === 2 && uploadedDataset && (
            <form onSubmit={handleCreate}>
              <div className="alert alert-success" style={{ marginBottom: 20 }}>
                ✅ Uploaded: <strong>{uploadedDataset.original_filename}</strong> — {uploadedDataset.rows.toLocaleString()} rows × {uploadedDataset.columns} columns
                {profileData?.suggested_target && ` · Suggested target: ${profileData.suggested_target}`}
              </div>

              <SuitabilityReport suit={suitability} />
              {suitLoading && !suitability && (
                <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Checking dataset readiness...</p>
              )}

              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Experiment Name</label>
                  <input className="form-input" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} required />
                </div>
                <div className="form-group">
                  <label className="form-label">Target Column *</label>
                  <select className="form-select" value={form.target_column} onChange={e => setForm({ ...form, target_column: e.target.value })} required>
                    <option value="">— Select target —</option>
                    {columns.map(c => <option key={c} value={c}>{c}{c === profileData?.suggested_target ? ' (suggested)' : ''}</option>)}
                  </select>
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Problem Type</label>
                  <select className="form-select" value={form.problem_type} onChange={e => handleMetricChange(e.target.value)}>
                    <option value="auto">Auto-detect</option>
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Primary Metric</label>
                  <select className="form-select" value={form.primary_metric} onChange={e => setForm({ ...form, primary_metric: e.target.value })}>
                    <option value="">Auto (based on problem type)</option>
                    {availableMetrics.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Mode</label>
                  <select className="form-select" value={form.mode} onChange={e => setForm({ ...form, mode: e.target.value })}>
                    <option value="baseline">Baseline (default params only)</option>
                    <option value="automl">AutoML (with Optuna optimization)</option>
                    <option value="llm_assisted">LLM Assisted (LLM model selection)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">CV Folds</label>
                  <select className="form-select" value={form.n_folds} onChange={e => setForm({ ...form, n_folds: e.target.value })}>
                    <option value="3">3-Fold (Fast)</option>
                    <option value="5">5-Fold (Default)</option>
                    <option value="10">10-Fold (Rigorous)</option>
                  </select>
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Optuna Trials per Model</label>
                  <select className="form-select" value={form.n_trials} onChange={e => setForm({ ...form, n_trials: e.target.value })}>
                    <option value="5">5 (Fast demo)</option>
                    <option value="10">10 (Default)</option>
                    <option value="20">20 (Thorough)</option>
                    <option value="50">50 (Extensive)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Task Description (optional)</label>
                  <input className="form-input" value={form.task_description} onChange={e => setForm({ ...form, task_description: e.target.value })}
                    placeholder="e.g., Predict customer churn for telecom..." />
                </div>
              </div>
              <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                <button type="submit" className="btn btn-primary"
                  disabled={creating || (suitability && !suitability.suitable) || !form.target_column}
                  title={
                    suitability && !suitability.suitable
                      ? 'Fix the blocking issues above before creating an experiment.'
                      : ''
                  }>
                  ▶ Create Experiment
                </button>
                <button type="button" className="btn" onClick={() => { setStep(1); setUploadedDataset(null); setSuitability(null) }}>
                  ← Change File
                </button>
                {suitability && !suitability.suitable && (
                  <span style={{ fontSize: 13, color: 'var(--danger)' }}>
                    Creation blocked: {suitability.blocking_issues[0]}
                  </span>
                )}
              </div>
            </form>
          )}

          {step === 3 && (
            <div className="loading-container">
              <div className="spinner" />
              <p>Creating experiment and redirecting...</p>
            </div>
          )}
        </div>
      )}

      {/* Recent Experiments */}
      <div className="card">
        <h2>Recent Experiments</h2>
        {loading ? <LoadingSpinner message="Loading experiments..." /> :
          error && !showForm ? <div className="alert alert-danger">{error}</div> :
            experiments.length === 0 ? (
            <EmptyState icon="🧪" title="No experiments yet" description="Upload a dataset and start your first AutoML experiment."
              action={<button className="btn btn-primary" onClick={() => setShowForm(true)}><FiPlus /> New Experiment</button>} />
          ) : (
            <div className="table-container">
              <table>
                <thead>
                  <tr><th>Name</th><th>Target</th><th>Type</th><th>Mode</th><th>Best Model</th><th>Score</th><th>Status</th><th>Date</th><th></th></tr>
                </thead>
                <tbody>
                  {experiments.map(e => (
                    <tr key={e.id}>
                      <td><strong>{e.name}</strong></td>
                      <td>{e.target_column}</td>
                      <td>{e.problem_type || 'auto'}</td>
                      <td>{e.mode}</td>
                      <td>{e.best_model_name || '—'}</td>
                      <td>{e.best_score?.toFixed(4) ?? '—'}</td>
                      <td><StatusBadge status={e.status} /></td>
                      <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>{new Date(e.created_at).toLocaleDateString()}</td>
                      <td><button className="btn btn-sm btn-outline" onClick={() => navigate(`/experiment/${e.id}`)}>View →</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
      </div>
    </div>
  )
}
