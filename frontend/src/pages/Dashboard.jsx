import React, { useState, useEffect, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import {
  FiPlus,
  FiZap,
  FiTrash2,
  FiPlay,
  FiCheckCircle,
  FiAlertTriangle,
  FiArrowRight,
  FiDatabase,
  FiFilter,
  FiSearch,
  FiFileText,
  FiSliders,
  FiX
} from 'react-icons/fi'
import { api } from '../services/api'
import StatusBadge from '../components/Common/StatusBadge'
import LoadingSpinner from '../components/Common/LoadingSpinner'

const DEFAULT_METRICS = {
  classification: ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'balanced_accuracy', 'roc_auc'],
  regression: ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],
}

function pctText(n) {
  return n == null ? '—' : `${Number(n).toFixed(1)}%`
}

// The profile endpoint reports the columns as `column_profiles`
// ([{ name, ... }]); `column_names` only exists on the *upload* response.
// Reading the wrong key left the Target Column <select> with no options, so
// the dropdown must always be derived from the profiles.
function columnNamesFromProfile(profile) {
  return (profile?.column_profiles || [])
    .map((cp) => cp?.name)
    .filter(Boolean)
}

function SuitabilityReport({ suit }) {
  if (!suit) return null
  const t = suit.target || {}
  const blocking = suit.blocking_issues || []
  const warnings = [...(suit.warnings || []), ...(suit.warnings_from_profile || [])]
  const types = suit.column_type_summary || {}

  return (
    <div className="netflix-card" style={{ marginTop: 16, border: suit.suitable ? '1px solid var(--success)' : '1px solid var(--danger)' }}>
      <h3 style={{ fontSize: '16px', fontWeight: '800', marginBottom: '12px' }}>Dataset Readiness</h3>

      <div style={{ padding: '12px', borderRadius: 'var(--radius-sm)', marginBottom: '16px', background: suit.suitable ? 'var(--success-bg)' : 'var(--danger-bg)', border: `1px solid ${suit.suitable ? 'var(--success)' : 'var(--danger)'}` }}>
        <div style={{ fontWeight: '800', fontSize: '14px', color: suit.suitable ? 'var(--success)' : 'var(--danger)' }}>
          {suit.suitable
            ? `✅ Suitable for AutoML (Detected task: ${suit.task_type || 'Classification/Regression'})`
            : '⛔ Dataset cannot be used yet. Resolve blocking issues below.'}
        </div>
        {suit.task_reason && (
          <div style={{ fontSize: 12, marginTop: 4, color: '#e0e0e0' }}>{suit.task_reason}</div>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
        <div style={{ background: '#111', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Rows</div>
          <div style={{ fontSize: '16px', fontWeight: '800' }}>{suit.rows?.toLocaleString()}</div>
        </div>
        <div style={{ background: '#111', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Columns</div>
          <div style={{ fontSize: '16px', fontWeight: '800' }}>{suit.columns}</div>
        </div>
        <div style={{ background: '#111', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Usable Features</div>
          <div style={{ fontSize: '16px', fontWeight: '800' }}>{suit.usable_feature_count}</div>
        </div>
        <div style={{ background: '#111', padding: '10px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Missing Cells</div>
          <div style={{ fontSize: '16px', fontWeight: '800' }}>{pctText(suit.total_missing_percentage)}</div>
        </div>
      </div>

      {blocking.length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: '13px', fontWeight: '700', color: 'var(--danger)', marginBottom: '4px' }}>Blocking Issues:</div>
          <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12, lineHeight: 1.6, color: 'var(--danger)' }}>
            {blocking.map((b, i) => <li key={i}>{b}</li>)}
          </ul>
        </div>
      )}

      {warnings.length > 0 && (
        <div>
          <div style={{ fontSize: '13px', fontWeight: '700', color: 'var(--warning)', marginBottom: '4px' }}>Warnings:</div>
          <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12, lineHeight: 1.6, color: 'var(--text-secondary)' }}>
            {warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}

export default function Dashboard() {
  const [experiments, setExperiments] = useState([])
  const [datasetsById, setDatasetsById] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filterType, setFilterType] = useState('all')
  const [search, setSearch] = useState('')

  // New Experiment Form State
  const [showForm, setShowForm] = useState(false)
  const [step, setStep] = useState(1)
  const [uploading, setUploading] = useState(false)
  const [uploadedDataset, setUploadedDataset] = useState(null)
  const [profileData, setProfileData] = useState(null)
  const [columns, setColumns] = useState([])
  const [suitability, setSuitability] = useState(null)
  const [suitLoading, setSuitLoading] = useState(false)
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState({
    name: '',
    target_column: '',
    problem_type: 'auto',
    primary_metric: '',
    mode: 'automl',
    n_folds: '5',
    n_trials: '10',
    task_description: '',
  })

  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    loadData()
    const params = new URLSearchParams(location.search)
    if (params.get('new') === '1') {
      setShowForm(true)
    }
  }, [location.search])

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [expRes, dsRes] = await Promise.allSettled([
        api.listExperiments(),
        api.listDatasets(),
      ])

      if (expRes.status === 'fulfilled') {
        setExperiments(expRes.value.data?.experiments || expRes.value.data || [])
      } else {
        setError('Failed to load experiments')
      }

      if (dsRes.status === 'fulfilled') {
        const map = {}
        ;(dsRes.value.data || []).forEach((d) => {
          map[d.id] = d
        })
        setDatasetsById(map)
      }
    } catch (err) {
      setError(err.message || 'Error loading dashboard data')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (expId, e) => {
    e.stopPropagation()
    if (!window.confirm(`Are you sure you want to delete experiment #${expId}?`)) return
    try {
      await api.deleteExperiment(expId)
      await loadData()
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to delete experiment')
    }
  }

  const MAX_DATASET_SIZE_BYTES = 1024 * 1024 * 1024 // 1 GB

  // Upload handling
  const onDrop = useCallback(async (acceptedFiles, fileRejections) => {
    if (fileRejections && fileRejections.length > 0) {
      const rej = fileRejections[0]
      if (rej.errors?.some((e) => e.code === 'file-too-large')) {
        setError('Dataset exceeds the maximum allowed size of 1 GB.')
      } else {
        setError(rej.errors?.[0]?.message || 'File rejected')
      }
      return
    }
    if (!acceptedFiles || acceptedFiles.length === 0) return
    const file = acceptedFiles[0]
    if (file.size > MAX_DATASET_SIZE_BYTES) {
      setError('Dataset exceeds the maximum allowed size of 1 GB.')
      return
    }
    setUploading(true)
    setError(null)

    try {
      const upRes = await api.uploadDataset(file)
      const ds = upRes.data
      setUploadedDataset(ds)

      const profRes = await api.profileDataset(ds.id)
      const profile = profRes.data || {}
      setProfileData(profile)

      // Derive the option list from the real profile, with a fallback to the
      // dedicated /columns endpoint if the profile ever lacks them.
      let cols = columnNamesFromProfile(profile)
      if (cols.length === 0) {
        try {
          const colRes = await api.getColumns(ds.id)
          cols = colRes.data?.columns || []
        } catch (colErr) {
          console.error('Column list fallback failed:', colErr)
        }
      }
      setColumns(cols)

      // Only pre-select the suggestion when it is a real option, so the
      // <select> value always matches a rendered <option> (otherwise the
      // browser blocks submission with "Select an item in the list").
      const suggested = profile.suggested_target
      const initialTarget = cols.includes(suggested) ? suggested : ''
      const initialType = profile.suggested_problem_type || 'auto'

      setForm((prev) => ({
        ...prev,
        name: `${file.name.replace(/\.[^/.]+$/, '')} AutoML`,
        target_column: initialTarget,
        problem_type: initialType,
      }))

      setStep(2)
      checkSuitability(ds.id, initialTarget, initialType)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Dataset upload failed')
    } finally {
      setUploading(false)
    }
  }, [])

  const checkSuitability = async (dsId, target, type) => {
    if (!dsId) return
    setSuitLoading(true)
    try {
      const res = await api.getSuitability(dsId, target, type)
      setSuitability(res.data)
    } catch (err) {
      console.error('Suitability error:', err)
    } finally {
      setSuitLoading(false)
    }
  }

  const handleTargetChange = (target) => {
    setForm((prev) => ({ ...prev, target_column: target }))
    if (uploadedDataset) {
      checkSuitability(uploadedDataset.id, target, form.problem_type)
    }
  }

  const handleCreate = async (e) => {
    e.preventDefault()
    if (!uploadedDataset) return
    setCreating(true)
    setError(null)

    try {
      const payload = {
        name: form.name,
        dataset_id: uploadedDataset.id,
        target_column: form.target_column,
        problem_type: form.problem_type === 'auto' ? null : form.problem_type,
        primary_metric: form.primary_metric || null,
        mode: form.mode,
        n_folds: Number(form.n_folds),
        n_trials: Number(form.n_trials),
        task_description: form.task_description || null,
      }
      const res = await api.createExperiment(payload)
      navigate(`/experiment/${res.data.id}`)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to create experiment')
      setCreating(false)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    maxSize: MAX_DATASET_SIZE_BYTES,
    maxFiles: 1,
    disabled: uploading,
  })

  // Filtered experiments
  const filteredExps = experiments.filter((e) => {
    const matchesFilter =
      filterType === 'all' ||
      (filterType === 'classification' && e.problem_type === 'classification') ||
      (filterType === 'regression' && e.problem_type === 'regression') ||
      (filterType === 'completed' && e.status === 'completed')

    const matchesSearch =
      e.name?.toLowerCase().includes(search.toLowerCase()) ||
      e.target_column?.toLowerCase().includes(search.toLowerCase()) ||
      String(e.id).includes(search)

    return matchesFilter && matchesSearch
  })

  const latestCompleted = experiments.find((e) => e.status === 'completed')

  return (
    <div className="page-container">
      {/* Featured Netflix Hero Banner */}
      {latestCompleted ? (
        <div className="netflix-hero-banner" style={{ marginBottom: '36px' }}>
          <div className="hero-pill-badge">
            <FiCheckCircle /> Top Performing Experiment
          </div>
          <h1 className="hero-title">{latestCompleted.name || `Experiment #${latestCompleted.id}`}</h1>
          <p className="hero-desc">
            Evaluated on <strong>{datasetsById[latestCompleted.dataset_id]?.original_filename || 'Tabular Data'}</strong> · Target: <strong>{latestCompleted.target_column}</strong> ({latestCompleted.problem_type})
            {latestCompleted.best_score && ` · Best Score: ${latestCompleted.best_score.toFixed(4)} (${latestCompleted.best_model_name || 'Ensemble'})`}
          </p>
          <div className="hero-actions">
            <button
              className="btn-netflix-primary"
              onClick={() => navigate(`/experiment/${latestCompleted.id}`)}
            >
              <FiPlay /> View Experiment & Results
            </button>
            <button
              className="btn-netflix-secondary"
              onClick={() => setShowForm(true)}
            >
              <FiPlus /> New Experiment
            </button>
          </div>
        </div>
      ) : (
        <div className="netflix-hero-banner" style={{ marginBottom: '36px' }}>
          <div className="hero-pill-badge">AutoML-Lens Studio</div>
          <h1 className="hero-title">Start Machine Learning</h1>
          <p className="hero-desc">
            Launch end-to-end automated machine learning with LLM-guided pipeline planning,
            hyperparameter optimization, and explainable AI.
          </p>
          <div className="hero-actions">
            <button
              className="btn-netflix-primary"
              onClick={() => setShowForm(true)}
            >
              <FiPlus /> Create Experiment
            </button>
            <button
              className="btn-netflix-secondary"
              onClick={() => navigate('/datasets')}
            >
              <FiDatabase /> Browse Datasets
            </button>
          </div>
        </div>
      )}

      {/* Filter and Search Controls */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', marginBottom: '24px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className={`btn-netflix-secondary ${filterType === 'all' ? 'active' : ''}`}
            onClick={() => setFilterType('all')}
            style={filterType === 'all' ? { borderColor: 'var(--netflix-red)', background: 'var(--netflix-red-subtle)', color: '#fff' } : {}}
          >
            All Experiments ({experiments.length})
          </button>
          <button
            className={`btn-netflix-secondary ${filterType === 'completed' ? 'active' : ''}`}
            onClick={() => setFilterType('completed')}
            style={filterType === 'completed' ? { borderColor: 'var(--netflix-red)', background: 'var(--netflix-red-subtle)', color: '#fff' } : {}}
          >
            Completed ({experiments.filter((e) => e.status === 'completed').length})
          </button>
          <button
            className={`btn-netflix-secondary ${filterType === 'classification' ? 'active' : ''}`}
            onClick={() => setFilterType('classification')}
            style={filterType === 'classification' ? { borderColor: 'var(--netflix-red)', background: 'var(--netflix-red-subtle)', color: '#fff' } : {}}
          >
            Classification
          </button>
          <button
            className={`btn-netflix-secondary ${filterType === 'regression' ? 'active' : ''}`}
            onClick={() => setFilterType('regression')}
            style={filterType === 'regression' ? { borderColor: 'var(--netflix-red)', background: 'var(--netflix-red-subtle)', color: '#fff' } : {}}
          >
            Regression
          </button>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div style={{ position: 'relative', width: '240px' }}>
            <FiSearch style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
            <input
              type="text"
              className="netflix-input"
              placeholder="Search experiments..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ paddingLeft: '36px', fontSize: '13px' }}
            />
          </div>

          <button
            className="btn-netflix-primary"
            onClick={() => setShowForm(true)}
            title="Create New Experiment"
          >
            <FiPlus /> New
          </button>
        </div>
      </div>

      {/* Experiments Grid */}
      {loading ? (
        <div style={{ padding: '60px', textAlign: 'center' }}><LoadingSpinner /></div>
      ) : filteredExps.length === 0 ? (
        <div className="netflix-card" style={{ padding: '60px', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <FiZap style={{ fontSize: '42px', color: 'var(--netflix-red)', marginBottom: '16px' }} />
          <h3>No experiments found</h3>
          <p style={{ marginTop: '8px', marginBottom: '20px' }}>Upload a dataset to train your first model!</p>
          <button className="btn-netflix-primary" onClick={() => setShowForm(true)}>
            <FiPlus /> Create Experiment
          </button>
        </div>
      ) : (
        <div className="netflix-grid" style={{ marginBottom: '40px' }}>
          {filteredExps.map((exp) => (
            <div
              key={exp.id}
              className="netflix-card netflix-card-featured"
              style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', cursor: 'pointer' }}
              onClick={() => navigate(`/experiment/${exp.id}`)}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <span style={{ fontSize: '12px', fontWeight: '800', color: 'var(--netflix-red)' }}>
                    #{exp.id}
                  </span>
                  <StatusBadge status={exp.status} />
                </div>

                <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '6px', color: '#ffffff' }}>
                  {exp.name || `Experiment ${exp.id}`}
                </h3>

                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '14px' }}>
                  Dataset: {datasetsById[exp.dataset_id]?.original_filename || `Dataset #${exp.dataset_id}`}
                </p>

                <div style={{ background: '#111111', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', marginBottom: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Target:</span>
                    <strong style={{ color: '#fff' }}>{exp.target_column}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Task Type:</span>
                    <span style={{ textTransform: 'capitalize' }}>{exp.problem_type || 'Auto'}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Best Score:</span>
                    <strong style={{ color: 'var(--success)' }}>
                      {exp.best_score ? exp.best_score.toFixed(4) : '—'}
                    </strong>
                  </div>
                  {exp.best_model_name && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginTop: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Best Model:</span>
                      <span style={{ color: '#fff' }}>{exp.best_model_name}</span>
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '12px', borderTop: '1px solid var(--border)' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  {exp.created_at ? new Date(exp.created_at).toLocaleDateString() : ''}
                </span>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    className="btn-netflix-secondary"
                    style={{ padding: '6px 12px', fontSize: '12px' }}
                    onClick={(e) => {
                      e.stopPropagation()
                      navigate(`/experiment/${exp.id}`)
                    }}
                  >
                    View →
                  </button>
                  <button
                    className="btn-netflix-danger"
                    style={{ padding: '6px 10px' }}
                    onClick={(e) => handleDelete(exp.id, e)}
                    title="Delete Experiment"
                  >
                    <FiTrash2 />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* New Experiment Modal */}
      {showForm && (
        <div className="netflix-modal-backdrop" onClick={() => setShowForm(false)}>
          <div className="netflix-modal" style={{ maxWidth: '720px' }} onClick={(e) => e.stopPropagation()}>
            <div className="netflix-modal-header">
              <h2 className="netflix-modal-title">Create New AutoML Experiment</h2>
              <button className="btn-netflix-ghost" onClick={() => setShowForm(false)}>
                <FiX />
              </button>
            </div>

            <div className="netflix-modal-body">
              {error && (
                <div style={{ padding: '12px', background: 'var(--danger-bg)', border: '1px solid var(--danger)', borderRadius: 'var(--radius-sm)', color: '#ff6b72', fontSize: '13px', marginBottom: '16px' }}>
                  <FiAlertTriangle style={{ marginRight: '6px' }} />
                  {error}
                </div>
              )}

              {step === 1 && (
                <div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '16px' }}>
                    Upload a CSV or Excel dataset to begin the automated machine learning workflow.
                  </p>
                  <div
                    {...getRootProps()}
                    className={`netflix-dropzone ${isDragActive ? 'active' : ''}`}
                  >
                    <input {...getInputProps()} />
                    <FiDatabase style={{ fontSize: '42px', color: isDragActive ? 'var(--netflix-red)' : 'var(--text-secondary)', marginBottom: '12px' }} />
                    {uploading ? (
                      <div>
                        <LoadingSpinner />
                        <p style={{ marginTop: '12px', color: 'var(--text-secondary)' }}>Uploading and profiling dataset...</p>
                      </div>
                    ) : (
                      <div>
                        <p style={{ fontSize: '16px', fontWeight: '600', marginBottom: '6px' }}>
                          Drag & drop your CSV or Excel file here
                        </p>
                        <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                          Supports CSV, XLSX up to 1 GB
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {step === 2 && uploadedDataset && (
                <form onSubmit={handleCreate}>
                  <div style={{ padding: '14px', background: 'var(--success-bg)', border: '1px solid var(--success)', borderRadius: 'var(--radius-sm)', color: '#e0e0e0', fontSize: '13px', marginBottom: '16px' }}>
                    <strong>✅ {uploadedDataset.original_filename}</strong> ({uploadedDataset.rows?.toLocaleString()} rows × {uploadedDataset.columns} cols)
                  </div>

                  <SuitabilityReport suit={suitability} />

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '16px' }}>
                    <div>
                      <label style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '6px', display: 'block' }}>
                        Experiment Name *
                      </label>
                      <input
                        type="text"
                        className="netflix-input"
                        value={form.name}
                        onChange={(e) => setForm({ ...form, name: e.target.value })}
                        required
                      />
                    </div>

                    <div>
                      <label style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '6px', display: 'block' }}>
                        Target Column *
                      </label>
                      <select
                        className="netflix-select"
                        value={form.target_column}
                        onChange={(e) => handleTargetChange(e.target.value)}
                        required
                      >
                        <option value="">— Select Target Column —</option>
                        {columns.map((c) => (
                          <option key={c} value={c}>
                            {c} {c === profileData?.suggested_target ? ' (Suggested)' : ''}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '6px', display: 'block' }}>
                        Problem Type
                      </label>
                      <select
                        className="netflix-select"
                        value={form.problem_type}
                        onChange={(e) => {
                          setForm({ ...form, problem_type: e.target.value })
                          if (uploadedDataset) checkSuitability(uploadedDataset.id, form.target_column, e.target.value)
                        }}
                      >
                        <option value="auto">Auto-detect</option>
                        <option value="classification">Classification</option>
                        <option value="regression">Regression</option>
                      </select>
                    </div>

                    <div>
                      <label style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '6px', display: 'block' }}>
                        Mode
                      </label>
                      <select
                        className="netflix-select"
                        value={form.mode}
                        onChange={(e) => setForm({ ...form, mode: e.target.value })}
                      >
                        <option value="automl">AutoML (Optuna Bayesian HPO)</option>
                        <option value="llm_assisted">LLM-Guided Pipeline Planning</option>
                        <option value="baseline">Baseline (Canonical Defaults)</option>
                      </select>
                    </div>

                    <div>
                      <label style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '6px', display: 'block' }}>
                        CV Folds
                      </label>
                      <select
                        className="netflix-select"
                        value={form.n_folds}
                        onChange={(e) => setForm({ ...form, n_folds: e.target.value })}
                      >
                        <option value="3">3 Folds (Fast)</option>
                        <option value="5">5 Folds (Standard)</option>
                        <option value="10">10 Folds (Rigorous)</option>
                      </select>
                    </div>

                    <div>
                      <label style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '6px', display: 'block' }}>
                        Optuna Trials per Model
                      </label>
                      <select
                        className="netflix-select"
                        value={form.n_trials}
                        onChange={(e) => setForm({ ...form, n_trials: e.target.value })}
                      >
                        <option value="5">5 Trials (Fast demo)</option>
                        <option value="10">10 Trials (Balanced)</option>
                        <option value="20">20 Trials (Thorough)</option>
                      </select>
                    </div>
                  </div>

                  <div style={{ marginTop: '24px', display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
                    <button
                      type="button"
                      className="btn-netflix-secondary"
                      onClick={() => setStep(1)}
                    >
                      ← Change Dataset
                    </button>
                    <button
                      type="submit"
                      disabled={creating || !form.target_column || (suitability && !suitability.suitable)}
                      className="btn-netflix-primary"
                    >
                      {creating ? <LoadingSpinner size="sm" /> : <><FiPlay /> Launch Experiment</>}
                    </button>
                  </div>
                </form>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
