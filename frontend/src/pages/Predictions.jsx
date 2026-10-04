import React, { useState, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  FiTarget,
  FiPlay,
  FiUploadCloud,
  FiCheckCircle,
  FiAlertCircle,
  FiDownload,
  FiLayers
} from 'react-icons/fi'
import { api } from '../services/api'
import LoadingSpinner from '../components/Common/LoadingSpinner'

export default function Predictions() {
  const [experiments, setExperiments] = useState([])
  const [selectedExpId, setSelectedExpId] = useState('')
  const [loading, setLoading] = useState(true)
  const [schema, setSchema] = useState(null)
  const [schemaLoading, setSchemaLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('single')

  // Single prediction state
  const [featureInputs, setFeatureInputs] = useState({})
  const [singleResult, setSingleResult] = useState(null)
  const [singleLoading, setSingleLoading] = useState(false)
  const [singleError, setSingleError] = useState(null)

  // Batch prediction state
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchResult, setBatchResult] = useState(null)
  const [batchError, setBatchError] = useState(null)

  useEffect(() => {
    loadCompletedExperiments()
  }, [])

  const loadCompletedExperiments = async () => {
    setLoading(true)
    try {
      const res = await api.listExperiments()
      const completed = (res.data?.experiments || res.data || []).filter((e) => e.status === 'completed')
      setExperiments(completed)
      if (completed.length > 0) {
        setSelectedExpId(String(completed[0].id))
        loadSchema(completed[0].id)
      }
    } catch (err) {
      console.error('Failed to load experiments:', err)
    } finally {
      setLoading(false)
    }
  }

  const loadSchema = async (expId) => {
    if (!expId) return
    setSchemaLoading(true)
    setSingleResult(null)
    setBatchResult(null)
    try {
      const res = await api.getInputSchema(expId)
      setSchema(res.data)
      const initial = {}
      if (res.data?.features) {
        res.data.features.forEach((f) => {
          initial[f.name] = f.type === 'numeric' ? '0' : ''
        })
      }
      setFeatureInputs(initial)
    } catch (err) {
      console.error('Failed to load schema:', err)
    } finally {
      setSchemaLoading(false)
    }
  }

  const handleExpChange = (id) => {
    setSelectedExpId(id)
    loadSchema(id)
  }

  const handleSinglePredict = async (e) => {
    e.preventDefault()
    if (!selectedExpId) return
    setSingleLoading(true)
    setSingleError(null)

    // Convert numeric inputs
    const payload = {}
    if (schema?.features) {
      schema.features.forEach((f) => {
        const val = featureInputs[f.name]
        if (f.type === 'numeric') {
          payload[f.name] = val === '' ? 0 : Number(val)
        } else {
          payload[f.name] = val || ''
        }
      })
    } else {
      Object.assign(payload, featureInputs)
    }

    try {
      const res = await api.predict(selectedExpId, payload)
      setSingleResult(res.data)
    } catch (err) {
      setSingleError(err.response?.data?.detail || err.message || 'Prediction failed')
    } finally {
      setSingleLoading(false)
    }
  }

  const onDropBatch = async (acceptedFiles) => {
    if (!acceptedFiles || acceptedFiles.length === 0 || !selectedExpId) return
    const file = acceptedFiles[0]
    setBatchLoading(true)
    setBatchError(null)
    setBatchResult(null)

    try {
      const res = await api.batchPredict(selectedExpId, file)
      setBatchResult(res.data)
    } catch (err) {
      setBatchError(err.response?.data?.detail || err.message || 'Batch prediction failed')
    } finally {
      setBatchLoading(false)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: onDropBatch,
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    disabled: batchLoading || !selectedExpId,
  })

  const selectedExp = experiments.find((e) => String(e.id) === String(selectedExpId))

  return (
    <div className="page-container">
      <div className="netflix-hero-banner" style={{ padding: '40px 36px', marginBottom: '32px' }}>
        <div className="hero-pill-badge">Inference & Serving</div>
        <h1 className="hero-title" style={{ fontSize: '36px' }}>Model Prediction Hub</h1>
        <p className="hero-desc" style={{ fontSize: '16px', maxWidth: '650px' }}>
          Execute real-time single record inference or batch CSV scoring using any trained and persisted
          AutoML model bundle.
        </p>
      </div>

      {/* Experiment Selector Bar */}
      <div className="netflix-card" style={{ marginBottom: '28px' }}>
        <label style={{ fontSize: '13px', fontWeight: '700', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px', display: 'block' }}>
          Select Trained Experiment
        </label>
        {loading ? (
          <LoadingSpinner />
        ) : experiments.length === 0 ? (
          <div style={{ color: 'var(--text-secondary)' }}>
            No completed experiments available for predictions. Run an experiment first!
          </div>
        ) : (
          <select
            className="netflix-select"
            value={selectedExpId}
            onChange={(e) => handleExpChange(e.target.value)}
          >
            {experiments.map((exp) => (
              <option key={exp.id} value={exp.id}>
                #{exp.id} - {exp.name || `Experiment ${exp.id}`} ({exp.problem_type}, best model: {exp.best_model_name || 'Ensemble'})
              </option>
            ))}
          </select>
        )}

        {selectedExp && (
          <div style={{ display: 'flex', gap: '16px', marginTop: '14px', fontSize: '13px', color: 'var(--text-secondary)' }}>
            <span>Task: <strong style={{ color: '#fff' }}>{selectedExp.problem_type}</strong></span>
            <span>Target: <strong style={{ color: '#fff' }}>{selectedExp.target_column}</strong></span>
            <span>Best Metric: <strong style={{ color: 'var(--success)' }}>{selectedExp.best_score ? selectedExp.best_score.toFixed(4) : 'N/A'}</strong></span>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
        <button
          className={`btn-netflix-secondary ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
          style={activeTab === 'single' ? { borderColor: 'var(--netflix-red)', background: 'var(--netflix-red-subtle)', color: '#fff' } : {}}
        >
          Single Record Inference
        </button>
        <button
          className={`btn-netflix-secondary ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
          style={activeTab === 'batch' ? { borderColor: 'var(--netflix-red)', background: 'var(--netflix-red-subtle)', color: '#fff' } : {}}
        >
          Batch CSV Scoring
        </button>
      </div>

      {/* Single Prediction Form */}
      {activeTab === 'single' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
          <div className="netflix-card">
            <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '16px' }}>Input Features</h3>
            {schemaLoading ? (
              <div style={{ padding: '40px', textAlign: 'center' }}><LoadingSpinner /></div>
            ) : !schema?.features ? (
              <p style={{ color: 'var(--text-secondary)' }}>No feature schema available for this experiment.</p>
            ) : (
              <form onSubmit={handleSinglePredict}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px', maxHeight: '420px', overflowY: 'auto', paddingRight: '8px', marginBottom: '20px' }}>
                  {schema.features.map((feat) => (
                    <div key={feat.name}>
                      <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>
                        {feat.name}
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginLeft: '4px' }}>({feat.type})</span>
                      </label>
                      <input
                        type={feat.type === 'numeric' ? 'number' : 'text'}
                        step="any"
                        className="netflix-input"
                        value={featureInputs[feat.name] ?? ''}
                        onChange={(e) => setFeatureInputs({ ...featureInputs, [feat.name]: e.target.value })}
                        style={{ padding: '8px 12px', fontSize: '13px' }}
                      />
                    </div>
                  ))}
                </div>

                <button
                  type="submit"
                  disabled={singleLoading || !selectedExpId}
                  className="btn-netflix-primary"
                  style={{ width: '100%' }}
                >
                  {singleLoading ? <LoadingSpinner size="sm" /> : <><FiPlay /> Run Prediction</>}
                </button>
              </form>
            )}

            {singleError && (
              <div style={{ marginTop: '16px', padding: '12px', background: 'var(--danger-bg)', border: '1px solid var(--danger)', borderRadius: 'var(--radius-sm)', color: '#ff6b72', fontSize: '13px' }}>
                <FiAlertCircle style={{ marginRight: '6px' }} />
                {singleError}
              </div>
            )}
          </div>

          {/* Prediction Result Display */}
          <div className="netflix-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '16px' }}>Prediction Outcome</h3>
            {singleLoading ? (
              <div style={{ padding: '60px', textAlign: 'center' }}><LoadingSpinner /></div>
            ) : singleResult ? (
              <div>
                <div style={{ background: '#111111', border: '1px solid rgba(229, 9, 20, 0.4)', borderRadius: 'var(--radius-md)', padding: '24px', textAlign: 'center', marginBottom: '20px' }}>
                  <div style={{ fontSize: '13px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px' }}>
                    Predicted Value
                  </div>
                  <div style={{ fontSize: '36px', fontWeight: '900', color: 'var(--netflix-red)', margin: '10px 0' }}>
                    {String(singleResult.prediction)}
                  </div>
                  {singleResult.probability !== undefined && (
                    <div style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                      Confidence: <strong>{(singleResult.probability * 100).toFixed(1)}%</strong>
                    </div>
                  )}
                </div>

                {singleResult.class_probabilities && (
                  <div>
                    <h4 style={{ fontSize: '13px', fontWeight: '700', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                      Class Probabilities
                    </h4>
                    {Object.entries(singleResult.class_probabilities).map(([cls, prob]) => (
                      <div key={cls} style={{ marginBottom: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                          <span>{cls}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{ width: '100%', height: '6px', background: '#222', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${prob * 100}%`, height: '100%', background: 'var(--netflix-red)' }} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '40px' }}>
                <FiTarget style={{ fontSize: '48px', color: 'var(--border-highlight)', marginBottom: '12px' }} />
                <p>Fill in feature values and click <strong>Run Prediction</strong> to see the result.</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Batch Prediction Form */}
      {activeTab === 'batch' && (
        <div className="netflix-card">
          <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FiUploadCloud style={{ color: 'var(--netflix-red)' }} />
            <span>Batch CSV Prediction</span>
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '20px' }}>
            Upload a CSV containing multiple feature records. The model will run inference on all rows and
            append a prediction column.
          </p>

          <div {...getRootProps()} className={`netflix-dropzone ${isDragActive ? 'active' : ''}`} style={{ marginBottom: '20px' }}>
            <input {...getInputProps()} />
            <FiUploadCloud style={{ fontSize: '42px', color: isDragActive ? 'var(--netflix-red)' : 'var(--text-secondary)', marginBottom: '12px' }} />
            {batchLoading ? (
              <div>
                <LoadingSpinner />
                <p style={{ marginTop: '12px', color: 'var(--text-secondary)' }}>Scoring batch dataset...</p>
              </div>
            ) : (
              <div>
                <p style={{ fontSize: '16px', fontWeight: '600', marginBottom: '6px' }}>Drop your CSV file here for batch inference</p>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Must contain matching feature column headers</p>
              </div>
            )}
          </div>

          {batchError && (
            <div style={{ padding: '12px', background: 'var(--danger-bg)', border: '1px solid var(--danger)', borderRadius: 'var(--radius-sm)', color: '#ff6b72', fontSize: '13px', marginBottom: '16px' }}>
              <FiAlertCircle style={{ marginRight: '6px' }} />
              {batchError}
            </div>
          )}

          {batchResult && (
            <div style={{ padding: '20px', background: 'var(--success-bg)', border: '1px solid var(--success)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--success)', fontWeight: '800', fontSize: '16px', marginBottom: '6px' }}>
                <FiCheckCircle /> Batch Prediction Completed!
              </div>
              <p style={{ fontSize: '14px', color: '#e0e0e0', marginBottom: '14px' }}>
                Successfully predicted {batchResult.rows_scored || batchResult.predictions?.length || 'all'} records.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
