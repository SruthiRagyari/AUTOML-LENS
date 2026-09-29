import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api } from '../services/api'
import { FiMessageCircle, FiX, FiSend, FiDownload, FiRefreshCw, FiZap } from 'react-icons/fi'
import ModelComparison from '../components/Charts/ModelComparison'
import FeatureImportance from '../components/Charts/FeatureImportance'
import ConfusionMatrix from '../components/Charts/ConfusionMatrix'
import StatusBadge from '../components/Common/StatusBadge'
import LoadingSpinner from '../components/Common/LoadingSpinner'

const STEPS = [
  { key: 'dataset', label: 'Dataset' },
  { key: 'profiling', label: 'Profiling' },
  { key: 'ai_analysis', label: 'AI Analysis' },
  { key: 'preprocessing', label: 'Preprocess' },
  { key: 'model_selection', label: 'Models' },
  { key: 'optimization', label: 'Optimize' },
  { key: 'evaluation', label: 'Evaluate' },
  { key: 'explainability', label: 'Explain' },
  { key: 'prediction', label: 'Predict' },
  { key: 'report', label: 'Report' },
]

export default function Experiment() {
  const { id } = useParams()
  const [exp, setExp] = useState(null)
  const [profile, setProfile] = useState(null)
  const [llmAnalysis, setLlmAnalysis] = useState(null)
  const [results, setResults] = useState(null)
  const [explainability, setExplainability] = useState(null)
  const [inputSchema, setInputSchema] = useState([])
  const [predInputs, setPredInputs] = useState({})
  const [predResult, setPredResult] = useState(null)
  const [batchFile, setBatchFile] = useState(null)
  const [batchResult, setBatchResult] = useState(null)
  const [reportUrl, setReportUrl] = useState(null)
  const [activeStep, setActiveStep] = useState(0)
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState(false)
  const [actionMsg, setActionMsg] = useState('')
  const [error, setError] = useState('')
  const [chatOpen, setChatOpen] = useState(false)
  const [chatMsgs, setChatMsgs] = useState([{ role: 'assistant', text: 'Hi! Ask me anything about this experiment.' }])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const chatBottom = useRef(null)

  const load = useCallback(async () => {
    try {
      const r = await api.getExperiment(id)
      setExp(r.data)
      if (r.data.status === 'completed' || r.data.status === 'analyzed') {
        try { const pr = await api.profileDataset(r.data.dataset_id); setProfile(pr.data) } catch {}
        if (r.data.llm_analysis_json || r.data.status !== 'created') {
          try { const la = await api.analyzeExperiment(id); setLlmAnalysis(la.data?.result || la.data) } catch {}
        }
        if (r.data.status === 'completed') {
          try { const rs = await api.getResults(id); setResults(rs.data) } catch {}
          try { const ex = await api.getExplainability(id); setExplainability(ex.data) } catch {}
          try { const sc = await api.getInputSchema(id); setInputSchema(sc.data?.fields || []) } catch {}
        }
      }
    } catch { setError('Failed to load experiment') }
    finally { setLoading(false) }
  }, [id])

  useEffect(() => { load() }, [load])
  useEffect(() => { chatBottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [chatMsgs])

  const runProfile = async () => {
    setActionLoading(true); setActionMsg('Profiling dataset...')
    try { const r = await api.profileDataset(exp.dataset_id); setProfile(r.data); setActiveStep(2) }
    catch { setError('Profiling failed') }
    finally { setActionLoading(false); setActionMsg('') }
  }

  const runAnalysis = async () => {
    setActionLoading(true); setActionMsg('Running AI analysis...')
    try { const r = await api.analyzeExperiment(id); setLlmAnalysis(r.data?.result || r.data); setActiveStep(3); await load() }
    catch { setError('AI analysis failed') }
    finally { setActionLoading(false); setActionMsg('') }
  }

  const runTraining = async (fastDemo) => {
    setActionLoading(true); setActionMsg(fastDemo ? 'Fast demo: 3 models, 5 trials...' : 'Training all models with Optuna optimization...')
    try {
      const r = await api.trainExperiment(id, fastDemo)
      setResults(r.data)
      await load()
      setActiveStep(6)
    }
    catch (e) { setError(`Training failed: ${e.response?.data?.detail || e.message}`) }
    finally { setActionLoading(false); setActionMsg('') }
  }

  const runPredict = async () => {
    setActionLoading(true)
    try {
      const r = await api.predict(id, predInputs)
      setPredResult(r.data)
    } catch (e) { setError(`Prediction failed: ${e.response?.data?.detail || e.message}`) }
    finally { setActionLoading(false) }
  }

  const runBatchPredict = async () => {
    if (!batchFile) return
    setActionLoading(true)
    try { const r = await api.batchPredict(id, batchFile); setBatchResult(r.data) }
    catch (e) { setError(`Batch prediction failed: ${e.response?.data?.detail || e.message}`) }
    finally { setActionLoading(false) }
  }

  const runReport = async () => {
    setActionLoading(true)
    try { const r = await api.getReport(id); setReportUrl(r.data.report_url) }
    catch { setError('Report generation failed') }
    finally { setActionLoading(false) }
  }

  const sendChat = async (msg) => {
    if (!msg.trim()) return
    setChatMsgs(prev => [...prev, { role: 'user', text: msg }])
    setChatInput('')
    setChatLoading(true)
    try {
      const r = await api.chat(msg, parseInt(id))
      setChatMsgs(prev => [...prev, { role: 'assistant', text: r.data.response }])
    } catch { setChatMsgs(prev => [...prev, { role: 'assistant', text: 'Sorry, something went wrong.' }]) }
    finally { setChatLoading(false) }
  }

  const statusToStep = (status) => {
    const map = { created: 0, profiling: 1, analyzed: 2, preprocessing: 3, training: 4, completed: 6, failed: 0 }
    return map[status] ?? 0
  }

  if (loading) return <LoadingSpinner message="Loading experiment..." />
  if (error && !exp) return <div className="alert alert-danger">{error}</div>

  const models = results?.models || []
  const bestModel = results?.best_model || {}
  const comparisonData = models.filter(m => m.status === 'COMPLETED').map(m => {
    const met = m.optimized_metrics || m.baseline_metrics || {}
    const score = Object.values(met).find(v => typeof v === 'number')
    return { display_name: m.display_name, score, model_name: m.model_name }
  })

  const expl = explainability || {}
  const cm = models.find(m => m.is_best)?.optimized_metrics?.confusion_matrix || []
  const cmLabels = models.find(m => m.is_best)?.optimized_metrics?.class_distribution
    ? Object.keys(models.find(m => m.is_best)?.optimized_metrics?.class_distribution || {}) : []

  return (
    <div className="page">
      {error && <div className="alert alert-danger" style={{ marginBottom: 16 }}>{error} <button className="btn btn-sm" onClick={() => setError('')}>✕</button></div>}

      <div className="page-header">
        <div>
          <h1>📊 {exp?.name || 'Experiment'}</h1>
          <p style={{ marginTop: 4 }}>Dataset #{exp?.dataset_id} · Target: <strong>{exp?.target_column}</strong> · <StatusBadge status={exp?.status} /></p>
        </div>
        <button className="btn btn-sm" onClick={load}><FiRefreshCw /> Refresh</button>
      </div>

      {/* Stepper */}
      <div className="card" style={{ marginBottom: 20, padding: '14px 20px', overflowX: 'auto' }}>
        <div className="stepper" style={{ padding: 0 }}>
          {STEPS.map((step, idx) => {
            const completed = idx < statusToStep(exp?.status) || (exp?.status === 'completed' && idx < 9)
            const active = activeStep === idx
            return (
              <div key={step.key} className="step">
                <button className={`step-button${active ? ' active' : ''}${completed ? ' completed' : ''}`} onClick={() => setActiveStep(idx)}>
                  <div className="step-circle">{completed ? '✓' : idx + 1}</div>
                  <div className="step-label">{step.label}</div>
                </button>
                {idx < STEPS.length - 1 && <div className={`step-connector${completed ? ' done' : ''}`} />}
              </div>
            )
          })}
        </div>
      </div>

      {actionLoading && (
        <div className="alert alert-info" style={{ marginBottom: 16 }}>
          <div className="spinner" style={{ display: 'inline-block', width: 16, height: 16, borderWidth: 2, marginRight: 10, verticalAlign: 'middle' }} />
          {actionMsg}
        </div>
      )}

      {/* Step Content */}
      <div className="card" style={{ minHeight: 380 }}>
        {/* Step 0: Dataset */}
        {activeStep === 0 && (
          <div>
            <h2>Dataset Information</h2>
            <div className="metrics-grid">
              <div className="metric-card"><div className="metric-value">{exp?.dataset_id}</div><div className="metric-label">Dataset ID</div></div>
              <div className="metric-card"><div className="metric-value" style={{ fontSize: 20 }}>{exp?.target_column}</div><div className="metric-label">Target Column</div></div>
              <div className="metric-card"><div className="metric-value" style={{ fontSize: 20, textTransform: 'capitalize' }}>{exp?.problem_type || 'Auto'}</div><div className="metric-label">Problem Type</div></div>
              <div className="metric-card"><div className="metric-value" style={{ fontSize: 20, textTransform: 'capitalize' }}>{exp?.mode}</div><div className="metric-label">Mode</div></div>
              <div className="metric-card"><div className="metric-value">{exp?.n_folds}</div><div className="metric-label">CV Folds</div></div>
              <div className="metric-card"><div className="metric-value">{exp?.n_trials}</div><div className="metric-label">Optuna Trials</div></div>
            </div>
            <button className="btn btn-primary" onClick={runProfile} disabled={actionLoading} style={{ marginTop: 16 }}>
              ▶ Profile Dataset
            </button>
          </div>
        )}

        {/* Step 1: Profiling */}
        {activeStep === 1 && (
          <div>
            <h2>Dataset Profile</h2>
            {!profile ? (
              <div>
                <p style={{ marginBottom: 16, color: 'var(--text-secondary)' }}>Run dataset profiling to analyze structure, missing values, and data quality.</p>
                <button className="btn btn-primary" onClick={runProfile} disabled={actionLoading}>▶ Run Profiling</button>
              </div>
            ) : (
              <div>
                <div className="metrics-grid" style={{ marginBottom: 20 }}>
                  <div className="metric-card"><div className="metric-value">{profile.rows?.toLocaleString()}</div><div className="metric-label">Rows</div></div>
                  <div className="metric-card"><div className="metric-value">{profile.columns}</div><div className="metric-label">Columns</div></div>
                  <div className="metric-card"><div className="metric-value">{profile.memory_usage_mb} MB</div><div className="metric-label">Memory</div></div>
                  <div className="metric-card"><div className="metric-value">{profile.total_missing_percentage}%</div><div className="metric-label">Missing</div></div>
                  <div className="metric-card"><div className="metric-value">{profile.duplicate_rows}</div><div className="metric-label">Duplicates</div></div>
                </div>
                {profile.warnings?.length > 0 && (
                  <div className="alert alert-warning" style={{ marginBottom: 16 }}>
                    <strong>⚠️ Warnings:</strong>
                    <ul style={{ marginTop: 8, paddingLeft: 20 }}>{profile.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
                  </div>
                )}
                <div className="section-title">Column Summary</div>
                <div className="table-container">
                  <table>
                    <thead><tr><th>Column</th><th>Type</th><th>Unique</th><th>Missing %</th><th>Sample Values</th></tr></thead>
                    <tbody>
                      {(profile.column_profiles || []).slice(0, 20).map(cp => (
                        <tr key={cp.name}>
                          <td><strong>{cp.name}</strong>{cp.name === exp?.target_column ? ' 🎯' : ''}</td>
                          <td><span className="tag">{cp.inferred_type}</span></td>
                          <td>{cp.unique_count}</td>
                          <td style={{ color: cp.missing_percentage > 10 ? 'var(--danger)' : 'inherit' }}>{cp.missing_percentage}%</td>
                          <td style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{cp.sample_values?.slice(0, 3).join(', ')}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <button className="btn btn-primary" onClick={() => setActiveStep(2)} style={{ marginTop: 16 }}>Next: AI Analysis →</button>
              </div>
            )}
          </div>
        )}

        {/* Step 2: AI Analysis */}
        {activeStep === 2 && (
          <div>
            <h2>🤖 AI Analysis</h2>
            {!llmAnalysis ? (
              <div>
                <p style={{ marginBottom: 16, color: 'var(--text-secondary)' }}>The LLM will analyze your dataset structure and recommend the optimal ML strategy.</p>
                <button className="btn btn-primary" onClick={runAnalysis} disabled={actionLoading}>▶ Run AI Analysis</button>
              </div>
            ) : (
              <div>
                <div className="alert alert-info" style={{ marginBottom: 20 }}>
                  <strong>Provider:</strong> {llmAnalysis.provider_used || 'Fallback'} · <strong>Confidence:</strong> {((llmAnalysis.confidence || 0.8) * 100).toFixed(0)}%
                </div>
                <div className="metrics-grid" style={{ marginBottom: 20 }}>
                  <div className="metric-card"><div className="metric-value" style={{ fontSize: 18, textTransform: 'capitalize' }}>{llmAnalysis.problem_type}</div><div className="metric-label">Problem Type</div></div>
                  <div className="metric-card"><div className="metric-value" style={{ fontSize: 16 }}>{llmAnalysis.recommended_metric}</div><div className="metric-label">Recommended Metric</div></div>
                </div>
                <div className="section-title">Reasoning</div>
                <p style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.7, marginBottom: 16 }}>{llmAnalysis.reasoning}</p>
                {llmAnalysis.candidate_models?.length > 0 && (
                  <><div className="section-title">Recommended Models</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
                    {llmAnalysis.candidate_models.map(m => <span key={m} className="tag">{m}</span>)}
                  </div></>
                )}
                {llmAnalysis.preprocessing?.length > 0 && (
                  <><div className="section-title">Preprocessing Advice</div>
                  <ul style={{ paddingLeft: 20, fontSize: 14, color: 'var(--text-secondary)' }}>
                    {llmAnalysis.preprocessing.map((p, i) => <li key={i}><strong>{p.column}</strong>: {p.action} — {p.reason}</li>)}
                  </ul></>
                )}
                {llmAnalysis.warnings?.length > 0 && (
                  <div className="alert alert-warning" style={{ marginTop: 16 }}>
                    <strong>Warnings:</strong>
                    <ul style={{ marginTop: 8, paddingLeft: 20 }}>{llmAnalysis.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
                  </div>
                )}
                <div style={{ marginTop: 20, display: 'flex', gap: 12 }}>
                  <button className="btn btn-primary" onClick={() => runTraining(false)} disabled={actionLoading}>
                    ▶ Start Full Training ({exp?.n_folds} folds, {exp?.n_trials} trials)
                  </button>
                  <button className="btn btn-outline" onClick={() => runTraining(true)} disabled={actionLoading}>
                    <FiZap /> Fast Demo (3 models, 5 trials)
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Steps 3-5: Training config */}
        {[3, 4, 5].includes(activeStep) && (
          <div>
            <h2>Step {activeStep + 1}: {STEPS[activeStep].label}</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 20 }}>
              {activeStep === 3 && 'Preprocessing: median imputation for numerical, mode imputation + one-hot encoding for categorical, TF-IDF for text.'}
              {activeStep === 4 && `Model selection based on LLM recommendations. ${llmAnalysis?.candidate_models?.length || 0} models queued.`}
              {activeStep === 5 && `Optuna hyperparameter optimization with ${exp?.n_trials || 20} trials per model using cross-validation.`}
            </p>
            {exp?.status === 'completed' ? (
              <div className="alert alert-success">Training completed successfully! View results in the Evaluation tab.</div>
            ) : (
              <div style={{ display: 'flex', gap: 12 }}>
                <button className="btn btn-primary" onClick={() => runTraining(false)} disabled={actionLoading}>
                  ▶ Start Training
                </button>
                <button className="btn btn-outline" onClick={() => runTraining(true)} disabled={actionLoading}>
                  <FiZap /> Fast Demo
                </button>
              </div>
            )}
          </div>
        )}

        {/* Step 6: Evaluation */}
        {activeStep === 6 && (
          <div>
            <h2>📈 Model Evaluation</h2>
            {!results ? (
              <div>
                <p style={{ marginBottom: 16, color: 'var(--text-secondary)' }}>Run training to see evaluation results.</p>
                <button className="btn btn-primary" onClick={() => runTraining(true)} disabled={actionLoading}>
                  <FiZap /> Start Training (Fast Demo)
                </button>
              </div>
            ) : (
              <div>
                {results.best_model && (
                  <div className="alert alert-success" style={{ marginBottom: 20 }}>
                    🏆 <strong>Best Model:</strong> {results.best_model_name} · Score: <strong>{results.best_score?.toFixed(4)}</strong> · Metric: {results.primary_metric}
                  </div>
                )}
                {results.llm_explanation && (
                  <div style={{ background: '#f8f9fc', border: '1px solid var(--border)', borderRadius: 8, padding: 16, marginBottom: 20, fontSize: 14, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                    {results.llm_explanation}
                  </div>
                )}
                <div className="chart-container">
                  <ModelComparison models={comparisonData} metric={results.primary_metric} />
                </div>
                <div className="section-title" style={{ marginTop: 20 }}>Model Results Table</div>
                <div className="table-container">
                  <table>
                    <thead><tr><th>Model</th><th>Status</th><th>CV Score</th><th>Test Score</th><th>Train Time</th><th>Best Params</th></tr></thead>
                    <tbody>
                      {models.map(m => {
                        const met = m.optimized_metrics || m.baseline_metrics || {}
                        const score = Object.values(met).find(v => typeof v === 'number' && !Array.isArray(v))
                        const cvMean = m.cv_scores?.length ? (m.cv_scores.reduce((a, b) => a + b, 0) / m.cv_scores.length).toFixed(4) : 'N/A'
                        return (
                          <tr key={m.model_name} style={m.is_best ? { background: '#eef1ff' } : {}}>
                            <td><strong>{m.display_name}</strong>{m.is_best ? ' 🏆' : ''}</td>
                            <td><StatusBadge status={m.status} /></td>
                            <td>{cvMean}</td>
                            <td>{score !== undefined ? score?.toFixed(4) : 'N/A'}</td>
                            <td>{m.training_time?.toFixed(2)}s</td>
                            <td style={{ fontSize: 12, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {Object.entries(m.best_params || {}).map(([k, v]) => `${k}=${v}`).join(', ') || '—'}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                {cm.length > 0 && (
                  <div style={{ marginTop: 20 }}>
                    <div className="section-title">Confusion Matrix (Best Model)</div>
                    <div className="chart-container">
                      <ConfusionMatrix matrix={cm} labels={cmLabels} />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Step 7: Explainability */}
        {activeStep === 7 && (
          <div>
            <h2>🔍 Explainability</h2>
            {!expl.feature_importance ? (
              <p style={{ color: 'var(--text-secondary)' }}>Complete training to see feature importance.</p>
            ) : (
              <div>
                <div className="alert alert-info" style={{ marginBottom: 16 }}>
                  Method: <strong>{expl.method_used || 'N/A'}</strong>
                </div>
                <p style={{ fontSize: 14, color: 'var(--text-secondary)', marginBottom: 20, lineHeight: 1.7 }}>{expl.explanation_text}</p>
                <div className="chart-container">
                  <FeatureImportance features={expl.feature_importance || []} method={expl.method_used} />
                </div>
                {expl.top_features?.length > 0 && (
                  <div style={{ marginTop: 16 }}>
                    <div className="section-title">Top Features</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                      {expl.top_features.map((f, i) => (
                        <span key={f} className="tag">#{i + 1} {f}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Step 8: Prediction */}
        {activeStep === 8 && (
          <div>
            <h2>🎯 Prediction</h2>
            {exp?.status !== 'completed' ? (
              <p style={{ color: 'var(--text-secondary)' }}>Complete training first to make predictions.</p>
            ) : (
              <div>
                <div className="section-title">Single Prediction</div>
                {inputSchema.length === 0 ? (
                  <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>Input schema not available. Training may have just completed — refresh the page.</p>
                ) : (
                  <div>
                    <div className="predict-grid">
                      {inputSchema.map(field => (
                        <div key={field.name} className="form-group" style={{ marginBottom: 0 }}>
                          <label className="form-label">{field.name}</label>
                          <input
                            className="form-input"
                            type={field.type === 'number' ? 'number' : 'text'}
                            placeholder={field.sample_values?.[0] ?? ''}
                            value={predInputs[field.name] ?? ''}
                            onChange={e => setPredInputs({ ...predInputs, [field.name]: e.target.value })}
                          />
                        </div>
                      ))}
                    </div>
                    <button className="btn btn-primary" onClick={runPredict} disabled={actionLoading} style={{ marginTop: 16 }}>
                      ▶ Predict
                    </button>
                    {predResult && (
                      <div className="predict-result">
                        <div className="predict-value">{String(predResult.prediction)}</div>
                        <div className="predict-label">Prediction</div>
                        {predResult.confidence !== undefined && (
                          <p style={{ marginTop: 8, color: 'var(--text-secondary)', fontSize: 14 }}>
                            Confidence: <strong>{(predResult.confidence * 100).toFixed(1)}%</strong>
                          </p>
                        )}
                        {predResult.probabilities && (
                          <div style={{ marginTop: 16, textAlign: 'left' }}>
                            {Object.entries(predResult.probabilities).map(([cls, prob]) => (
                              <div key={cls} className="prob-bar">
                                <div className="prob-label"><span>{cls}</span><span>{(prob * 100).toFixed(1)}%</span></div>
                                <div className="progress-bar"><div className="progress-fill" style={{ width: `${prob * 100}%` }} /></div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                <div className="divider" />
                <div className="section-title">Batch Prediction</div>
                <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>Upload a CSV file (without the target column) to get batch predictions.</p>
                <input type="file" accept=".csv" onChange={e => setBatchFile(e.target.files[0])} style={{ marginBottom: 12 }} />
                {batchFile && <p style={{ fontSize: 13, color: 'var(--success)', marginBottom: 12 }}>📁 {batchFile.name}</p>}
                <button className="btn btn-primary" onClick={runBatchPredict} disabled={!batchFile || actionLoading}>
                  ▶ Run Batch Prediction
                </button>
                {batchResult && (
                  <div className="alert alert-success" style={{ marginTop: 16 }}>
                    ✅ {batchResult.num_predictions} predictions generated.
                    {batchResult.predictions_path && (
                      <a href={`/predictions/${batchResult.predictions_path.split('/').pop()}`} download className="btn btn-sm btn-primary" style={{ marginLeft: 12 }}>
                        <FiDownload /> Download
                      </a>
                    )}
                    {batchResult.preview?.length > 0 && (
                      <div className="table-container" style={{ marginTop: 12 }}>
                        <table>
                          <thead><tr>{Object.keys(batchResult.preview[0]).map(k => <th key={k}>{k}</th>)}</tr></thead>
                          <tbody>{batchResult.preview.slice(0, 5).map((row, i) => (
                            <tr key={i}>{Object.values(row).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>
                          ))}</tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Step 9: Report */}
        {activeStep === 9 && (
          <div>
            <h2>📄 Report & Download</h2>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
              <button className="btn btn-primary" onClick={runReport} disabled={actionLoading}>
                📄 Generate HTML Report
              </button>
              <a href={api.downloadModelUrl(id)} className="btn btn-outline" download>
                <FiDownload /> Download Model (.joblib)
              </a>
              <a href={api.downloadMetaUrl(id)} className="btn btn-outline" download>
                <FiDownload /> Download Metadata (.json)
              </a>
            </div>
            {reportUrl && (
              <div className="alert alert-success">
                ✅ Report generated! <a href={reportUrl} target="_blank" rel="noreferrer" className="btn btn-sm btn-primary" style={{ marginLeft: 12 }}>Open Report</a>
              </div>
            )}
            {exp?.status === 'completed' && (
              <div className="card card-sm bg-light">
                <div className="section-title">Experiment Summary</div>
                <table style={{ fontSize: 13 }}>
                  <tbody>
                    <tr><td style={{ padding: '6px 16px 6px 0', color: 'var(--text-secondary)', fontWeight: 500 }}>Best Model</td><td>{results?.best_model_name || '—'}</td></tr>
                    <tr><td style={{ padding: '6px 16px 6px 0', color: 'var(--text-secondary)', fontWeight: 500 }}>Score</td><td>{results?.best_score?.toFixed(4) || '—'}</td></tr>
                    <tr><td style={{ padding: '6px 16px 6px 0', color: 'var(--text-secondary)', fontWeight: 500 }}>Metric</td><td>{results?.primary_metric || '—'}</td></tr>
                    <tr><td style={{ padding: '6px 16px 6px 0', color: 'var(--text-secondary)', fontWeight: 500 }}>Problem Type</td><td>{exp?.problem_type}</td></tr>
                    <tr><td style={{ padding: '6px 16px 6px 0', color: 'var(--text-secondary)', fontWeight: 500 }}>Mode</td><td>{exp?.mode}</td></tr>
                    <tr><td style={{ padding: '6px 16px 6px 0', color: 'var(--text-secondary)', fontWeight: 500 }}>Models Trained</td><td>{models.length}</td></tr>
                    <tr><td style={{ padding: '6px 16px 6px 0', color: 'var(--text-secondary)', fontWeight: 500 }}>Top Features</td><td>{expl.top_features?.slice(0, 5).join(', ') || '—'}</td></tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Chat FAB */}
      <button className="chat-fab" onClick={() => setChatOpen(o => !o)}>
        <FiMessageCircle />
      </button>

      {chatOpen && (
        <div className="chat-panel">
          <div className="chat-header">
            <h3>🤖 AI Assistant</h3>
            <button className="chat-close" onClick={() => setChatOpen(false)}><FiX /></button>
          </div>
          <div className="chat-messages">
            {chatMsgs.map((m, i) => (
              <div key={i} className={`chat-msg ${m.role}`}>{m.text}</div>
            ))}
            {chatLoading && <div className="chat-msg assistant">Thinking...</div>}
            <div ref={chatBottom} />
          </div>
          <div className="chat-suggestions">
            {['What is my target?', 'Best model?', 'Top features?'].map(s => (
              <button key={s} className="suggestion-btn" onClick={() => sendChat(s)}>{s}</button>
            ))}
          </div>
          <div className="chat-input-row">
            <input className="chat-input" value={chatInput} onChange={e => setChatInput(e.target.value)}
              placeholder="Ask about this experiment..." onKeyDown={e => e.key === 'Enter' && sendChat(chatInput)} />
            <button className="btn btn-primary chat-send" onClick={() => sendChat(chatInput)} disabled={chatLoading || !chatInput.trim()}>
              <FiSend />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
