import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { api } from '../services/api'
import { FiMessageCircle, FiX, FiSend, FiDownload, FiRefreshCw, FiZap } from 'react-icons/fi'
import ModelComparison from '../components/Charts/ModelComparison'
import FeatureImportance from '../components/Charts/FeatureImportance'
import ConfusionMatrix from '../components/Charts/ConfusionMatrix'
import StatusBadge from '../components/Common/StatusBadge'
import LoadingSpinner from '../components/Common/LoadingSpinner'
import TrainingProgressPanel from '../components/Common/TrainingProgressPanel'
import CodeBlock from '../components/Common/CodeBlock'

/** Human label for a metric key the backend actually reported. */
const METRIC_LABELS = {
  accuracy: 'Accuracy',
  precision_weighted: 'Precision (weighted)',
  recall_weighted: 'Recall (weighted)',
  f1_weighted: 'F1 (weighted)',
  balanced_accuracy: 'Balanced accuracy',
  roc_auc: 'ROC-AUC',
  pr_auc: 'PR-AUC',
  r2: 'R²',
  mse: 'MSE',
  rmse: 'RMSE',
  mae: 'MAE',
  mape: 'MAPE (%)',
}

/** Metrics worth showing, in a fixed readable order, per task type. */
const DISPLAY_METRICS = {
  classification: ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'balanced_accuracy', 'roc_auc', 'pr_auc'],
  regression: ['r2', 'rmse', 'mae', 'mse', 'mape'],
}

/**
 * The metrics a model actually computed, in DISPLAY_METRICS order.
 * A metric that was not calculated is simply absent - never defaulted, never
 * substituted with a different metric.
 */
function realMetrics(metrics, problemType) {
  if (!metrics) return []
  const keys = DISPLAY_METRICS[problemType] || DISPLAY_METRICS.classification
  return keys
    .filter(k => typeof metrics[k] === 'number' && Number.isFinite(metrics[k]))
    .map(k => ({ key: k, label: METRIC_LABELS[k] || k, value: metrics[k] }))
}

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

/**
 * Renders the Model Fusion & Ensemble Optimization section.
 */
function ModelFusionSection({ ensembleCandidates, models, bestModelName, primaryMetric }) {
  const ensembleModels = (ensembleCandidates && ensembleCandidates.length > 0)
    ? ensembleCandidates
    : (models || []).filter(
        m => m.optimization?.is_ensemble || m.model_name?.startsWith('ensemble')
      )
  if (ensembleModels.length === 0) return null

  const isEnsembleWinner = bestModelName && ensembleModels.some(m => m.display_name === bestModelName || m.model_name === bestModelName)

  return (
    <div style={{ marginTop: 24, padding: 18, background: '#181818', border: '1px solid var(--border)', borderRadius: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
        <div style={{ fontWeight: 600, fontSize: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>🧬 Model Fusion &amp; Ensemble Optimization</span>
          {isEnsembleWinner ? (
            <span className="tag" style={{ background: 'rgba(46,125,50,0.2)', color: '#4caf50', borderColor: '#4caf50' }}>
              Ensemble Winner Selected 🏆
            </span>
          ) : (
            <span className="tag" style={{ background: 'rgba(67,97,238,0.2)', color: '#7090ff', borderColor: '#7090ff' }}>
              Individual Model Winner
            </span>
          )}
        </div>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          🛡️ Weights optimized on OOF training predictions · Holdout untouched
        </span>
      </div>

      <div className="table-container" style={{ marginBottom: 12 }}>
        <table>
          <thead>
            <tr>
              <th>Ensemble Candidate</th>
              <th>Strategy</th>
              <th>OOF CV Score</th>
              <th>Holdout Score</th>
              <th>Member Weights</th>
            </tr>
          </thead>
          <tbody>
            {ensembleModels.map(em => {
              const weights = em.optimization?.weights || {}
              const isBest = em.is_best || em.display_name === bestModelName
              return (
                <tr key={em.model_name} style={isBest ? { background: 'rgba(229,9,20,0.1)' } : {}}>
                  <td>
                    <strong>{em.display_name}</strong>
                    {isBest && ' 🏆'}
                  </td>
                  <td><code>{em.optimization?.strategy || 'weighted'}</code></td>
                  <td>{em.selection_score != null ? em.selection_score.toFixed(4) : em.optimization?.best_cv_score != null ? em.optimization.best_cv_score.toFixed(4) : '—'}</td>
                  <td>{em.holdout_primary != null ? em.holdout_primary.toFixed(4) : '—'}</td>
                  <td>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {typeof weights === 'object' && Object.keys(weights).length > 0 ? (
                        Object.entries(weights).map(([k, v]) => (
                          <span key={k} className="tag" style={{ fontSize: 11, padding: '2px 6px' }}>
                            {k}: {typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : v}
                          </span>
                        ))
                      ) : (
                        <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Equal / uniform</span>
                      )}
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0 }}>
        Ensemble candidates combine predictions from real trained models. Weights are learned using strictly out-of-fold cross-validation
        on the training split. Holdout metrics were computed once after selection.
      </p>
    </div>
  )
}

/**
 * Renders the Research-Grade Benchmarking & Evaluation Section.
 */
function ResearchEvaluationSection({ results, exp }) {
  const evalData = results?.research_evaluation
  if (!evalData) return null

  const isEnsemble = evalData.winner_type === 'ensemble'
  const isLlm = evalData.condition === 'llm_assisted'
  const repro = evalData.reproducibility || {}
  const advisory = evalData.llm_advisory || {}

  return (
    <div style={{ marginTop: 24, padding: 18, background: '#141414', border: '1px solid var(--border)', borderRadius: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, flexWrap: 'wrap', gap: 8 }}>
        <div style={{ fontWeight: 600, fontSize: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>📊 Research Benchmarking &amp; Evaluation</span>
          <span className="tag" style={{ background: isLlm ? 'rgba(156,39,176,0.2)' : 'rgba(33,150,243,0.2)', color: isLlm ? '#ce93d8' : '#64b5f6', borderColor: isLlm ? '#ce93d8' : '#64b5f6' }}>
            Condition: {isLlm ? 'LLM-Assisted' : 'Deterministic Fallback'}
          </span>
          {isEnsemble ? (
            <span className="tag" style={{ background: 'rgba(76,175,80,0.2)', color: '#81c784', borderColor: '#81c784' }}>
              Ensemble Winner
            </span>
          ) : (
            <span className="tag" style={{ background: 'rgba(255,152,0,0.2)', color: '#ffb74d', borderColor: '#ffb74d' }}>
              Single Winner
            </span>
          )}
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Seed <code>{repro.seed ?? '42'}</code> · {repro.n_folds ?? 3} Folds · {repro.n_trials ?? 5} Trials
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginBottom: 16 }}>
        <div style={{ background: '#1c1c1c', padding: 12, borderRadius: 6, border: '1px solid #2a2a2a' }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 4 }}>Dataset &amp; Task</div>
          <div style={{ fontWeight: 600, fontSize: 14, wordBreak: 'break-all' }}>{evalData.dataset_name || exp?.name || 'Dataset'}</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
            {evalData.dataset_rows ? `${evalData.dataset_rows} rows × ` : ''}{evalData.dataset_columns ? `${evalData.dataset_columns} cols` : ''} · <span style={{ textTransform: 'capitalize' }}>{evalData.problem_type}</span>
          </div>
        </div>

        <div style={{ background: '#1c1c1c', padding: 12, borderRadius: 6, border: '1px solid #2a2a2a' }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 4 }}>Empirical Winner</div>
          <div style={{ fontWeight: 600, fontSize: 14, color: '#fff' }}>
            {evalData.winner_name || results?.best_model_name}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
            Ranked by {evalData.primary_metric} CV evidence
          </div>
        </div>

        <div style={{ background: '#1c1c1c', padding: 12, borderRadius: 6, border: '1px solid #2a2a2a' }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 4 }}>Cross-Validation Score</div>
          <div style={{ fontWeight: 700, fontSize: 16, color: '#4caf50' }}>
            {evalData.winner_cv_score != null ? evalData.winner_cv_score.toFixed(4) : '—'}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            Evidence used for selection (training split)
          </div>
        </div>

        <div style={{ background: '#1c1c1c', padding: 12, borderRadius: 6, border: '1px solid #2a2a2a' }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 4 }}>Holdout Score</div>
          <div style={{ fontWeight: 700, fontSize: 16, color: '#7090ff' }}>
            {evalData.winner_holdout_score != null ? evalData.winner_holdout_score.toFixed(4) : '—'}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            Evaluated once, strictly post-selection
          </div>
        </div>
      </div>

      <div style={{ background: '#181818', border: '1px solid #262626', borderRadius: 6, padding: 12, fontSize: 13, lineHeight: 1.6 }}>
        <div style={{ fontWeight: 600, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
          <span>🔍 LLM Advisory vs. Empirical ML Reality:</span>
        </div>
        <div style={{ color: 'var(--text-secondary)' }}>
          LLM Provider: <strong>{evalData.llm_provider}</strong>.
          {advisory.recommended_models?.length > 0 && (
            <span> Recommended models: <code>{advisory.recommended_models.join(', ')}</code>.</span>
          )}
          {advisory.applied_operations?.length > 0 ? (
            <span> Proposed {advisory.proposed_operations?.length || 0} feature ops ({advisory.applied_operations.length} applied).</span>
          ) : (
            <span> No feature operations applied.</span>
          )}
          <br />
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Note: LLM recommendations are strictly advisory. The winning model was selected solely on fold-safe cross-validation evidence on the training split with zero holdout leakage.
          </span>
        </div>
      </div>

      <div style={{ marginTop: 12, padding: 12, background: '#181818', border: '1px solid #262626', borderRadius: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
        <div style={{ fontWeight: 600, color: '#e0e0e0', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
          <span>🔬 Experiment Provenance &amp; Scientific Reproducibility:</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 8 }}>
          <div><strong>Seed:</strong> <code>{repro.seed ?? '42'}</code> (propagated to CV, HPO, models)</div>
          <div><strong>Holdout Isolation:</strong> <span style={{ color: '#81c784' }}>{repro.holdout_isolation || '100% Unseen (Post-selection only)'}</span></div>
          <div><strong>Selection Evidence:</strong> <span>{repro.selection_evidence || 'CV on training split only'}</span></div>
          <div><strong>Selection Metric:</strong> <code>{repro.selection_metric || evalData.primary_metric}</code> ({repro.selection_direction || 'maximize'})</div>
          {repro.git_commit && (
            <div><strong>Git Revision:</strong> <code>{repro.git_commit}</code></div>
          )}
          {repro.dataset_hash && (
            <div><strong>Dataset Hash (SHA-256):</strong> <code>{repro.dataset_hash}</code></div>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * Renders the Multi-Seed Benchmark Evaluation Section.
 */
function MultiSeedBenchmarkSection() {
  const [summary, setSummary] = useState(null)
  const [selectedDataset, setSelectedDataset] = useState('adult_census_income')
  const [isExpanded, setIsExpanded] = useState(false)

  useEffect(() => {
    fetch('/api/experiments/benchmarks/summary')
      .then((r) => r.json())
      .then((data) => {
        if (data && data.datasets) {
          setSummary(data)
          const firstKey = Object.keys(data.datasets)[0]
          if (firstKey) setSelectedDataset(firstKey)
        }
      })
      .catch(() => {})
  }, [])

  if (!summary || !summary.datasets || Object.keys(summary.datasets).length === 0) {
    return null
  }

  const dsNames = Object.keys(summary.datasets)
  const curDs = summary.datasets[selectedDataset] || summary.datasets[dsNames[0]]
  const conditions = curDs?.conditions || {}
  const comparisons = curDs?.comparisons || {}
  const behavior = curDs?.pipeline_behavior || {}
  const proto = summary.protocol || {}

  return (
    <div style={{ marginTop: 20, padding: 18, background: '#12161f', border: '1px solid #233554', borderRadius: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 16 }}>📈</span>
          <span style={{ fontWeight: 600, fontSize: 15, color: '#90caf9' }}>
            Multi-Seed Empirical Benchmark Evaluation
          </span>
          <span className="tag" style={{ background: 'rgba(33,150,243,0.15)', color: '#64b5f6', borderColor: '#64b5f6' }}>
            N={proto.seeds?.length || 3} Seeds Tested
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <a
            href="/api/experiments/benchmarks/report"
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary btn-sm"
            style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 4 }}
          >
            <span>Open Research Report (HTML)</span> ↗
          </a>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? 'Collapse' : 'Expand Matrix'}
          </button>
        </div>
      </div>

      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6 }}>
        {proto.holdout_policy} <em>{proto.statistical_statement}</em>
      </div>

      {isExpanded && (
        <div style={{ marginTop: 16 }}>
          {/* Dataset Tabs */}
          <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
            {dsNames.map((name) => (
              <button
                key={name}
                type="button"
                className={`btn btn-sm ${selectedDataset === name ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setSelectedDataset(name)}
                style={{ textTransform: 'none' }}
              >
                {name} ({summary.datasets[name].problem_type})
              </button>
            ))}
          </div>

          {/* Aggregate Benchmark Table */}
          <div className="table-container" style={{ marginBottom: 16 }}>
            <table>
              <thead>
                <tr>
                  <th>Condition</th>
                  <th>CV Score (Mean ± Std)</th>
                  <th>Holdout Score (Mean ± Std)</th>
                  <th>Holdout [Min, Max]</th>
                  <th>Mean Runtime</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(conditions).map(([cName, cData]) => {
                  const isDet = cName === 'deterministic'
                  const isFull = cName === 'llm_guided'
                  return (
                    <tr key={cName}>
                      <td>
                        <strong>{cName}</strong>
                        {isDet && <span style={{ color: 'var(--text-muted)', fontSize: 11 }}> (Rule-based)</span>}
                        {isFull && <span style={{ color: '#ce93d8', fontSize: 11 }}> (Full Pipeline Plan)</span>}
                      </td>
                      <td>{cData.cv_mean != null ? `${cData.cv_mean.toFixed(4)} ± ${cData.cv_std.toFixed(4)}` : '—'}</td>
                      <td>
                        <strong style={{ color: isFull ? '#ba68c8' : isDet ? '#64b5f6' : '#fff' }}>
                          {cData.holdout_mean != null ? `${cData.holdout_mean.toFixed(4)} ± ${cData.holdout_std.toFixed(4)}` : '—'}
                        </strong>
                      </td>
                      <td>{cData.holdout_min != null ? `[${cData.holdout_min.toFixed(4)}, ${cData.holdout_max.toFixed(4)}]` : '—'}</td>
                      <td>{cData.runtime_mean != null ? `${cData.runtime_mean.toFixed(2)}s` : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Comparison Delta Box */}
          {comparisons.llm_guided_vs_deterministic && (
            <div style={{ background: '#1c2433', border: '1px solid #2e4368', borderRadius: 6, padding: 12, marginBottom: 16, fontSize: 13 }}>
              <div style={{ fontWeight: 600, color: '#90caf9', marginBottom: 4 }}>
                Empirical Delta: Full LLM-Guided vs. Deterministic Baseline ({curDs.primary_metric})
              </div>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
                <div>Deterministic Mean: <strong>{comparisons.llm_guided_vs_deterministic.deterministic_holdout_mean?.toFixed(4)}</strong></div>
                <div>LLM-Guided Mean: <strong>{comparisons.llm_guided_vs_deterministic.llm_holdout_mean?.toFixed(4)}</strong></div>
                <div>
                  Direction-Adjusted Δ: <strong style={{ color: comparisons.llm_guided_vs_deterministic.direction_adjusted_delta > 0 ? '#81c784' : '#e57373' }}>
                    {comparisons.llm_guided_vs_deterministic.direction_adjusted_delta > 0 ? '+' : ''}
                    {comparisons.llm_guided_vs_deterministic.direction_adjusted_delta?.toFixed(4)}
                  </strong>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 6 }}>
                    ({comparisons.llm_guided_vs_deterministic.interpretation})
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Pipeline Behavior Summary */}
          <div style={{ background: '#181818', border: '1px solid #262626', borderRadius: 6, padding: 12, fontSize: 12 }}>
            <div style={{ fontWeight: 600, color: '#e0e0e0', marginBottom: 6 }}>
              🔍 LLM Pipeline Behavior &amp; Constraint Enforcement:
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 8, color: 'var(--text-secondary)' }}>
              <div><strong>Models Proposed:</strong> <code>{behavior.models_proposed_unique?.join(', ') || 'None'}</code></div>
              <div><strong>Models Validated:</strong> <code>{behavior.models_accepted_unique?.join(', ') || 'None'}</code></div>
              <div><strong>Feature Operations:</strong> {behavior.operations_proposed_count || 0} proposed ({behavior.operations_accepted_count || 0} accepted)</div>
              <div><strong>Fallback Runs:</strong> {behavior.fallback_runs || 0} / {behavior.total_llm_runs || 0} ({((behavior.fallback_rate || 0) * 100).toFixed(0)}%)</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Renders the LLM-Guided AutoML Pipeline Plan and Execution Provenance.
 */
function PipelinePlanSection({ planProvenance, fallbackPlan, exp }) {
  const prov = planProvenance || {}
  const plan = prov.validated_plan || fallbackPlan || {}
  const executed = prov.executed_pipeline || {}
  const raw = prov.raw_recommendation || {}

  if (!plan && !prov.source) return null

  const isLlm = (prov.source === 'llm' || plan.source === 'llm') && !prov.is_fallback
  const status = prov.validation_status || plan.validation_status || 'valid'
  const statusColor = status === 'valid' ? '#81c784' : status === 'repaired' ? '#ffb74d' : '#64b5f6'

  const prep = plan.preprocessing_strategy || {}
  const hpo = plan.hyperparameter_strategy || {}
  const ens = plan.ensemble_strategy || {}

  return (
    <div style={{ marginTop: 20, padding: 18, background: '#141414', border: '1px solid var(--border)', borderRadius: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, flexWrap: 'wrap', gap: 8 }}>
        <div style={{ fontWeight: 600, fontSize: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>📋 LLM AutoML Pipeline Plan</span>
          <span className="tag" style={{ background: isLlm ? 'rgba(156,39,176,0.2)' : 'rgba(33,150,243,0.2)', color: isLlm ? '#ce93d8' : '#64b5f6', borderColor: isLlm ? '#ce93d8' : '#64b5f6' }}>
            Source: {isLlm ? 'LLM' : 'Deterministic'}
          </span>
          <span className="tag" style={{ background: `${statusColor}22`, color: statusColor, borderColor: statusColor, textTransform: 'capitalize' }}>
            Status: {status}
          </span>
          {prov.provider_used && (
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              ({prov.provider_used})
            </span>
          )}
        </div>
        {hpo.n_folds && (
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            HPO: {hpo.n_folds} Folds · {hpo.n_trials} Trials · Mode: <code>{hpo.mode || 'automl'}</code>
          </div>
        )}
      </div>

      {/* Plan summary attributes */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginBottom: 16 }}>
        <div style={{ background: '#1c1c1c', padding: 12, borderRadius: 6, border: '1px solid #2a2a2a' }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 4 }}>Problem &amp; Target</div>
          <div style={{ fontWeight: 600, fontSize: 14 }}>
            <span style={{ textTransform: 'capitalize' }}>{plan.problem_type || exp?.problem_type}</span> · <code>{plan.target_column || exp?.target_column}</code>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
            Metric: <code>{plan.primary_metric || exp?.primary_metric}</code>
          </div>
        </div>

        <div style={{ background: '#1c1c1c', padding: 12, borderRadius: 6, border: '1px solid #2a2a2a' }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 4 }}>Preprocessing Strategy</div>
          <div style={{ fontSize: 13, color: '#fff', fontWeight: 500 }}>
            Numeric: {prep.numeric_imputation || 'median'} · Scale: {prep.scaling || 'standard'}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
            Categorical: {prep.categorical_imputation || 'mode'} (max {prep.max_categories || 20})
          </div>
        </div>

        <div style={{ background: '#1c1c1c', padding: 12, borderRadius: 6, border: '1px solid #2a2a2a' }}>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 4 }}>Ensemble Strategy</div>
          <div style={{ fontSize: 13, color: ens.enabled !== false ? '#81c784' : '#ffb74d', fontWeight: 500 }}>
            {ens.enabled !== false ? 'Enabled' : 'Disabled'}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
            {ens.strategies?.length > 0 ? ens.strategies.join(', ') : 'Default fusion'}
          </div>
        </div>
      </div>

      {/* Tri-stage comparison: LLM Recommendation vs Validated Plan vs Actually Executed */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12, marginBottom: 16 }}>
        {/* Stage 1: LLM Advisory */}
        <div style={{ background: '#181818', border: '1px solid #2a2a2a', borderRadius: 6, padding: 12 }}>
          <div style={{ fontWeight: 600, fontSize: 13, color: '#ce93d8', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>1️⃣ LLM Recommendation</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            <div><strong>Candidate Models:</strong></div>
            <div style={{ marginTop: 2, marginBottom: 6 }}>
              {(raw.candidate_models || plan.candidate_models || []).length > 0 ? (
                (raw.candidate_models || plan.candidate_models).map(m => (
                  <code key={typeof m === 'string' ? m : m.model_id} style={{ marginRight: 4, display: 'inline-block' }}>{typeof m === 'string' ? m : m.model_id}</code>
                ))
              ) : 'None specified'}
            </div>
            <div><strong>Feature Operations:</strong></div>
            <div style={{ marginTop: 2 }}>
              {(raw.feature_operations || plan.feature_operations || []).length > 0 ? (
                `${(raw.feature_operations || plan.feature_operations).length} operation(s) proposed`
              ) : '0 proposed'}
            </div>
            {plan.rationale && (
              <div style={{ marginTop: 6, fontStyle: 'italic', color: 'var(--text-muted)' }}>
                "{plan.rationale}"
              </div>
            )}
          </div>
        </div>

        {/* Stage 2: Validated Plan */}
        <div style={{ background: '#181818', border: '1px solid #2a2a2a', borderRadius: 6, padding: 12 }}>
          <div style={{ fontWeight: 600, fontSize: 13, color: '#64b5f6', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>2️⃣ Validated Plan</span>
            <span style={{ fontSize: 11, color: statusColor }}>({status})</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            <div><strong>Registry Approved Models:</strong></div>
            <div style={{ marginTop: 2, marginBottom: 6 }}>
              {(plan.candidate_models || []).map(m => (
                <code key={m} style={{ marginRight: 4, display: 'inline-block' }}>{m}</code>
              ))}
            </div>
            <div><strong>Approved Operations:</strong></div>
            <div style={{ marginTop: 2 }}>
              {(plan.feature_operations || []).length > 0 ? (
                plan.feature_operations.map((op, i) => (
                  <div key={i}>• <code>{op.operation}</code> on <code>{op.column}</code></div>
                ))
              ) : '0 approved operations'}
            </div>
            {(prov.rejected_models?.length > 0 || prov.rejected_operations?.length > 0) && (
              <div style={{ marginTop: 6, color: '#e57373' }}>
                Refused: {prov.rejected_models?.length || 0} models, {prov.rejected_operations?.length || 0} ops
              </div>
            )}
          </div>
        </div>

        {/* Stage 3: Actually Executed Pipeline */}
        <div style={{ background: '#181818', border: '1px solid #2a2a2a', borderRadius: 6, padding: 12 }}>
          <div style={{ fontWeight: 600, fontSize: 13, color: '#81c784', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>3️⃣ Actually Executed</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            {executed.candidate_models_trained ? (
              <>
                <div><strong>Models Trained ({executed.candidate_models_trained.length}):</strong></div>
                <div style={{ marginTop: 2, marginBottom: 6 }}>
                  {executed.candidate_models_trained.map(m => (
                    <code key={m} style={{ marginRight: 4, display: 'inline-block' }}>{m}</code>
                  ))}
                </div>
                <div><strong>Operations Applied:</strong> {executed.feature_operations_applied?.length || 0}</div>
                <div><strong>Ensemble Evaluated:</strong> {executed.ensemble_enabled ? 'Yes (OOF CV)' : 'No'}</div>
                <div><strong>Execution Seed:</strong> <code>{executed.seed ?? '42'}</code></div>
              </>
            ) : (
              <div style={{ color: 'var(--text-muted)' }}>
                Pipeline execution starts when you run training.
              </div>
            )}
          </div>
        </div>
      </div>

      <div style={{ fontSize: 12, color: 'var(--text-muted)', borderTop: '1px solid #222', paddingTop: 8 }}>
        🔒 <strong>Safety Guarantee:</strong> Controlled registry execution only. The LLM produces structured decisions, never executable code. All models are validated against <code>ModelRegistry</code> and operations against <code>OPERATION_REGISTRY</code> with strict target leakage prevention.
      </div>
    </div>
  )
}

/**
 * Renders the REAL per-trial Optuna history persisted at train time.
 * Only trials the backend actually recorded are shown: no progress bar, no
 * invented counts, no placeholder rows. A model whose search failed shows its
 * recorded status and the real error message instead of a trial table.
 */
function OptunaTrials({ models }) {
  const [selected, setSelected] = useState(
    models.find(m => m.is_best)?.model_name || models[0]?.model_name
  )
  const model = models.find(m => m.model_name === selected)
  const trials = model?.optimization_history || []
  const opt = model?.optimization || {}

  if (!model) return null

  return (
    <div style={{ marginTop: 20 }}>
      <div className="section-title">Optuna Trials (real search history)</div>
      <div style={{ marginBottom: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {models.map(m => (
          <button
            key={m.model_name}
            className={m.model_name === selected ? 'btn btn-primary' : 'btn btn-secondary'}
            style={{ fontSize: 12, padding: '6px 10px' }}
            onClick={() => setSelected(m.model_name)}
          >
            {m.display_name}
          </button>
        ))}
      </div>

      <div style={{ fontSize: 13, marginBottom: 8, color: 'var(--text-secondary)' }}>
        {trials.length === 0 ? (
          <>No trials were run for this model ({opt.status || 'unknown'}).</>
        ) : (
          <>
            {trials.length} recorded trial{trials.length === 1 ? '' : 's'} ·
            metric <code>{opt.metric}</code> ·
            scoring <code>{opt.scoring}</code> ·
            study direction <code>{opt.direction}</code> ·
            reported metric <code>{opt.raw_direction}</code> ·
            {opt.n_folds_used} × {opt.cv_strategy} CV on the training split ·
            seed {opt.seed}
          </>
        )}
      </div>

      {opt.error && (
        <div className="alert alert-danger" style={{ marginBottom: 10, fontSize: 13 }}>
          Optimization status <strong>{opt.status}</strong>: {opt.error}
        </div>
      )}
      {opt.fold_note && (
        <div className="alert alert-warning" style={{ marginBottom: 10, fontSize: 13 }}>
          {opt.fold_note}
        </div>
      )}

      {trials.length > 0 && (
        <div className="table-container" style={{ maxHeight: 320, overflowY: 'auto' }}>
          <table>
            <thead>
              <tr><th>Trial</th><th>Status</th><th>CV Score</th><th>Duration (s)</th><th>Params</th><th>Error</th></tr>
            </thead>
            <tbody>
              {trials.map(t => (
                <tr key={t.trial_number}>
                  <td>{t.trial_number}</td>
                  <td>{t.status}</td>
                  <td>{t.score != null ? t.score.toFixed(6) : '—'}</td>
                  <td>{t.duration?.toFixed(3)}</td>
                  <td style={{ fontSize: 11, maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {Object.entries(t.params || {}).map(([k, v]) => `${k}=${v}`).join(', ') || '—'}
                  </td>
                  <td style={{ fontSize: 11, color: 'var(--danger)' }}>{t.error || ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

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
  const pollRef = useRef(null)

  const load = useCallback(async () => {
    try {
      const r = await api.getExperiment(id)
      setExp(r.data)
      if (r.data.status === 'completed' || r.data.status === 'analyzed') {
        try { const pr = await api.profileDataset(r.data.dataset_id); setProfile(pr.data) } catch {}
        // Hydrate from the stored analysis; only ask the backend to run one when
        // there is none (and never while it is training or already completed).
        if (r.data.has_analysis && r.data.llm_analysis) {
          setLlmAnalysis(r.data.llm_analysis.result || r.data.llm_analysis)
        } else if (!r.data.has_analysis && r.data.status !== 'completed') {
          try { const la = await api.analyzeExperiment(id); setLlmAnalysis(la.data?.result || la.data) } catch {}
        }
        if (r.data.status === 'completed') {
          try { const rs = await api.getResults(id); setResults(rs.data) } catch {}
          try { const ex = await api.getExplainability(id); setExplainability(ex.data) } catch {}
          try { const sc = await api.getInputSchema(id); setInputSchema(sc.data?.fields || []) } catch {}
          try { const pg = await api.getProgress(id); setProgress(pg.data) } catch {}
        }
        if (r.data.report_path) setReportUrl(`/reports/report_${id}.html`)
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

  const [progress, setProgress] = useState(null)
  const [polling, setPolling] = useState(false)

  // Poll the REAL progress endpoint while a training request is in flight.
  // The endpoint reports observed state only, so there is nothing to fake
  // here; when the run ends, the last snapshot stays on screen.
  const startProgressPolling = useCallback(() => {
    if (polling) return
    setPolling(true)
    const tick = async () => {
      try {
        const r = await api.getProgress(id)
        setProgress(r.data)
        const st = r.data?.progress?.status
        if (st && st !== 'running' && st !== 'queued') setPolling(false)
      } catch {
        // A failed poll is not a failed run: keep polling silently.
      }
    }
    tick()
    pollRef.current = setInterval(tick, 1500)
  }, [id, polling])

  const stopProgressPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = null
    setPolling(false)
  }, [])

  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current)
  }, [])

  const runTraining = async (fastDemo) => {
    setActionLoading(true); setActionMsg(fastDemo ? 'Fast demo: 3 models, 5 trials...' : 'Training all models with Optuna optimization...')
    startProgressPolling()
    try {
      const r = await api.trainExperiment(id, fastDemo)
      setResults(r.data)
      await load()
      setActiveStep(6)
    }
    catch (e) { setError(`Training failed: ${e.response?.data?.detail || e.message}`) }
    finally {
      setActionLoading(false); setActionMsg('')
      stopProgressPolling()
      // Always take a final reading so the persisted snapshot is shown.
      try { const r = await api.getProgress(id); setProgress(r.data) } catch {}
    }
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
    const newMsgs = [...chatMsgs, { role: 'user', text: msg }]
    setChatMsgs(newMsgs)
    setChatInput('')
    setChatLoading(true)
    try {
      const history = newMsgs.slice(-10).map(m => ({ role: m.role, content: m.text }))
      const r = await api.chat(msg, parseInt(id), 'project', history)
      setChatMsgs(prev => [...prev, { role: 'assistant', text: r.data.response }])
    } catch {
      setChatMsgs(prev => [...prev, { role: 'assistant', text: 'Sorry, something went wrong.' }])
    } finally {
      setChatLoading(false)
    }
  }

  const statusToStep = (status) => {
    const map = { created: 0, profiling: 1, analyzed: 2, preprocessing: 3, training: 4, completed: 6, failed: 0 }
    return map[status] ?? 0
  }

  if (loading) return <LoadingSpinner message="Loading experiment..." />
  if (error && !exp) return <div className="alert alert-danger">{error}</div>

  const models = results?.models || []
  const bestModel = results?.best_model || {}
  const problemType = results?.problem_type || exp?.problem_type
  // Comparison chart uses the backend-resolved primary metric. It used to take
  // "the first number in the metrics dict", which showed accuracy for an F1
  // experiment - a real, silently wrong number.
  const comparisonData = models.filter(m => m.status === 'COMPLETED').map(m => ({
    display_name: m.display_name,
    score: m.holdout_primary ?? null,
    metric_key: m.holdout_primary_key,
    model_name: m.model_name,
  }))

  const expl = explainability || {}
  const cm = models.find(m => m.is_best)?.optimized_metrics?.confusion_matrix || []
  const cmLabels = models.find(m => m.is_best)?.optimized_metrics?.class_distribution
    ? Object.keys(models.find(m => m.is_best)?.optimized_metrics?.class_distribution || {}) : []

  return (
    <div className="page">
      {error && <div className="alert alert-danger" style={{ marginBottom: 16 }}>{error} <button className="btn btn-sm" onClick={() => setError('')}>✕</button></div>}

      <div className="page-header">
        <div>
          <h1>{exp?.name || 'Experiment'}</h1>
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
            <h2>AI Analysis</h2>
            {!llmAnalysis ? (
              <div>
                <p style={{ marginBottom: 16, color: 'var(--text-secondary)' }}>The LLM will analyze your dataset structure and recommend the optimal ML strategy.</p>
                <button className="btn btn-primary" onClick={runAnalysis} disabled={actionLoading}>▶ Run AI Analysis</button>
              </div>
            ) : (
              <div>
                <div className="alert alert-info" style={{ marginBottom: 20 }}>
                  <strong>Provider:</strong> {llmAnalysis.provider_used || 'Fallback'}{llmAnalysis.is_fallback ? ' (rule-based — no LLM API key)' : ''} · <strong>Confidence:</strong> {llmAnalysis.confidence != null ? `${(llmAnalysis.confidence * 100).toFixed(0)}%` : 'not reported'}
                </div>
                <div className="metrics-grid" style={{ marginBottom: 20 }}>
                  <div className="metric-card"><div className="metric-value" style={{ fontSize: 18, textTransform: 'capitalize' }}>{llmAnalysis.problem_type}</div><div className="metric-label">Problem Type</div></div>
                  <div className="metric-card"><div className="metric-value" style={{ fontSize: 16 }}>{llmAnalysis.recommended_metric}</div><div className="metric-label">Recommended Metric</div></div>
                </div>
                {llmAnalysis.problem_understanding && (
                  <p style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.7, marginBottom: 16 }}><strong>Understanding:</strong> {llmAnalysis.problem_understanding}</p>
                )}
                <div className="section-title">Reasoning</div>
                <p style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.7, marginBottom: 16 }}>{llmAnalysis.reasoning}</p>
                {llmAnalysis.candidate_models?.length > 0 && (
                  <><div className="section-title">Recommended Models</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
                    {llmAnalysis.candidate_models.map(m => <span key={m} className="tag">{m}</span>)}
                  </div></>
                )}
                {llmAnalysis.model_recommendations?.length > 0 && (
                  <><div className="section-title">Model Recommendations · {llmAnalysis.model_selection_source === 'provider' ? 'LLM registry-validated' : llmAnalysis.model_selection_source === 'deterministic_defaults' ? 'deterministic (rule-based)' : llmAnalysis.model_selection_source}</div>
                  <ul style={{ paddingLeft: 20, fontSize: 14, color: 'var(--text-secondary)', marginBottom: 12 }}>
                    {llmAnalysis.model_recommendations.map((r, i) => (
                      <li key={i}><strong>{r.display_name || r.model_id}</strong>{r.reason ? ` — ${r.reason}` : ''}</li>
                    ))}
                  </ul></>
                )}
                {llmAnalysis.rejected_model_recommendations?.length > 0 && (
                  <><div className="section-title">Rejected Model Suggestions</div>
                  <ul style={{ paddingLeft: 20, fontSize: 14, color: 'var(--text-secondary)', marginBottom: 12 }}>
                    {llmAnalysis.rejected_model_recommendations.map((r, i) => (
                      <li key={i}><strong>{r.model_id}</strong> — {r.reason}</li>
                    ))}
                  </ul></>
                )}
                {llmAnalysis.preprocessing?.length > 0 && (
                  <><div className="section-title">Preprocessing Advice</div>
                  <ul style={{ paddingLeft: 20, fontSize: 14, color: 'var(--text-secondary)' }}>
                    {llmAnalysis.preprocessing.map((p, i) => <li key={i}><strong>{p.column}</strong>: {p.action} — {p.reason}</li>)}
                  </ul></>
                )}
                {(llmAnalysis.suggested_operations?.length > 0 || llmAnalysis.rejected_operations?.length > 0) && (
                  <><div className="section-title">Feature Operations · {llmAnalysis.operation_source === 'provider' ? 'registry-validated' : 'deterministic defaults'}</div>
                  {llmAnalysis.suggested_operations?.length > 0 && (
                    <ul style={{ paddingLeft: 20, fontSize: 14, color: 'var(--text-secondary)', marginBottom: 12 }}>
                      {llmAnalysis.suggested_operations.map((op, i) => (
                        <li key={i}><strong>{op.column}</strong>: {op.operation}{op.params && Object.keys(op.params).length > 0 ? ` (${Object.entries(op.params).map(([k, v]) => `${k}=${v}`).join(', ')})` : ''}{op.reason ? ` — ${op.reason}` : ''}</li>
                      ))}
                    </ul>
                  )}
                  {llmAnalysis.rejected_operations?.length > 0 && (
                    <details style={{ fontSize: 14, color: 'var(--text-secondary)', marginBottom: 16 }}>
                      <summary>{llmAnalysis.rejected_operations.length} suggestion(s) rejected by the safety registry</summary>
                      <ul style={{ paddingLeft: 20, marginTop: 8 }}>
                        {llmAnalysis.rejected_operations.map((op, i) => (
                          <li key={i}>{op.column ? <><strong>{op.column}</strong> {op.operation} — </> : null}{op.reason}</li>
                        ))}
                      </ul>
                    </details>
                  )}</>
                )}
                {llmAnalysis.warnings?.length > 0 && (
                  <div className="alert alert-warning" style={{ marginTop: 16 }}>
                    <strong>Warnings:</strong>
                    <ul style={{ marginTop: 8, paddingLeft: 20 }}>{llmAnalysis.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
                  </div>
                )}
                {llmAnalysis.pipeline_plan && (
                  <PipelinePlanSection
                    planProvenance={{
                      validated_plan: llmAnalysis.pipeline_plan,
                      source: llmAnalysis.pipeline_plan.source || (llmAnalysis.is_fallback ? 'deterministic_fallback' : 'llm'),
                      provider_used: llmAnalysis.provider_used,
                      validation_status: llmAnalysis.pipeline_plan.validation_status || 'valid',
                      raw_recommendation: llmAnalysis.pipeline_plan,
                    }}
                    fallbackPlan={llmAnalysis.pipeline_plan}
                    exp={exp}
                  />
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
        {progress && (
              <TrainingProgressPanel payload={progress} />
            )}
            {activeStep === 6 && (
          <div>
            <h2>Model Evaluation</h2>
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
                {results.selection && (
                  <div style={{ background: '#1f1f1f', border: '1px solid var(--border)', borderRadius: 8, padding: 12, marginBottom: 20, fontSize: 13, lineHeight: 1.8 }}>
                    <strong>Selection protocol:</strong> ranked by{' '}
                    <code>{results.selection.evidence_source?.replace(/_/g, ' ')}</code>{' '}
                    using <code>{results.selection.selection_metric}</code>{' '}
                    (scoring <code>{results.selection.selection_scoring}</code>, direction{' '}
                    <code>{results.selection.selection_direction}</code>).{' '}
                    Selected on CV score{' '}
                    <strong>
                      {results.selection.selected_model_cv_score != null
                        ? results.selection.selected_model_cv_score.toFixed(4)
                        : 'n/a'}
                    </strong>{' '}
                    over {results.selection.selected_model_cv_folds ?? '?'} folds.
                    <br />
                    Final holdout metrics were computed <strong>once, after selection</strong>;
                    the holdout was not used to choose the model.
                  </div>
                )}
                {results.llm_explanation && (
                  <div style={{ background: '#1f1f1f', border: '1px solid var(--border)', borderRadius: 8, padding: 16, marginBottom: 20, fontSize: 14, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                    {results.llm_explanation}
                  </div>
                )}
                <div className="chart-container">
                  <ModelComparison models={comparisonData} metric={results.primary_metric} />
                </div>
                <div className="section-title" style={{ marginTop: 20 }}>Model Results Table</div>
                {results.optimization_budget && (
                  <div style={{ fontSize: 13, marginBottom: 10, color: 'var(--text-secondary)' }}>
                    Optuna: metric <code>{results.optimization_budget.metric_spec?.metric}</code>{' '}
                    · scoring <code>{results.optimization_budget.metric_spec?.scoring}</code>{' '}
                    · study direction <code>{results.optimization_budget.metric_spec?.direction}</code>
                    {results.optimization_budget.metric_spec?.raw_direction !== results.optimization_budget.metric_spec?.direction && (
                      <> (reported metric: {results.optimization_budget.metric_spec?.raw_direction})</>
                    )}
                    {' '}· {results.optimization_budget.n_trials_used} trials ×{' '}
                    {results.optimization_budget.n_folds_used} folds per model
                    {' '}· seed {results.optimization_budget.seed}
                    {results.optimization_budget.fast_demo && (
                      <strong> · fast demo: reduced search budget</strong>
                    )}
                  </div>
                )}
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Status</th>
                        <th title="Cross-validated score - the evidence selection ranked on">CV score (selection)</th>
                        <th title="Standard deviation of the CV fold scores">CV std</th>
                        <th>Folds</th>
                        <th>Optuna</th>
                        <th>Trials</th>
                        <th title={`Holdout evaluation, computed once after selection`}>Holdout {METRIC_LABELS[results.models?.find(mm => mm.holdout_primary_key)?.holdout_primary_key] || ''}</th>
                        <th>Train time</th>
                        <th>Best params</th>
                      </tr>
                    </thead>
                    <tbody>
                      {models.map(m => {
                        const met = m.optimized_metrics || m.baseline_metrics
                        const o = m.optimization || {}
                        const holdout = realMetrics(met, problemType)
                        const failed = m.status === 'FAILED'
                        return (
                          <tr key={m.model_name} style={m.is_best ? { background: 'rgba(229,9,20,0.12)' } : {}}>
                            <td><strong>{m.display_name}</strong>{m.is_best ? ' 🏆' : ''}</td>
                            <td><StatusBadge status={m.status} /></td>
                            <td title="Cross-validated score: the evidence model selection ranked on">
                              {m.selection_score != null ? m.selection_score.toFixed(4) : '—'}
                            </td>
                            <td title="Spread of the CV fold scores — lower means more fold-stable">
                              {m.selection_cv_std != null ? m.selection_cv_std.toFixed(4) : '—'}
                            </td>
                            <td>{m.selection_cv_folds ?? '—'}</td>
                            <td title={o.error ? `Error: ${o.error}` : (o.fold_note || '')}>
                              {o.status || '—'}
                            </td>
                            <td>
                              {o.n_trials_completed != null
                                ? `${o.n_trials_completed}/${o.n_trials_requested}`
                                : '—'}
                              {o.n_trials_failed > 0 ? ` (${o.n_trials_failed} failed)` : ''}
                            </td>
                            <td>
                              {failed ? (
                                <span style={{ color: 'var(--danger)', fontSize: 12 }}>
                                  not evaluated — {m.error_message || 'training failed'}
                                </span>
                              ) : holdout.length === 0 ? (
                                <span style={{ color: 'var(--text-muted)' }}>—</span>
                              ) : (
                                <span title={`Keys: ${holdout.map(h => h.key).join(', ')}`}>
                                  {holdout.map(h => `${METRIC_LABELS[h.key] || h.key} ${h.value.toFixed(4)}`).join(' · ')}
                                </span>
                              )}
                            </td>
                            <td>{m.training_time != null ? `${m.training_time.toFixed(2)}s` : '—'}</td>
                            <td style={{ fontSize: 12, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {Object.entries(m.best_params || {}).map(([k, v]) => `${k}=${v}`).join(', ') || '—'}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 10 }}>
                  CV columns come from cross-validation on the training split and are what
                  selection ranked on. Holdout columns were computed once, after selection,
                  on rows the model never saw. A metric is listed only when the evaluator
                  actually computed it.
                </p>
                <ResearchEvaluationSection results={results} exp={exp} />
                <PipelinePlanSection
                  planProvenance={results.pipeline_plan_provenance}
                  fallbackPlan={results.research_evaluation?.pipeline_plan}
                  exp={exp}
                />
                <MultiSeedBenchmarkSection />
                <ModelFusionSection ensembleCandidates={results.ensemble_candidates} models={models} bestModelName={results.best_model_name} primaryMetric={results.primary_metric} />
                <OptunaTrials models={models} />
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
            <h2>Explainability</h2>
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
            <h2>Prediction</h2>
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
            <h2>Report & Download</h2>
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
            <h3>AI Assistant</h3>
            <button className="chat-close" onClick={() => setChatOpen(false)}><FiX /></button>
          </div>
          <div className="chat-messages">
            {chatMsgs.map((m, i) => (
              <div key={i} className={`chat-msg ${m.role}`}>
                {m.role === 'assistant' ? (
                  <ReactMarkdown
                    components={{
                      code({ node, inline, className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || '')
                        return !inline ? (
                          <CodeBlock
                            code={String(children).replace(/\n$/, '')}
                            language={match ? match[1] : ''}
                          />
                        ) : (
                          <code className="inline-code" {...props}>
                            {children}
                          </code>
                        )
                      },
                    }}
                  >
                    {m.text}
                  </ReactMarkdown>
                ) : (
                  m.text
                )}
              </div>
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
