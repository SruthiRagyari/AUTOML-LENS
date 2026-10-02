import React from 'react'

const PHASE_LABEL = {
  preparing: 'Preparing',
  training: 'Fitting baseline',
  optimizing: 'Optuna search',
  refining: 'Refitting best params',
  scoring: 'Scoring holdout',
  selecting: 'Selecting best model',
  persisting: 'Saving model',
}

const MODEL_STATUS_STYLE = {
  queued: { color: 'var(--text-muted)', icon: '○' },
  running: { color: 'var(--primary)', icon: '◐' },
  completed: { color: 'var(--success)', icon: '✓' },
  failed: { color: 'var(--danger)', icon: '✕' },
}

/**
 * Real observed training state, straight from
 * GET /api/experiments/{id}/progress.
 *
 * Deliberately renders NO progress bar and NO ETA. The work is not linearly
 * divisible, so a percentage would have to be invented. What is shown is
 * measured: which model is actually running, which phase it is in, which
 * Optuna trial is being evaluated, and exact completed/failed counts.
 *
 * A failed model is shown with its real error - it is never counted as a
 * success and never hidden.
 */
export default function TrainingProgressPanel({ payload }) {
  if (!payload) return null
  const p = payload.progress
  if (!p) {
    return (
      <div className="alert alert-info">
        {payload.source === 'none'
          ? 'Training has not started yet, or the run is not owned by this server process.'
          : 'No progress snapshot available.'}
      </div>
    )
  }

  const running = p.status === 'running' || p.status === 'queued'
  const models = p.models || []

  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 8 }}>
        <h2 style={{ margin: 0 }}>Training Progress</h2>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <span className={`badge ${running ? 'badge-warning' : p.status === 'completed' ? 'badge-success' : 'badge-danger'}`}>
            {p.status}
          </span>
          <span className="badge badge-default">{payload.source}</span>
        </div>
      </div>

      <div className="metrics-grid" style={{ marginBottom: 16 }}>
        <div className="metric-card">
          <div className="metric-value">{p.models_completed}/{p.models_total}</div>
          <div className="metric-label">Models completed</div>
        </div>
        <div className="metric-card">
          <div className="metric-value" style={{ color: p.models_failed > 0 ? 'var(--danger)' : undefined }}>
            {p.models_failed}
          </div>
          <div className="metric-label">Failed</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{p.models_remaining}</div>
          <div className="metric-label">Remaining</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{p.elapsed_seconds != null ? `${p.elapsed_seconds.toFixed(0)}s` : '—'}</div>
          <div className="metric-label">Elapsed (measured)</div>
        </div>
      </div>

      <div style={{ fontSize: 14, marginBottom: 16, lineHeight: 1.9 }}>
        <div>
          <strong>Phase:</strong>{' '}
          {p.phase ? (PHASE_LABEL[p.phase] || p.phase) : <em>not reported yet</em>}
        </div>
        <div>
          <strong>Current model:</strong>{' '}
          {p.current_model_display_name
            ? <>{p.current_model_display_name} <span style={{ color: 'var(--text-muted)' }}>({p.current_model})</span></>
            : <em>{p.status === 'completed' ? 'none — run finished' : 'none reported'}</em>}
        </div>
        <div>
          <strong>Optuna trial:</strong>{' '}
          {p.current_trial != null
            ? <>trial {p.current_trial} of {p.n_trials_per_model ?? '?'}</>
            : <em>no trial in flight</em>}
        </div>
      </div>

      {p.error && (
        <div className="alert alert-danger" style={{ marginBottom: 16 }}>
          <strong>Run error:</strong> {p.error}
        </div>
      )}

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Status</th>
              <th>Phase</th>
              <th>Trials</th>
              <th>Elapsed</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody>
            {models.map(m => {
              const st = MODEL_STATUS_STYLE[m.status] || { color: 'var(--text-muted)', icon: '?' }
              return (
                <tr key={m.model_name}>
                  <td>
                    <span style={{ color: st.color, marginRight: 6 }}>{st.icon}</span>
                    <strong>{m.display_name}</strong>
                  </td>
                  <td>
                    <span className={`badge ${m.status === 'completed' ? 'badge-success'
                      : m.status === 'failed' ? 'badge-danger'
                        : m.status === 'running' ? 'badge-warning' : 'badge-info'}`}>
                      {m.status}
                    </span>
                  </td>
                  <td>{m.phase ? (PHASE_LABEL[m.phase] || m.phase) : '—'}</td>
                  <td>
                    {m.trials_completed ?? 0}
                    {m.trials_failed > 0 ? ` (${m.trials_failed} failed)` : ''}
                  </td>
                  <td>{m.elapsed_seconds != null ? `${m.elapsed_seconds.toFixed(1)}s` : '—'}</td>
                  <td style={{ fontSize: 12, color: m.error ? 'var(--danger)' : 'var(--text-muted)', maxWidth: 260 }}>
                    {m.error || '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 12, marginBottom: 0 }}>
        {p.notes}
      </p>
    </div>
  )
}