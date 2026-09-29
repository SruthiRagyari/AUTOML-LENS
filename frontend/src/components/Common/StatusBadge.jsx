import React from 'react'
const MAP = {
  completed: 'badge-success', success: 'badge-success',
  running: 'badge-warning', training: 'badge-warning', optimizing: 'badge-warning', analyzing: 'badge-warning',
  failed: 'badge-danger', error: 'badge-danger',
  created: 'badge-info', queued: 'badge-info', analyzed: 'badge-info',
}
export default function StatusBadge({ status }) {
  const cls = MAP[(status || '').toLowerCase()] || 'badge-default'
  return <span className={`badge ${cls}`}>{status || 'unknown'}</span>
}
