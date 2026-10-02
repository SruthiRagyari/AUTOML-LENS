import React from 'react'
export default function EmptyState({ icon = '📭', title, description, action }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">{icon}</div>
      <div className="empty-title">{title}</div>
      {description && <p className="empty-desc">{description}</p>}
      {action && <div style={{ marginTop: 20 }}>{action}</div>}
    </div>
  )
}
