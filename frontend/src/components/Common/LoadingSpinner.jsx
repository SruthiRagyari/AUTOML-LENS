import React from 'react'
export default function LoadingSpinner({ message = 'Loading...' }) {
  return (
    <div className="loading-container">
      <div className="spinner" />
      {message && <p className="loading-text">{message}</p>}
    </div>
  )
}
