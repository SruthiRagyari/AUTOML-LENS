import React from 'react'
import Plot from 'react-plotly.js'

export default function ModelComparison({ models = [], metric = 'Score' }) {
  if (!models.length) return <div className="empty-state"><p>No models to compare yet.</p></div>
  const sorted = [...models].sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
  return (
    <Plot
      data={[{
        type: 'bar',
        x: sorted.map(m => (m.score ?? 0).toFixed(4)),
        y: sorted.map(m => m.display_name),
        orientation: 'h',
        marker: { color: sorted.map((_, i) => i === 0 ? '#4361ee' : '#8ba7ff') },
        text: sorted.map(m => (m.score ?? 0).toFixed(4)),
        textposition: 'auto',
      }]}
      layout={{
        title: { text: `Model Comparison — ${metric}`, font: { size: 15 } },
        xaxis: { title: metric },
        yaxis: { autorange: 'reversed' },
        margin: { l: 180, r: 40, t: 50, b: 50 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter, sans-serif', size: 13 },
        height: 300 + sorted.length * 36,
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ displayModeBar: false }}
    />
  )
}
