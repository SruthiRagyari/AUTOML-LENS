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
        marker: { color: sorted.map((_, i) => i === 0 ? '#E50914' : '#6d6d6d') },
        text: sorted.map(m => (m.score ?? 0).toFixed(4)),
        textposition: 'auto',
        textfont: { color: '#ffffff' },
      }]}
      layout={{
        title: { text: `Model Comparison — ${metric}`, font: { size: 15, color: '#ffffff' } },
        xaxis: { title: metric, tickfont: { color: '#b3b3b3' }, gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' },
        yaxis: { autorange: 'reversed', tickfont: { color: '#b3b3b3' } },
        margin: { l: 180, r: 40, t: 50, b: 50 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Helvetica Neue, Arial, sans-serif', size: 13, color: '#b3b3b3' },
        height: 300 + sorted.length * 36,
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ displayModeBar: false }}
    />
  )
}
