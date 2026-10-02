import React from 'react'
import Plot from 'react-plotly.js'

export default function FeatureImportance({ features = [], title = 'Feature Importance', method = '' }) {
  if (!features.length) return <div className="empty-state"><p>No feature importance data available.</p></div>
  const top = [...features].sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance)).slice(0, 15).reverse()
  return (
    <Plot
      data={[{
        type: 'bar',
        x: top.map(f => Math.abs(f.importance)),
        y: top.map(f => f.feature),
        orientation: 'h',
        marker: {
          color: top.map((_, i) => `hsl(355, 72%, ${42 + i * 3}%)`),
        },
        text: top.map(f => Math.abs(f.importance).toFixed(4)),
        textposition: 'auto',
        textfont: { color: '#ffffff' },
      }]}
      layout={{
        title: { text: `${title}${method ? ` (${method})` : ''}`, font: { size: 15, color: '#ffffff' } },
        xaxis: { title: 'Importance Score', tickfont: { color: '#b3b3b3' }, gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' },
        margin: { l: 180, r: 40, t: 50, b: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Helvetica Neue, Arial, sans-serif', size: 12, color: '#b3b3b3' },
        height: 420,
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ displayModeBar: false }}
    />
  )
}
