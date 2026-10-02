import React from 'react'
import Plot from 'react-plotly.js'

export default function ConfusionMatrix({ matrix = [], labels = [] }) {
  if (!matrix.length) return <div className="empty-state"><p>No confusion matrix available.</p></div>
  return (
    <Plot
      data={[{
        z: matrix,
        x: labels.length ? labels : matrix[0].map((_, i) => `Class ${i}`),
        y: labels.length ? labels : matrix.map((_, i) => `Class ${i}`),
        type: 'heatmap',
        colorscale: 'Blues',
        text: matrix.map(row => row.map(v => String(v))),
        texttemplate: '%{text}',
        showscale: true,
      }]}
      layout={{
        title: { text: 'Confusion Matrix', font: { size: 15 } },
        xaxis: { title: 'Predicted' },
        yaxis: { title: 'Actual', autorange: 'reversed' },
        margin: { l: 80, r: 40, t: 50, b: 80 },
        paper_bgcolor: 'transparent',
        font: { family: 'Inter, sans-serif', size: 12 },
        height: 360,
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ displayModeBar: false }}
    />
  )
}
