import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { FiChevronLeft, FiChevronRight } from 'react-icons/fi'
import StatusBadge from './StatusBadge'
import EmptyState from './EmptyState'
import LoadingSpinner from './LoadingSpinner'

/**
 * Deterministic tile background derived from the item's own label: the same
 * name always produces the same gradient. Nothing is invented - only a hue is
 * picked from the characters that are already on screen.
 */
export function tileGradient(seed = '') {
  let h = 0
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) % 360
  return `linear-gradient(135deg, hsl(${h}, 42%, 26%), hsl(${(h + 45) % 360}, 48%, 12%))`
}

/** One experiment as a gradient tile: dataset name, problem type, status. */
export function ExperimentTile({ experiment, datasetName }) {
  const name = datasetName || `Dataset #${experiment.dataset_id}`
  const problemType = experiment.problem_type || 'auto'
  return (
    <Link
      to={`/experiment/${experiment.id}`}
      className="tile"
      style={{ background: tileGradient(`${name}-${problemType}`) }}
      aria-label={`Experiment ${experiment.name} on ${name}: ${problemType}, status ${experiment.status}`}
    >
      <div className="tile-name">{name}</div>
      <div className="tile-problem">{problemType}</div>
      <div className="tile-sub">{experiment.name}</div>
      <div className="tile-foot"><StatusBadge status={experiment.status} /></div>
    </Link>
  )
}

/** One uploaded dataset as a gradient tile: name, shape, upload date. */
export function DatasetTile({ dataset }) {
  const name = dataset.original_filename || dataset.filename || `Dataset #${dataset.id}`
  return (
    <div
      className="tile"
      style={{ background: tileGradient(name) }}
      aria-label={`Dataset ${name}: ${dataset.rows} rows, ${dataset.columns} columns`}
    >
      <div className="tile-name">{name}</div>
      <div className="tile-problem">{dataset.rows} rows · {dataset.columns} columns</div>
      <div className="tile-sub">uploaded {new Date(dataset.uploaded_at).toLocaleDateString()}</div>
    </div>
  )
}

/**
 * Horizontal, scroll-snapping row of cards with left/right arrow buttons.
 * The track is focusable so the keyboard can scroll it directly, and both
 * arrows are real buttons with labels. Empty and error states are explicit.
 */
export default function ScrollRow({
  title, items = [], renderItem, loading = false, error = '',
  emptyTitle, emptyDescription, emptyIcon = '📼',
}) {
  const trackRef = useRef(null)
  const [atStart, setAtStart] = useState(true)
  const [atEnd, setAtEnd] = useState(true)

  const update = () => {
    const el = trackRef.current
    if (!el) return
    setAtStart(el.scrollLeft <= 4)
    setAtEnd(el.scrollLeft + el.clientWidth >= el.scrollWidth - 4)
  }

  useEffect(() => {
    update()
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [items])

  const page = (dir) => {
    const el = trackRef.current
    if (!el) return
    el.scrollBy({ left: dir * Math.max(el.clientWidth * 0.8, 240), behavior: 'smooth' })
  }

  return (
    <section className="scroll-row">
      <div className="scroll-row-head">
        <h2>{title}</h2>
        <div className="scroll-row-arrows">
          <button type="button" className="row-arrow" onClick={() => page(-1)}
            disabled={loading || atStart} aria-label={`Scroll ${title} left`}>
            <FiChevronLeft />
          </button>
          <button type="button" className="row-arrow" onClick={() => page(1)}
            disabled={loading || atEnd} aria-label={`Scroll ${title} right`}>
            <FiChevronRight />
          </button>
        </div>
      </div>
      {error ? (
        <div className="alert alert-danger">{error}</div>
      ) : loading ? (
        <LoadingSpinner message="Loading..." />
      ) : items.length === 0 ? (
        <EmptyState icon={emptyIcon} title={emptyTitle} description={emptyDescription} />
      ) : (
        <div className="scroll-row-track" ref={trackRef} onScroll={update}
          tabIndex={0} role="list" aria-label={title}>
          {items.map((item, i) => (
            <div className="scroll-row-item" role="listitem" key={item.id ?? i}>
              {renderItem(item)}
            </div>
          ))}
        </div>
      )}
    </section>
  )
}