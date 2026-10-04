import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import {
  FiDatabase,
  FiUploadCloud,
  FiEye,
  FiCheckCircle,
  FiAlertTriangle,
  FiTrash2,
  FiPlus,
  FiX,
  FiFileText
} from 'react-icons/fi'
import { api } from '../services/api'
import LoadingSpinner from '../components/Common/LoadingSpinner'

export default function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [profileModal, setProfileModal] = useState(null)
  const [profileLoading, setProfileLoading] = useState(false)
  const [suitabilityModal, setSuitabilityModal] = useState(null)
  const [suitabilityLoading, setSuitabilityLoading] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    loadDatasets()
  }, [])

  const loadDatasets = async () => {
    setLoading(true)
    try {
      const res = await api.listDatasets()
      setDatasets(res.data || [])
    } catch (err) {
      console.error('Failed to load datasets:', err)
    } finally {
      setLoading(false)
    }
  }

  const MAX_DATASET_SIZE_BYTES = 1024 * 1024 * 1024 // 1 GB

  const onDrop = async (acceptedFiles, fileRejections) => {
    if (fileRejections && fileRejections.length > 0) {
      const rej = fileRejections[0]
      if (rej.errors?.some((e) => e.code === 'file-too-large')) {
        setUploadError('Dataset exceeds the maximum allowed size of 1 GB.')
      } else {
        setUploadError(rej.errors?.[0]?.message || 'File rejected')
      }
      return
    }
    if (!acceptedFiles || acceptedFiles.length === 0) return
    const file = acceptedFiles[0]
    if (file.size > MAX_DATASET_SIZE_BYTES) {
      setUploadError('Dataset exceeds the maximum allowed size of 1 GB.')
      return
    }
    setUploading(true)
    setUploadError(null)

    try {
      await api.uploadDataset(file)
      await loadDatasets()
    } catch (err) {
      setUploadError(err.response?.data?.detail || err.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    maxSize: MAX_DATASET_SIZE_BYTES,
    maxFiles: 1,
    disabled: uploading,
  })

  const viewProfile = async (dsId) => {
    setProfileLoading(true)
    setProfileModal({ loading: true, id: dsId })
    try {
      const res = await api.profileDataset(dsId)
      setProfileModal(res.data)
    } catch (err) {
      setProfileModal({ error: err.response?.data?.detail || err.message })
    } finally {
      setProfileLoading(false)
    }
  }

  const viewSuitability = async (dsId) => {
    setSuitabilityLoading(true)
    setSuitabilityModal({ loading: true, id: dsId })
    try {
      const res = await api.getSuitability(dsId)
      setSuitabilityModal(res.data)
    } catch (err) {
      setSuitabilityModal({ error: err.response?.data?.detail || err.message })
    } finally {
      setSuitabilityLoading(false)
    }
  }

  const handleDelete = async (dsId, e) => {
    e.stopPropagation()
    if (!window.confirm(`Are you sure you want to delete dataset #${dsId}?`)) return
    try {
      await api.deleteDataset(dsId)
      await loadDatasets()
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to delete dataset. It might be referenced by experiments.')
    }
  }

  return (
    <div className="page-container">
      {/* Hero Header */}
      <div className="netflix-hero-banner" style={{ padding: '40px 36px', marginBottom: '32px' }}>
        <div className="hero-pill-badge">Dataset Ingestion & Quality</div>
        <h1 className="hero-title" style={{ fontSize: '36px' }}>Dataset Management Hub</h1>
        <p className="hero-desc" style={{ fontSize: '16px', maxWidth: '650px' }}>
          Upload raw CSV or XLSX tabular data, inspect column distributions, assess ML task suitability,
          and launch end-to-end reproducible AutoML experiments.
        </p>
      </div>

      {/* Upload Dropzone */}
      <div className="netflix-card" style={{ marginBottom: '36px' }}>
        <h2 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FiUploadCloud style={{ color: 'var(--netflix-red)' }} />
          <span>Upload New Tabular Dataset</span>
        </h2>
        <div
          {...getRootProps()}
          className={`netflix-dropzone ${isDragActive ? 'active' : ''}`}
        >
          <input {...getInputProps()} />
          <FiDatabase style={{ fontSize: '42px', color: isDragActive ? 'var(--netflix-red)' : 'var(--text-secondary)', marginBottom: '12px' }} />
          {uploading ? (
            <div>
              <LoadingSpinner />
              <p style={{ marginTop: '12px', color: 'var(--text-secondary)' }}>Parsing and validating tabular structure...</p>
            </div>
          ) : (
            <div>
              <p style={{ fontSize: '16px', fontWeight: '600', marginBottom: '6px' }}>
                {isDragActive ? 'Drop your CSV/XLSX file here...' : 'Drag & drop your CSV or XLSX file here'}
              </p>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                Supports UTF-8 CSV, Excel spreadsheets up to 1 GB
              </p>
            </div>
          )}
        </div>
        {uploadError && (
          <div style={{ marginTop: '14px', padding: '12px', background: 'var(--danger-bg)', border: '1px solid var(--danger)', borderRadius: 'var(--radius-sm)', color: '#ff6b72', fontSize: '13px' }}>
            <FiAlertTriangle style={{ marginRight: '6px' }} />
            {uploadError}
          </div>
        )}
      </div>

      {/* Dataset Table / Catalog */}
      <div className="netflix-card">
        <div className="netflix-row-header">
          <h2 className="netflix-row-title">Uploaded Datasets ({datasets.length})</h2>
          <button className="btn-netflix-secondary" onClick={loadDatasets} title="Refresh dataset list">
            Refresh
          </button>
        </div>

        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center' }}><LoadingSpinner /></div>
        ) : datasets.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <FiDatabase style={{ fontSize: '36px', marginBottom: '12px' }} />
            <p>No datasets uploaded yet. Upload your first CSV or XLSX above!</p>
          </div>
        ) : (
          <div className="netflix-table-wrapper">
            <table className="netflix-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Dataset Name</th>
                  <th>Rows</th>
                  <th>Columns</th>
                  <th>File Size</th>
                  <th>Uploaded</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((ds) => (
                  <tr key={ds.id}>
                    <td style={{ fontWeight: '700', color: 'var(--netflix-red)' }}>#{ds.id}</td>
                    <td>
                      <div style={{ fontWeight: '600' }}>{ds.original_filename || ds.filename}</div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{ds.filename}</div>
                    </td>
                    <td>{ds.rows?.toLocaleString() || '-'}</td>
                    <td>{ds.columns || '-'}</td>
                    <td>{(ds.file_size / 1024).toFixed(1)} KB</td>
                    <td>{ds.uploaded_at ? new Date(ds.uploaded_at).toLocaleDateString() : '-'}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <button
                          className="btn-netflix-secondary"
                          style={{ padding: '6px 12px', fontSize: '12px' }}
                          onClick={() => viewProfile(ds.id)}
                          title="View statistical profile"
                        >
                          <FiEye /> Profile
                        </button>
                        <button
                          className="btn-netflix-secondary"
                          style={{ padding: '6px 12px', fontSize: '12px' }}
                          onClick={() => viewSuitability(ds.id)}
                          title="Check ML task suitability"
                        >
                          <FiCheckCircle /> Suitability
                        </button>
                        <button
                          className="btn-netflix-primary"
                          style={{ padding: '6px 12px', fontSize: '12px' }}
                          onClick={() => navigate(`/dashboard?new=1&dataset_id=${ds.id}`)}
                          title="Create an experiment with this dataset"
                        >
                          <FiPlus /> Experiment
                        </button>
                        <button
                          className="btn-netflix-danger"
                          style={{ padding: '6px 10px' }}
                          onClick={(e) => handleDelete(ds.id, e)}
                          title="Delete dataset"
                        >
                          <FiTrash2 />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Profile Modal */}
      {profileModal && (
        <div className="netflix-modal-backdrop" onClick={() => setProfileModal(null)}>
          <div className="netflix-modal" style={{ maxWidth: '800px' }} onClick={(e) => e.stopPropagation()}>
            <div className="netflix-modal-header">
              <h3 className="netflix-modal-title">Dataset Profile Summary</h3>
              <button className="btn-netflix-ghost" onClick={() => setProfileModal(null)}><FiX /></button>
            </div>
            <div className="netflix-modal-body">
              {profileLoading ? (
                <div style={{ padding: '40px', textAlign: 'center' }}><LoadingSpinner /></div>
              ) : profileModal.error ? (
                <div style={{ color: 'var(--danger)' }}>{profileModal.error}</div>
              ) : (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
                    <div className="netflix-card" style={{ padding: '16px' }}>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Suggested Target</div>
                      <div style={{ fontSize: '18px', fontWeight: '800', color: 'var(--netflix-red)', marginTop: '4px' }}>
                        {profileModal.suggested_target || 'None'}
                      </div>
                    </div>
                    <div className="netflix-card" style={{ padding: '16px' }}>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Suggested Task</div>
                      <div style={{ fontSize: '18px', fontWeight: '800', color: '#ffffff', marginTop: '4px' }}>
                        {profileModal.suggested_problem_type || 'Unknown'}
                      </div>
                    </div>
                    <div className="netflix-card" style={{ padding: '16px' }}>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Missing Values</div>
                      <div style={{ fontSize: '18px', fontWeight: '800', color: '#ffffff', marginTop: '4px' }}>
                        {profileModal.missing_cells_percentage?.toFixed(1) || '0.0'}%
                      </div>
                    </div>
                  </div>

                  <h4 style={{ fontSize: '15px', fontWeight: '700', marginBottom: '12px' }}>Column Breakdown</h4>
                  <div className="netflix-table-wrapper" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                    <table className="netflix-table">
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Inferred Type</th>
                          <th>Unique</th>
                          <th>Missing</th>
                        </tr>
                      </thead>
                      <tbody>
                        {profileModal.column_profiles && Object.entries(profileModal.column_profiles).map(([col, info]) => (
                          <tr key={col}>
                            <td style={{ fontWeight: '600' }}>{col}</td>
                            <td><span className="badge-netflix badge-created">{info.type || 'unknown'}</span></td>
                            <td>{info.unique_count ?? '-'}</td>
                            <td>{info.missing_percentage ? `${info.missing_percentage.toFixed(1)}%` : '0%'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
            <div className="netflix-modal-footer">
              <button className="btn-netflix-secondary" onClick={() => setProfileModal(null)}>Close</button>
            </div>
          </div>
        </div>
      )}

      {/* Suitability Modal */}
      {suitabilityModal && (
        <div className="netflix-modal-backdrop" onClick={() => setSuitabilityModal(null)}>
          <div className="netflix-modal" style={{ maxWidth: '650px' }} onClick={(e) => e.stopPropagation()}>
            <div className="netflix-modal-header">
              <h3 className="netflix-modal-title">ML Task Suitability Assessment</h3>
              <button className="btn-netflix-ghost" onClick={() => setSuitabilityModal(null)}><FiX /></button>
            </div>
            <div className="netflix-modal-body">
              {suitabilityLoading ? (
                <div style={{ padding: '40px', textAlign: 'center' }}><LoadingSpinner /></div>
              ) : suitabilityModal.error ? (
                <div style={{ color: 'var(--danger)' }}>{suitabilityModal.error}</div>
              ) : (
                <div>
                  <div style={{ padding: '14px', borderRadius: 'var(--radius-sm)', marginBottom: '16px', background: suitabilityModal.trainable ? 'var(--success-bg)' : 'var(--danger-bg)', border: `1px solid ${suitabilityModal.trainable ? 'var(--success)' : 'var(--danger)'}` }}>
                    <div style={{ fontWeight: '800', fontSize: '15px', color: suitabilityModal.trainable ? 'var(--success)' : 'var(--danger)' }}>
                      {suitabilityModal.trainable ? 'Trainable for AutoML' : 'Blocking Issues Detected'}
                    </div>
                    <div style={{ fontSize: '13px', marginTop: '4px', color: '#e0e0e0' }}>
                      {suitabilityModal.task_reason || 'Dataset meets structural and target requirements for machine learning.'}
                    </div>
                  </div>

                  {suitabilityModal.warnings && suitabilityModal.warnings.length > 0 && (
                    <div style={{ marginTop: '16px' }}>
                      <h4 style={{ fontSize: '14px', fontWeight: '700', marginBottom: '8px', color: 'var(--warning)' }}>Warnings:</h4>
                      <ul style={{ paddingLeft: '20px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                        {suitabilityModal.warnings.map((w, idx) => (
                          <li key={idx} style={{ marginBottom: '4px' }}>{w}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="netflix-modal-footer">
              <button className="btn-netflix-secondary" onClick={() => setSuitabilityModal(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
