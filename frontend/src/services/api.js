import axios from 'axios'

// Production API origin.
//
// Set VITE_API_BASE_URL (e.g. https://api.automl-lens.example) when the frontend is
// hosted on a different origin than the FastAPI backend. When it is empty the app
// calls its own origin ('/api'), which is the default used in development and when the
// backend serves the built frontend from the same domain.
//
// Only the public backend URL belongs here - never an API key.
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

/** Absolute (or root-relative) URL for an API path, honouring VITE_API_BASE_URL. */
export const apiUrl = (path) => `${API_BASE}/api${path}`

const API = axios.create({ baseURL: `${API_BASE}/api` })

// Response interceptor for error handling
API.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export const api = {
  // Health
  health: () => API.get('/health'),

  // Datasets
  uploadDataset: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return API.post('/datasets/upload', fd)
  },
  listDatasets: () => API.get('/datasets'),
  getDataset: (id) => API.get(`/datasets/${id}`),
  deleteDataset: (id) => API.delete(`/datasets/${id}`),
  profileDataset: (id) => API.get(`/datasets/${id}/profile`),
  getColumns: (id) => API.get(`/datasets/${id}/columns`),

  // Experiments
  createExperiment: (data) => API.post('/experiments', data),
  listExperiments: () => API.get('/experiments'),
  getExperiment: (id) => API.get(`/experiments/${id}`),
  deleteExperiment: (id) => API.delete(`/experiments/${id}`),
  analyzeExperiment: (id) => API.post(`/experiments/${id}/analyze`),
  trainExperiment: (id, fastDemo = false) => API.post(`/experiments/${id}/train?fast_demo=${fastDemo}`),
  getExperimentStatus: (id) => API.get(`/experiments/${id}/status`),
  getResults: (id) => API.get(`/experiments/${id}/results`),
  getModels: (id) => API.get(`/experiments/${id}/models`),
  getModelsCatalog: () => API.get('/experiments/models/catalog'),
  getExplainability: (id) => API.get(`/experiments/${id}/explainability`),
  predict: (id, features) => API.post(`/experiments/${id}/predict`, features),
  batchPredict: (id, file) => {
    const fd = new FormData()
    fd.append('file', file)
    return API.post(`/experiments/${id}/batch-predict`, fd)
  },
  getReport: (id) => API.get(`/experiments/${id}/report`),
  getInputSchema: (id) => API.get(`/experiments/${id}/input-schema`),
  getBenchmarksSummary: () => API.get('/experiments/benchmarks/summary'),
  downloadBenchmarkReportUrl: () => apiUrl('/experiments/benchmarks/report'),
  // Suitability: blocking issues vs warnings, target distribution, task type.
  getSuitability: (id, targetColumn, problemType) => {
    const params = {}
    if (targetColumn) params.target_column = targetColumn
    if (problemType && problemType !== 'auto') params.problem_type = problemType
    return API.get(`/datasets/${id}/suitability`, { params })
  },
  // Real observed training state (live, or the persisted final snapshot).
  getProgress: (id) => API.get(`/experiments/${id}/progress`),
  downloadModelUrl: (id) => apiUrl(`/experiments/${id}/download-model`),
  downloadMetaUrl: (id) => apiUrl(`/experiments/${id}/download-metadata`),

  // Chat
  chat: (message, experimentId = null, contextMode = 'general', history = []) =>
    API.post('/chat', {
      message,
      experiment_id: experimentId,
      context_mode: contextMode,
      history,
    }),
}

export default API
