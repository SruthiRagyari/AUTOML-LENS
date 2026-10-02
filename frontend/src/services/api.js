import axios from 'axios'

const API = axios.create({ baseURL: '/api' })

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
  getDataset: (id) => API.get(`/datasets/${id}`),
  profileDataset: (id) => API.get(`/datasets/${id}/profile`),
  getColumns: (id) => API.get(`/datasets/${id}/columns`),

  // Experiments
  createExperiment: (data) => API.post('/experiments', data),
  listExperiments: () => API.get('/experiments'),
  getExperiment: (id) => API.get(`/experiments/${id}`),
  analyzeExperiment: (id) => API.post(`/experiments/${id}/analyze`),
  trainExperiment: (id, fastDemo = false) => API.post(`/experiments/${id}/train?fast_demo=${fastDemo}`),
  getExperimentStatus: (id) => API.get(`/experiments/${id}/status`),
  getResults: (id) => API.get(`/experiments/${id}/results`),
  getModels: (id) => API.get(`/experiments/${id}/models`),
  getExplainability: (id) => API.get(`/experiments/${id}/explainability`),
  predict: (id, features) => API.post(`/experiments/${id}/predict`, features),
  batchPredict: (id, file) => {
    const fd = new FormData()
    fd.append('file', file)
    return API.post(`/experiments/${id}/batch-predict`, fd)
  },
  getReport: (id) => API.get(`/experiments/${id}/report`),
  getInputSchema: (id) => API.get(`/experiments/${id}/input-schema`),
// Suitability: blocking issues vs warnings, target distribution, task type.
  getSuitability: (id, targetColumn, problemType) => {
    const params = {}
    if (targetColumn) params.target_column = targetColumn
    if (problemType && problemType !== 'auto') params.problem_type = problemType
    return API.get(`/datasets/${id}/suitability`, { params })
  },
  // Real observed training state (live, or the persisted final snapshot).
  getProgress: (id) => API.get(`/experiments/${id}/progress`),
  downloadModelUrl: (id) => `/api/experiments/${id}/download-model`,
  downloadMetaUrl: (id) => `/api/experiments/${id}/download-metadata`,

  // Chat
  chat: (message, experimentId) => API.post('/chat', { message, experiment_id: experimentId }),
}

export default API
