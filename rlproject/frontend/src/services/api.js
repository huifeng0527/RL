import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
const WS_BASE = import.meta.env.VITE_WS_BASE_URL || API_BASE.replace(/^http/, 'ws').replace(/\/api\/?$/, '');

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

const withParams = (params, config = {}) => ({
  ...config,
  params: { ...params, ...config.params },
});

// Patients
export const getPatients = (config = {}) => api.get('/patients', config);
export const getPatient = (id, config = {}) => api.get(`/patients/${id}`, config);
export const createPatient = (data, config = {}) => api.post('/patients', data, config);
export const updatePatient = (id, data, config = {}) => api.put(`/patients/${id}`, data, config);
export const deletePatient = (id, config = {}) => api.delete(`/patients/${id}`, config);

// Sessions
export const getSessions = (patientId, config = {}) => {
  const params = patientId ? { patient_id: patientId } : {};
  return api.get('/sessions', withParams(params, config));
};
export const getSessionDetail = (id, config = {}) => api.get(`/sessions/${id}`, config);
export const createSession = (patientId, config = {}) => api.post('/sessions', { patient_id: patientId }, config);
export const deleteSession = (id, config = {}) => api.delete(`/sessions/${id}`, config);
export const updateSessionNotes = (id, notes, config = {}) => api.patch(`/sessions/${id}/notes`, { notes }, config);

// Evaluation
export const startEvaluation = (sessionId, config = {}) => api.post('/eval/start', null, withParams({ session_id: sessionId }, config));
export const stopEvaluation = (config = {}) => api.post('/eval/stop', null, config);
export const getEvalStatus = (config = {}) => api.get('/eval/status', config);

// Statistics
export const getPatientStats = (config = {}) => api.get('/stats/patients', config);
export const getTaskStats = (config = {}) => api.get('/stats/tasks', config);
export const getSinglePatientStats = (id, config = {}) => api.get(`/stats/patient/${id}`, config);
export const exportExcel = (config = {}) => api.get('/export/excel', { responseType: 'blob', ...config });

// WebSocket for real-time updates
export const createEvalSocket = ({ onMessage, onOpen, onClose, onError } = {}) => {
  const ws = new WebSocket(`${WS_BASE}/ws/eval`);
  ws.onopen = onOpen || null;
  ws.onmessage = (event) => {
    try {
      onMessage?.(JSON.parse(event.data));
    } catch (error) {
      console.error('Invalid WebSocket message:', error);
    }
  };
  ws.onclose = onClose || null;
  ws.onerror = onError || ((error) => console.error('WebSocket error:', error));
  return ws;
};

export default api;
