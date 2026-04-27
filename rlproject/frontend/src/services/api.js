import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

// Patients
export const getPatients = () => api.get('/patients');
export const getPatient = (id) => api.get(`/patients/${id}`);
export const createPatient = (data) => api.post('/patients', data);
export const updatePatient = (id, data) => api.put(`/patients/${id}`, data);
export const deletePatient = (id) => api.delete(`/patients/${id}`);

// Sessions
export const getSessions = (patientId) => api.get('/sessions', { params: { patient_id: patientId } });
export const getSessionDetail = (id) => api.get(`/sessions/${id}`);
export const createSession = (patientId) => api.post('/sessions', { patient_id: patientId });

// Evaluation
export const startEvaluation = (sessionId) => api.post('/eval/start', null, { params: { session_id: sessionId } });
export const stopEvaluation = () => api.post('/eval/stop');
export const getEvalStatus = () => api.get('/eval/status');

// WebSocket for real-time updates
export const createEvalSocket = (onMessage) => {
  const ws = new WebSocket('ws://localhost:8000/ws/eval');
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };
  ws.onerror = (error) => console.error('WebSocket error:', error);
  return ws;
};

export default api;
