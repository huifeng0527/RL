import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Patients from './pages/Patients';
import PatientDetail from './pages/PatientDetail';
import Evaluation from './pages/Evaluation';
import History from './pages/History';
import Layout from './components/Layout';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/patients" replace />} />
          <Route path="patients" element={<Patients />} />
          <Route path="patients/:id" element={<PatientDetail />} />
          <Route path="evaluate/:sessionId" element={<Evaluation />} />
          <Route path="history" element={<History />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
