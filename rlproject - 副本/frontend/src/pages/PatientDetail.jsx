import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { getPatient, getSessions, createSession } from '../services/api';

export default function PatientDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, [id]);

  const loadData = async () => {
    try {
      const [patientRes, sessionsRes] = await Promise.all([
        getPatient(id),
        getSessions(id)  // getSessions will return all sessions, filter in component
      ]);
      setPatient(patientRes.data);
      // Filter sessions for this patient
      const filteredSessions = sessionsRes.data.filter(s => s.patient_id === parseInt(id));
      setSessions(filteredSessions);
    } catch (error) {
      console.error('Failed to load patient:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleNewEval = async () => {
    try {
      const res = await createSession(id);
      navigate(`/evaluate/${res.data.id}`);
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleDateString('zh-CN');
  };

  if (loading) {
    return <div className="text-center py-12 text-gray-500">加载中...</div>;
  }

  if (!patient) {
    return <div className="text-center py-12 text-gray-500">患者不存在</div>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link to="/patients" className="text-blue-600 hover:text-blue-800 mb-2 inline-block">
            ← 返回患者列表
          </Link>
          <h2 className="text-2xl font-bold text-gray-800">{patient.name}</h2>
        </div>
        <button
          onClick={handleNewEval}
          className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium text-lg"
        >
          开始新评估
        </button>
      </div>

      {/* Patient Info Card */}
      <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
        <h3 className="text-lg font-bold text-gray-800 mb-4">基本信息</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <span className="text-gray-500 text-sm">性别</span>
            <p className="font-medium text-gray-900">
              {patient.gender === 'M' ? '男' : patient.gender === 'F' ? '女' : '-'}
            </p>
          </div>
          <div>
            <span className="text-gray-500 text-sm">出生日期</span>
            <p className="font-medium text-gray-900">{formatDate(patient.birth_date)}</p>
          </div>
          <div>
            <span className="text-gray-500 text-sm">诊断</span>
            <p className="font-medium text-gray-900">{patient.diagnosis || '-'}</p>
          </div>
          <div>
            <span className="text-gray-500 text-sm">添加时间</span>
            <p className="font-medium text-gray-900">{formatDate(patient.created_at)}</p>
          </div>
        </div>
        {patient.notes && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <span className="text-gray-500 text-sm">备注</span>
            <p className="text-gray-900">{patient.notes}</p>
          </div>
        )}
      </div>

      {/* Sessions History */}
      <div className="bg-white rounded-xl shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-bold text-gray-800">评估历史</h3>
        </div>
        {sessions.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            <p className="mb-4">暂无评估记录</p>
            <button
              onClick={handleNewEval}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              开始第一次评估
            </button>
          </div>
        ) : (
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">评估时间</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">总分</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {sessions.map((session) => (
                <tr key={session.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-gray-900">{formatDate(session.created_at)}</td>
                  <td className="px-6 py-4">
                    {session.total_score !== null ? (
                      <span className={`font-medium ${session.total_score >= 70 ? 'text-green-600' : session.total_score >= 40 ? 'text-yellow-600' : 'text-red-600'}`}>
                        {session.total_score.toFixed(1)}
                      </span>
                    ) : (
                      <span className="text-gray-400">进行中</span>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    <Link
                      to={`/evaluate/${session.id}`}
                      className="text-blue-600 hover:text-blue-800 font-medium"
                    >
                      查看详情
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
