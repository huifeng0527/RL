import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getSessions, getPatients } from '../services/api';

export default function History() {
  const [sessions, setSessions] = useState([]);
  const [patients, setPatients] = useState({});
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [sessionsRes, patientsRes] = await Promise.all([
        getSessions(),
        getPatients()
      ]);
      setSessions(sessionsRes.data);
      const patientMap = {};
      patientsRes.data.forEach(p => { patientMap[p.id] = p; });
      setPatients(patientMap);
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredSessions = sessions.filter(session => {
    if (!filter) return true;
    const patient = patients[session.patient_id];
    if (!patient) return false;
    return patient.name.toLowerCase().includes(filter.toLowerCase());
  });

  const formatDate = (dateStr) => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getScoreColor = (score) => {
    if (score === null) return 'text-gray-400';
    if (score >= 70) return 'text-green-600';
    if (score >= 40) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (loading) {
    return <div className="text-center py-12 text-gray-500">加载中...</div>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800">评估历史</h2>
        <input
          type="text"
          placeholder="搜索患者姓名..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
      </div>

      {filteredSessions.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <p className="text-lg mb-4">暂无评估记录</p>
          <Link to="/patients" className="text-blue-600 hover:text-blue-800">
            去添加患者
          </Link>
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">患者</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">评估时间</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">总分</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">诊断</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {filteredSessions.map((session) => {
                const patient = patients[session.patient_id];
                return (
                  <tr key={session.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4">
                      <Link
                        to={`/patients/${session.patient_id}`}
                        className="font-medium text-gray-900 hover:text-blue-600"
                      >
                        {patient?.name || '未知患者'}
                      </Link>
                    </td>
                    <td className="px-6 py-4 text-gray-600">{formatDate(session.created_at)}</td>
                    <td className="px-6 py-4">
                      {session.total_score !== null ? (
                        <span className={`font-bold text-lg ${getScoreColor(session.total_score)}`}>
                          {session.total_score.toFixed(1)}
                        </span>
                      ) : (
                        <span className="text-gray-400">进行中</span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-gray-600 max-w-xs truncate">
                      {patient?.diagnosis || '-'}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <Link
                        to={`/evaluate/${session.id}`}
                        className="text-blue-600 hover:text-blue-800 font-medium"
                      >
                        查看详情
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
