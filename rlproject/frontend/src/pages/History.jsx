import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getSessions, getPatients, deleteSession, updateSessionNotes } from '../services/api';

export default function History() {
  const [sessions, setSessions] = useState([]);
  const [patients, setPatients] = useState({});
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('');
  const [showNotesModal, setShowNotesModal] = useState(false);
  const [selectedSession, setSelectedSession] = useState(null);
  const [notesText, setNotesText] = useState('');

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

  const handleDelete = async (sessionId) => {
    if (!confirm('确定要删除这次评估记录吗？关联的录像也会一并删除。')) return;
    try {
      await deleteSession(sessionId);
      loadData();
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  const handleOpenNotes = (session) => {
    setSelectedSession(session);
    setNotesText(session.notes || '');
    setShowNotesModal(true);
  };

  const handleSaveNotes = async () => {
    if (!selectedSession) return;
    try {
      await updateSessionNotes(selectedSession.id, notesText);
      setShowNotesModal(false);
      loadData();
    } catch (error) {
      console.error('Failed to update notes:', error);
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
    if (score === null) return 'text-slate-400';
    if (score >= 70) return 'text-emerald-600';
    if (score >= 40) return 'text-amber-600';
    return 'text-red-600';
  };

  const getScoreBg = (score) => {
    if (score === null) return 'bg-slate-100 text-slate-500';
    if (score >= 70) return 'bg-emerald-50 text-emerald-700 border border-emerald-200';
    if (score >= 40) return 'bg-amber-50 text-amber-700 border border-amber-200';
    return 'bg-red-50 text-red-700 border border-red-200';
  };

  if (loading) {
    return (
      <div className="animate-fade-in">
        <div className="card p-6 mb-6">
          <div className="skeleton h-8 w-1/4 mb-4"></div>
          <div className="skeleton h-10 w-full"></div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="card p-6">
              <div className="skeleton h-4 w-1/2 mb-2"></div>
              <div className="skeleton h-3 w-3/4"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in-up">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">评估历史</h2>
          <p className="text-slate-500 mt-1">查看所有患者的康复评估记录</p>
        </div>
        <div className="relative">
          <svg className="w-5 h-5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            placeholder="搜索患者姓名..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="input pl-10 w-64"
          />
        </div>
      </div>

      {filteredSessions.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-slate-700 mb-2">暂无评估记录</h3>
          <p className="text-slate-500 mb-6">开始为患者进行康复评估</p>
          <Link to="/patients" className="btn btn-primary">
            去添加患者
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredSessions.map((session, index) => {
            const patient = patients[session.patient_id];
            return (
              <div
                key={session.id}
                className="card-elevated p-6 group"
                style={{ animationDelay: `${index * 30}ms` }}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center">
                      <svg className="w-6 h-6 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <div>
                      <p className="font-semibold text-slate-800">{patient?.name || '未知患者'}</p>
                      <p className="text-sm text-slate-500">{formatDate(session.created_at)}</p>
                    </div>
                  </div>
                  {session.total_score !== null ? (
                    <div className={`px-3 py-1.5 rounded-xl text-sm font-bold ${getScoreBg(session.total_score)}`}>
                      {session.total_score.toFixed(1)}
                    </div>
                  ) : (
                    <span className="px-3 py-1.5 rounded-xl bg-slate-100 text-slate-500 text-sm">进行中</span>
                  )}
                </div>

                {patient?.diagnosis && (
                  <div className="mb-4">
                    <span className="text-xs text-slate-400 uppercase tracking-wide">诊断</span>
                    <p className="text-sm text-slate-600 truncate">{patient.diagnosis}</p>
                  </div>
                )}

                <div className="pt-4 border-t border-slate-100 flex gap-2">
                  <button
                    onClick={() => handleOpenNotes(session)}
                    className="btn btn-ghost flex-1 justify-center py-2.5 text-slate-600 hover:bg-blue-50 hover:text-blue-600 transition-colors"
                  >
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                    备注
                  </button>
                  <Link
                    to={`/evaluate/${session.id}`}
                    className="btn btn-ghost flex-1 justify-center py-2.5 group-hover:bg-blue-50 group-hover:text-blue-600 transition-colors"
                  >
                    查看详情
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </Link>
                  <button
                    onClick={() => handleDelete(session.id)}
                    className="btn btn-ghost py-2.5 text-red-500 hover:bg-red-50 hover:text-red-600 transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Notes Modal */}
      {showNotesModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-white rounded-2xl p-6 w-full max-w-md mx-4 shadow-2xl">
            <h3 className="text-lg font-bold text-slate-800 mb-4">评估备注</h3>
            <textarea
              value={notesText}
              onChange={(e) => setNotesText(e.target.value)}
              placeholder="添加备注信息..."
              className="w-full h-32 p-3 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
            />
            <div className="flex gap-3 mt-4">
              <button
                onClick={() => setShowNotesModal(false)}
                className="flex-1 btn btn-ghost py-2.5"
              >
                取消
              </button>
              <button
                onClick={handleSaveNotes}
                className="flex-1 btn btn-primary py-2.5"
              >
                保存
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
