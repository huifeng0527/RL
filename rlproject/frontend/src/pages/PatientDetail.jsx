import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { getPatient, getSessions, createSession } from '../services/api';

export default function PatientDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const abortRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    abortRef.current = controller;
    loadData(controller.signal);
    return () => {
      controller.abort();
    };
  }, [id]);

  const loadData = async (signal) => {
    try {
      const [patientRes, sessionsRes] = await Promise.all([
        getPatient(id, { signal }),
        getSessions(id, { signal })
      ]);
      setPatient(patientRes.data);
      const filteredSessions = sessionsRes.data.filter(s => s.patient_id === parseInt(id));
      setSessions(filteredSessions);
    } catch (error) {
      if (error.name === 'CanceledError' || error.name === 'AbortError') return;
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

  const getScoreColor = (score) => {
    if (score === null) return 'text-slate-400';
    if (score >= 70) return 'text-emerald-600';
    if (score >= 40) return 'text-amber-600';
    return 'text-red-600';
  };

  const getScoreBg = (score) => {
    if (score === null) return 'bg-slate-100';
    if (score >= 70) return 'bg-emerald-50 border-emerald-200';
    if (score >= 40) return 'bg-amber-50 border-amber-200';
    return 'bg-red-50 border-red-200';
  };

  if (loading) {
    return (
      <div className="animate-fade-in">
        <div className="card p-6 mb-6">
          <div className="skeleton h-8 w-1/4 mb-4"></div>
          <div className="skeleton h-4 w-3/4"></div>
        </div>
      </div>
    );
  }

  if (!patient) {
    return <div className="text-center py-12 text-slate-500">患者不存在</div>;
  }

  return (
    <div className="animate-fade-in-up">
      <div className="flex items-center justify-between mb-8">
        <div>
          <Link to="/patients" className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 mb-2 transition-colors">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            返回患者列表
          </Link>
          <h2 className="text-2xl font-bold text-slate-800">{patient.name}</h2>
        </div>
        <button
          onClick={handleNewEval}
          className="btn btn-success"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          开始新评估
        </button>
      </div>

      {/* Patient Info Card */}
      <div className="card-elevated p-6 mb-6">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center text-white text-2xl font-bold shadow-lg shadow-blue-200">
            {patient.name.charAt(0).toUpperCase()}
          </div>
          <div>
            <h3 className="text-xl font-bold text-slate-800">{patient.name}</h3>
            <p className="text-slate-500">{patient.diagnosis || '暂无诊断'}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div className="p-4 bg-slate-50 rounded-xl">
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">性别</p>
            <p className="font-semibold text-slate-700">
              {patient.gender === 'M' ? '男' : patient.gender === 'F' ? '女' : '-'}
            </p>
          </div>
          <div className="p-4 bg-slate-50 rounded-xl">
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">出生日期</p>
            <p className="font-semibold text-slate-700">{formatDate(patient.birth_date)}</p>
          </div>
          <div className="p-4 bg-slate-50 rounded-xl">
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">评估次数</p>
            <p className="font-semibold text-slate-700">{sessions.length} 次</p>
          </div>
          <div className="p-4 bg-slate-50 rounded-xl">
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">最近评估</p>
            <p className="font-semibold text-slate-700">
              {sessions.length > 0 ? formatDate(sessions[0].created_at) : '-'}
            </p>
          </div>
        </div>

        {patient.notes && (
          <div className="mt-6 pt-6 border-t border-slate-100">
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-2">备注</p>
            <p className="text-slate-700">{patient.notes}</p>
          </div>
        )}
      </div>

      {/* Sessions History */}
      <div className="card-elevated overflow-hidden">
        <div className="px-6 py-5 border-b border-slate-100 bg-gradient-to-r from-slate-50 to-white">
          <h3 className="text-lg font-bold text-slate-800">评估历史</h3>
        </div>

        {sessions.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-slate-700 mb-2">暂无评估记录</h3>
            <p className="text-slate-500 mb-6">为该患者创建第一次康复评估</p>
            <button
              onClick={handleNewEval}
              className="btn btn-primary"
            >
              开始评估
            </button>
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {sessions.map((session, index) => (
              <div
                key={session.id}
                className="p-6 hover:bg-slate-50/50 transition-colors group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center text-slate-600 group-hover:bg-blue-100 group-hover:text-blue-600 transition-colors">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium text-slate-700">评估 #{session.id}</p>
                      <p className="text-sm text-slate-500">{formatDate(session.created_at)}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    {session.total_score !== null ? (
                      <div className={`px-4 py-2 rounded-xl border ${getScoreBg(session.total_score)}`}>
                        <span className={`text-xl font-bold ${getScoreColor(session.total_score)}`}>
                          {session.total_score.toFixed(1)}
                        </span>
                        <span className="text-xs text-slate-500 ml-1">/ 100</span>
                      </div>
                    ) : (
                      <span className="px-4 py-2 rounded-xl bg-slate-100 text-slate-500 text-sm">进行中</span>
                    )}

                    <Link
                      to={`/evaluate/${session.id}`}
                      className="btn btn-ghost py-2 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      查看详情
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
