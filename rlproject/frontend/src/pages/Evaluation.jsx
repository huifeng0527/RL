import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { getSessionDetail, startEvaluation, stopEvaluation, getEvalStatus, createEvalSocket } from '../services/api';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';

const TASKS = [
  { id: 'sprint', name: 'Sprint', subtitle: '反应与爆发力', icon: 'M13 10V3L4 14h7v7l9-11h-7z', color: 'blue', gradient: 'from-blue-500 to-blue-600' },
  { id: 'tracking', name: 'Tracking', subtitle: '多轨迹追踪', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z', color: 'emerald', gradient: 'from-emerald-500 to-emerald-600' },
  { id: 'league', name: 'LeagueGame', subtitle: '对抗与安全距离', icon: 'M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z', color: 'orange', gradient: 'from-orange-500 to-orange-600' },
  { id: 'boundary', name: 'Boundary', subtitle: '活动范围与稳定性', icon: 'M4 5a1 1 0 011-1h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5z M4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6z M16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z', color: 'purple', gradient: 'from-purple-500 to-purple-600' },
];

export default function Evaluation() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [evalStatus, setEvalStatus] = useState('idle');
  const [progress, setProgress] = useState({ task: '', progress: 0, message: '' });
  const [currentTaskIndex, setCurrentTaskIndex] = useState(0);
  const [ws, setWs] = useState(null);
  const [results, setResults] = useState(null);
  const [scores, setScores] = useState(null);
  const [error, setError] = useState(null);
  const [frameUrl, setFrameUrl] = useState(null);
  const [celebrating, setCelebrating] = useState(false);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    loadSession();
    return () => {
      if (ws) ws.close();
    };
  }, [sessionId]);

  const loadSession = async () => {
    try {
      const res = await getSessionDetail(sessionId);
      setSession(res.data);
      if (res.data.sprint) {
        setResults(res.data);
        setScores({
          sprint: res.data.sprint_score || 0,
          tracking: res.data.tracking_score || 0,
          league: res.data.league_score || 0,
          boundary: res.data.boundary_score || 0,
        });
      }
    } catch (error) {
      console.error('Failed to load session:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleWebSocketMessage = useCallback((data) => {
    if (data.type === 'progress') {
      setEvalStatus('running');
      setProgress({ task: data.task, progress: data.progress, message: data.message });
      if (data.fps !== undefined) setFps(data.fps);
      const taskIndex = TASKS.findIndex(t => t.name === data.task);
      if (taskIndex >= 0) setCurrentTaskIndex(taskIndex);
    } else if (data.type === 'frame') {
      const blob = new Blob([Uint8Array.from(atob(data.data), c => c.charCodeAt(0))], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      setFrameUrl(url);
    } else if (data.type === 'complete') {
      setEvalStatus('complete');
      setCelebrating(true);
      setTimeout(() => setCelebrating(false), 2000);
      setTimeout(() => loadSession(), 100);
    } else if (data.type === 'error') {
      setError(data.message);
      setEvalStatus('idle');
    }
  }, []);

  const handleStart = async () => {
    setError(null);
    setEvalStatus('countdown');
    const socket = createEvalSocket(handleWebSocketMessage);
    setWs(socket);
    try {
      await startEvaluation(sessionId);
    } catch (error) {
      console.error('Failed to start evaluation:', error);
      setError('启动评估失败');
      setEvalStatus('idle');
      socket.close();
    }
  };

  const handleStop = async () => {
    try {
      await stopEvaluation();
      setEvalStatus('idle');
      if (ws) ws.close();
    } catch (error) {
      console.error('Failed to stop evaluation:', error);
    }
  };

  const getRadarData = () => {
    if (!scores) return [];
    return TASKS.map((t, i) => ({
      subject: t.name,
      score: scores[t.id] || 0,
      fullMark: 100
    }));
  };

  const getTotalScore = () => {
    return session?.total_score?.toFixed(1) || '0.0';
  };

  if (loading) {
    return (
      <div className="animate-fade-in">
        <div className="card p-6">
          <div className="skeleton h-8 w-1/4 mb-4"></div>
          <div className="skeleton h-4 w-3/4"></div>
        </div>
      </div>
    );
  }

  if (!session) {
    return <div className="text-center py-12 text-slate-500">评估会话不存在</div>;
  }

  return (
    <div className="animate-fade-in-up">
      <div className="flex items-center justify-between mb-8">
        <div>
          <Link to={`/patients/${session.patient_id}`} className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 mb-2 transition-colors">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            返回患者详情
          </Link>
          <h2 className="text-2xl font-bold text-slate-800">评估会话 #{session.id}</h2>
        </div>

        {evalStatus === 'idle' && !results && (
          <button onClick={handleStart} className="btn btn-success">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            开始评估
          </button>
        )}

        {evalStatus === 'running' && (
          <button onClick={handleStop} className="btn btn-danger">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
            </svg>
            停止评估
          </button>
        )}
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-2xl mb-6 flex items-center gap-3 animate-fade-in">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {error}
        </div>
      )}

      {/* Progress Section */}
      {(evalStatus === 'countdown' || evalStatus === 'running') && (
        <div className="card-elevated p-6 mb-6 animate-fade-in">
          <h3 className="text-lg font-bold text-slate-800 mb-6">评估进度</h3>
          <div className="space-y-4">
            {TASKS.map((task, idx) => {
              const isCompleted = idx < currentTaskIndex;
              const isActive = idx === currentTaskIndex && evalStatus === 'running';
              const isPending = idx > currentTaskIndex;

              return (
                <div key={task.id} className="flex items-center gap-4">
                  <div className={`relative w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-500 ${
                    isCompleted ? 'bg-emerald-500 text-white' :
                    isActive ? `bg-gradient-to-br ${task.gradient} text-white shadow-lg shadow-${task.color}-200` :
                    'bg-slate-100 text-slate-400'
                  }`}>
                    {isCompleted ? (
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={task.icon} />
                      </svg>
                    )}
                    {isActive && (
                      <span className="absolute inset-0 rounded-2xl ring-4 ring-offset-2 ring-blue-400 animate-pulse-soft" />
                    )}
                  </div>

                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className={`font-semibold ${isPending ? 'text-slate-400' : 'text-slate-800'}`}>{task.name}</span>
                      <span className={`text-sm ${isPending ? 'text-slate-300' : 'text-slate-500'}`}>({task.subtitle})</span>
                    </div>
                    {isActive && (
                      <div className="mt-2 space-y-2 animate-fade-in">
                        <div className="progress-bar">
                          <div
                            className={`progress-fill bg-gradient-to-r ${task.gradient}`}
                            style={{ width: `${progress.progress * 100}%` }}
                          />
                        </div>
                        <p className="text-sm text-slate-600">{progress.message}</p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Results Section */}
      {(results || evalStatus === 'complete') && scores && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Radar Chart */}
          <div className="card-elevated p-6 animate-fade-in-up">
            <h3 className="text-lg font-bold text-slate-800 mb-4">评估雷达图</h3>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={getRadarData()}>
                <PolarGrid stroke="#e2e8f0" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 12 }} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                <Radar name="得分" dataKey="score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
            <div className="text-center mt-4">
              <div className="inline-flex flex-col items-center p-4 bg-slate-50 rounded-2xl">
                <span className={`text-4xl font-bold ${celebrating ? 'animate-celebrate' : ''} text-blue-600`}>
                  {getTotalScore()}
                </span>
                <span className="text-slate-500 text-sm">/ 100 总分</span>
              </div>
            </div>
          </div>

          {/* Task Scores */}
          <div className="card-elevated p-6 animate-fade-in-up" style={{ animationDelay: '100ms' }}>
            <h3 className="text-lg font-bold text-slate-800 mb-4">各维度得分</h3>
            <div className="space-y-5">
              {TASKS.map((task) => (
                <div key={task.id}>
                  <div className="flex justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className={`font-semibold text-slate-700`}>{task.name}</span>
                      <span className="text-xs text-slate-400">({task.subtitle})</span>
                    </div>
                    <span className="font-bold text-slate-800">{(scores[task.id] || 0).toFixed(1)}</span>
                  </div>
                  <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full bg-gradient-to-r ${task.gradient} transition-all duration-700 ease-out`}
                      style={{ width: `${scores[task.id] || 0}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Sprint Details */}
          {results?.sprint && (
            <div className="card-elevated p-6 animate-fade-in-up" style={{ animationDelay: '150ms' }}>
              <h3 className="text-lg font-bold text-slate-800 mb-4">Sprint 详情</h3>
              <div className="grid grid-cols-5 gap-3">
                {results.sprint.catch_times.map((time, i) => (
                  <div key={i} className="bg-blue-50 rounded-xl p-3 text-center border border-blue-100">
                    <p className="text-xs text-blue-500 mb-1">第{i + 1}次</p>
                    <p className="text-lg font-bold text-blue-600">{time.toFixed(2)}s</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tracking Details */}
          {results?.tracking && (
            <div className="card-elevated p-6 animate-fade-in-up" style={{ animationDelay: '200ms' }}>
              <h3 className="text-lg font-bold text-slate-800 mb-4">Tracking 详情</h3>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={results.tracking.rmse_list.map((v, i) => ({ t: i + 1, rmse: v }))}>
                  <XAxis dataKey="t" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} />
                  <Tooltip
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}
                  />
                  <Line type="monotone" dataKey="rmse" stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
              <div className="mt-3 text-center p-3 bg-emerald-50 rounded-xl">
                <p className="text-sm text-emerald-600">平均 RMSE: <span className="font-bold">{(results.tracking.rmse_list.reduce((a, b) => a + b, 0) / results.tracking.rmse_list.length).toFixed(3)}</span></p>
              </div>
            </div>
          )}

          {/* League Details */}
          {results?.league && (
            <div className="card-elevated p-6 animate-fade-in-up" style={{ animationDelay: '250ms' }}>
              <h3 className="text-lg font-bold text-slate-800 mb-4">LeagueGame 详情</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className={`rounded-xl p-4 ${results.league.is_caught ? 'bg-red-50 border border-red-100' : 'bg-emerald-50 border border-emerald-100'}`}>
                  <p className="text-xs text-slate-500 mb-2">结果</p>
                  <p className={`text-xl font-bold ${results.league.is_caught ? 'text-red-600' : 'text-emerald-600'}`}>
                    {results.league.is_caught ? '被抓到' : '成功躲避'}
                  </p>
                </div>
                <div className="bg-orange-50 rounded-xl p-4 border border-orange-100">
                  <p className="text-xs text-slate-500 mb-2">生存时间</p>
                  <p className="text-xl font-bold text-orange-600">{results.league.survival_time.toFixed(1)}s</p>
                </div>
              </div>
              {results.league.dist_list && results.league.dist_list.length > 0 && (
                <div className="mt-4 p-4 bg-slate-50 rounded-xl">
                  <p className="text-xs text-slate-500 mb-2">距离变化</p>
                  <ResponsiveContainer width="100%" height={100}>
                    <LineChart data={results.league.dist_list.map((v, i) => ({ t: i + 1, d: v }))}>
                      <XAxis dataKey="t" hide />
                      <YAxis stroke="#94a3b8" fontSize={10} width={30} />
                      <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 2px 10px rgba(0,0,0,0.1)' }} />
                      <Line type="monotone" dataKey="d" stroke="#f97316" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}

          {/* Boundary Details */}
          {results?.boundary && (
            <div className="card-elevated p-6 animate-fade-in-up" style={{ animationDelay: '300ms' }}>
              <h3 className="text-lg font-bold text-slate-800 mb-4">Boundary 详情</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-purple-50 rounded-xl p-4 border border-purple-100">
                  <p className="text-xs text-slate-500 mb-2">X 轴范围</p>
                  <p className="text-lg font-bold text-purple-600">
                    {results.boundary.min_x.toFixed(1)} ~ {results.boundary.max_x.toFixed(1)}
                  </p>
                </div>
                <div className="bg-purple-50 rounded-xl p-4 border border-purple-100">
                  <p className="text-xs text-slate-500 mb-2">Y 轴范围</p>
                  <p className="text-lg font-bold text-purple-600">
                    {results.boundary.min_y.toFixed(1)} ~ {results.boundary.max_y.toFixed(1)}
                  </p>
                </div>
              </div>
              {results.boundary.vel_list && results.boundary.vel_list.length > 0 && (
                <div className="mt-4 p-4 bg-slate-50 rounded-xl">
                  <p className="text-xs text-slate-500 mb-2">速度曲线</p>
                  <ResponsiveContainer width="100%" height={100}>
                    <LineChart data={results.boundary.vel_list.map((v, i) => ({ t: i + 1, v: v }))}>
                      <XAxis dataKey="t" hide />
                      <YAxis stroke="#94a3b8" fontSize={10} width={30} />
                      <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 2px 10px rgba(0,0,0,0.1)' }} />
                      <Line type="monotone" dataKey="v" stroke="#a855f7" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Idle State */}
      {evalStatus === 'idle' && !results && (
        <div className="card-elevated p-12 text-center animate-fade-in">
          <div className={`w-24 h-24 mx-auto rounded-full bg-gradient-to-br from-blue-100 to-blue-200 flex items-center justify-center mb-6 ${celebrating ? 'animate-celebrate' : 'animate-float'}`}>
            <svg className="w-12 h-12 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <h3 className="text-2xl font-bold text-slate-800 mb-3">准备开始评估</h3>
          <p className="text-slate-500 mb-8 max-w-md mx-auto">
            将依次进行 4 个评估任务：Sprint、Tracking、LeagueGame、Boundary，每个任务将全面评估患者的康复情况
          </p>
          <button
            onClick={handleStart}
            className="btn btn-success text-lg px-8 py-4"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            开始评估
          </button>
        </div>
      )}

      {/* Video Feed Section */}
      {(evalStatus === 'countdown' || evalStatus === 'running') && (
        <div className="card-elevated p-6 mb-6 animate-fade-in">
          <h3 className="text-lg font-bold text-slate-800 mb-4">实时画面</h3>
          <div className="video-container" style={{ maxWidth: '640px' }}>
            {frameUrl ? (
              <img
                src={frameUrl}
                alt="Camera feed"
                className="w-full h-auto"
                style={{ maxHeight: '480px', objectFit: 'contain' }}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-slate-400">
                <svg className="w-12 h-12 mb-3 animate-pulse-soft" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                等待画面...
              </div>
            )}
            <div className="video-overlay pointer-events-none" />
            <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-start">
              <div>
                <p className="text-white font-bold text-lg drop-shadow-lg">{progress.task || '准备中...'}</p>
                <p className="text-white/80 text-sm drop-shadow-lg">{progress.message}</p>
              </div>
              <div className="bg-black/50 px-3 py-1 rounded-lg">
                <p className="text-white font-mono text-sm">FPS: {fps.toFixed(1)}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
