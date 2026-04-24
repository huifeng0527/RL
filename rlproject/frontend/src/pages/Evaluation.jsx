import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { getSessionDetail, startEvaluation, stopEvaluation, getEvalStatus, createEvalSocket } from '../services/api';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';

const TASKS = [
  { id: 'sprint', name: 'Sprint', subtitle: '反应与爆发力', description: '5次快速捕捉目标', color: 'blue' },
  { id: 'tracking', name: 'Tracking', subtitle: '多轨迹追踪', description: '追踪移动目标', color: 'green' },
  { id: 'league', name: 'LeagueGame', subtitle: '对抗与安全距离', description: '躲避机器人攻击', color: 'orange' },
  { id: 'boundary', name: 'Boundary', subtitle: '活动范围与稳定性', description: '沿边界追踪目标', color: 'purple' },
];

export default function Evaluation() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [evalStatus, setEvalStatus] = useState('idle'); // idle, countdown, running, complete
  const [progress, setProgress] = useState({ task: '', progress: 0, message: '' });
  const [currentTaskIndex, setCurrentTaskIndex] = useState(0);
  const [ws, setWs] = useState(null);
  const [results, setResults] = useState(null);
  const [scores, setScores] = useState(null);
  const [error, setError] = useState(null);
  const [frameUrl, setFrameUrl] = useState(null);
  const canvasRef = useRef(null);

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
        calculateScores(res.data);
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
      const taskIndex = TASKS.findIndex(t => t.name === data.task);
      if (taskIndex >= 0) setCurrentTaskIndex(taskIndex);
    } else if (data.type === 'frame') {
      // Update video frame
      const blob = new Blob([Uint8Array.from(atob(data.data), c => c.charCodeAt(0))], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      setFrameUrl(url);
    } else if (data.type === 'complete') {
      setEvalStatus('complete');
      loadSession(); // Reload to get full results
    } else if (data.type === 'error') {
      setError(data.message);
      setEvalStatus('idle');
    }
  }, []);

  const handleStart = async () => {
    setError(null);
    setEvalStatus('countdown');

    // Connect WebSocket for real-time updates
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

  const calculateScores = (data) => {
    const s = {};
    if (data.sprint?.catch_times) {
      const avg = data.sprint.catch_times.reduce((a, b) => a + b, 0) / data.sprint.catch_times.length;
      s.sprint = Math.max(0, Math.min(100, (6 - avg) / 5 * 100));
    }
    if (data.tracking?.rmse_list) {
      const avg = data.tracking.rmse_list.reduce((a, b) => a + b, 0) / data.tracking.rmse_list.length;
      s.tracking = Math.max(0, Math.min(100, (5 - avg) / 4.5 * 100));
    }
    if (data.league) {
      s.league = data.league.is_caught ? Math.max(0, data.league.survival_time / 30 * 100) : 100;
    }
    if (data.boundary) {
      const range = (data.boundary.max_x - data.boundary.min_x) + (data.boundary.max_y - data.boundary.min_y);
      s.boundary = Math.max(0, Math.min(100, (range - 5) / 15 * 100));
    }
    setScores(s);
  };

  const getRadarData = () => {
    if (!scores) return [];
    return TASKS.map((t, i) => ({
      subject: t.name,
      score: scores[t.id] || 0,
      fullMark: 100
    }));
  };

  if (loading) {
    return <div className="text-center py-12 text-gray-500">加载中...</div>;
  }

  if (!session) {
    return <div className="text-center py-12 text-gray-500">评估会话不存在</div>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link to={`/patients/${session.patient_id}`} className="text-blue-600 hover:text-blue-800 mb-2 inline-block">
            ← 返回患者详情
          </Link>
          <h2 className="text-2xl font-bold text-gray-800">评估会话 #{session.id}</h2>
        </div>
        {evalStatus === 'idle' && !results && (
          <button
            onClick={handleStart}
            className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium text-lg"
          >
            开始评估
          </button>
        )}
        {evalStatus === 'running' && (
          <button
            onClick={handleStop}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium text-lg"
          >
            停止评估
          </button>
        )}
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg mb-6">
          {error}
        </div>
      )}

      {/* Progress Section */}
      {(evalStatus === 'countdown' || evalStatus === 'running') && (
        <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
          <h3 className="text-lg font-bold text-gray-800 mb-4">评估进度</h3>
          <div className="space-y-4">
            {TASKS.map((task, idx) => (
              <div key={task.id} className="flex items-center gap-4">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                  idx < currentTaskIndex ? 'bg-green-500 text-white' :
                  idx === currentTaskIndex ? `bg-${task.color}-500 text-white` :
                  'bg-gray-200 text-gray-500'
                }`}>
                  {idx < currentTaskIndex ? '✓' : idx + 1}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-800">{task.name}</span>
                    <span className="text-gray-500 text-sm">({task.subtitle})</span>
                  </div>
                  {idx === currentTaskIndex && evalStatus === 'running' && (
                    <div className="mt-1">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`bg-${task.color}-500 h-2 rounded-full transition-all`}
                          style={{ width: `${progress.progress * 100}%` }}
                        />
                      </div>
                      <p className="text-sm text-gray-600 mt-1">{progress.message}</p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Results Section */}
      {(results || evalStatus === 'complete') && scores && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Radar Chart */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4">评估雷达图</h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={getRadarData()}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar name="得分" dataKey="score" stroke="#2196F3" fill="#2196F3" fillOpacity={0.3} />
              </RadarChart>
            </ResponsiveContainer>
            <div className="text-center mt-4">
              <span className="text-3xl font-bold text-blue-600">
                {((scores.sprint * 0.2 + scores.tracking * 0.3 + scores.league * 0.3 + scores.boundary * 0.2)).toFixed(1)}
              </span>
              <span className="text-gray-500">/ 100 总分</span>
            </div>
          </div>

          {/* Task Scores */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4">各维度得分</h3>
            <div className="space-y-4">
              {TASKS.map((task) => (
                <div key={task.id}>
                  <div className="flex justify-between mb-1">
                    <span className="font-medium text-gray-700">{task.name} ({task.subtitle})</span>
                    <span className="font-bold text-gray-800">{(scores[task.id] || 0).toFixed(1)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className={`bg-${task.color}-500 h-3 rounded-full`}
                      style={{ width: `${scores[task.id] || 0}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Sprint Details */}
          {results?.sprint && (
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Sprint 详情</h3>
              <div className="grid grid-cols-2 gap-4">
                {results.sprint.catch_times.map((time, i) => (
                  <div key={i} className="bg-blue-50 rounded-lg p-3">
                    <p className="text-sm text-gray-600">第{i + 1}次</p>
                    <p className="text-xl font-bold text-blue-600">{time.toFixed(2)}s</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tracking Details */}
          {results?.tracking && (
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Tracking 详情</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={results.tracking.rmse_list.map((v, i) => ({ t: i, rmse: v }))}>
                  <XAxis dataKey="t" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="rmse" stroke="#4CAF50" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-center text-gray-600 mt-2">
                平均 RMSE: {(results.tracking.rmse_list.reduce((a, b) => a + b, 0) / results.tracking.rmse_list.length).toFixed(3)}
              </p>
            </div>
          )}

          {/* League Details */}
          {results?.league && (
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">LeagueGame 详情</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-orange-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">结果</p>
                  <p className={`text-xl font-bold ${results.league.is_caught ? 'text-red-600' : 'text-green-600'}`}>
                    {results.league.is_caught ? '被抓到' : '成功躲避'}
                  </p>
                </div>
                <div className="bg-orange-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">生存时间</p>
                  <p className="text-xl font-bold text-orange-600">{results.league.survival_time.toFixed(1)}s</p>
                </div>
              </div>
            </div>
          )}

          {/* Boundary Details */}
          {results?.boundary && (
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Boundary 详情</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-purple-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">X 范围</p>
                  <p className="text-lg font-bold text-purple-600">
                    {results.boundary.min_x.toFixed(1)} - {results.boundary.max_x.toFixed(1)}
                  </p>
                </div>
                <div className="bg-purple-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">Y 范围</p>
                  <p className="text-lg font-bold text-purple-600">
                    {results.boundary.min_y.toFixed(1)} - {results.boundary.max_y.toFixed(1)}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Idle State */}
      {evalStatus === 'idle' && !results && (
        <div className="bg-white rounded-xl shadow-sm p-12 text-center">
          <div className="text-6xl mb-4">🎯</div>
          <h3 className="text-xl font-bold text-gray-800 mb-2">准备开始评估</h3>
          <p className="text-gray-500 mb-6">
            将依次进行 4 个评估任务：Sprint、Tracking、LeagueGame、Boundary
          </p>
          <button
            onClick={handleStart}
            className="px-8 py-4 bg-green-600 text-white rounded-xl hover:bg-green-700 font-medium text-lg"
          >
            开始评估
          </button>
        </div>
      )}

      {/* Video Feed Section */}
      {(evalStatus === 'countdown' || evalStatus === 'running') && (
        <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
          <h3 className="text-lg font-bold text-gray-800 mb-4">实时画面</h3>
          <div className="relative bg-black rounded-lg overflow-hidden" style={{ maxWidth: '640px' }}>
            {frameUrl ? (
              <img
                src={frameUrl}
                alt="Camera feed"
                className="w-full h-auto"
                style={{ maxHeight: '480px', objectFit: 'contain' }}
              />
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-400">
                等待画面...
              </div>
            )}
            {/* Task Info Overlay */}
            <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/70 to-transparent p-4">
              <p className="text-white font-bold text-lg">{progress.task || '准备中...'}</p>
              <p className="text-white/80 text-sm">{progress.message}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
