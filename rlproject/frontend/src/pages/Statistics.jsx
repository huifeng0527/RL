import { useState, useEffect } from 'react';
import { getPatientStats, getTaskStats, exportExcel } from '../services/api';

export default function Statistics() {
  const [patientStats, setPatientStats] = useState([]);
  const [taskStats, setTaskStats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      getPatientStats().then(r => r.data),
      getTaskStats().then(r => r.data),
    ]).then(([patients, tasks]) => {
      setPatientStats(patients);
      setTaskStats(tasks);
    }).catch(err => {
      if (err.name === 'CanceledError' || err.name === 'AbortError') return;
      console.error('Failed to load stats:', err);
    })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  const handleExport = async () => {
    setExporting(true);
    try {
      const resp = await exportExcel();
      const url = window.URL.createObjectURL(new Blob([resp.data]));
      const a = document.createElement('a');
      a.href = url;
      a.download = `mhecs_report_${Date.now()}.xlsx`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export failed:', err);
      alert('导出失败');
    } finally {
      setExporting(false);
    }
  };

  const taskLabels = { Sprint: '💨', Tracking: '🎯', League: '⚔️', Boundary: '📐' };
  const taskColors = {
    Sprint: 'from-orange-400 to-orange-500',
    Tracking: 'from-blue-400 to-blue-500',
    League: 'from-purple-400 to-purple-500',
    Boundary: 'from-green-400 to-green-500',
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">数据统计</h2>
          <p className="text-slate-500 mt-1">所有患者评估数据的汇总与分析</p>
        </div>
        <button
          onClick={handleExport}
          disabled={exporting}
          className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl font-medium shadow-lg shadow-blue-200 hover:shadow-blue-300 transition-all disabled:opacity-50"
        >
          {exporting ? (
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          )}
          {exporting ? '导出中...' : '导出 Excel'}
        </button>
      </div>

      {/* Task Summary Cards */}
      <div>
        <h3 className="text-lg font-semibold text-slate-700 mb-4">各项目平均分</h3>
        <div className="grid grid-cols-4 gap-4">
          {taskStats.map((task) => {
            const score = task.avg_score ?? '--';
            const colorClass = taskColors[task.task] || 'from-gray-400 to-gray-500';
            return (
              <div key={task.task} className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
                <div className={`h-1.5 bg-gradient-to-r ${colorClass}`} />
                <div className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-2xl">{taskLabels[task.task] || '📊'}</span>
                    <span className="font-semibold text-slate-700">{task.task}</span>
                  </div>
                  <div className="text-4xl font-bold text-slate-800 mb-1">
                    {typeof score === 'number' ? score.toFixed(1) : score}
                    {typeof score === 'number' && <span className="text-lg text-slate-400 font-normal"> / 100</span>}
                  </div>
                  <div className="text-sm text-slate-400">
                    {task.session_count} 条记录
                    {task.min_score != null && (
                      <span className="ml-2">范围: {task.min_score} ~ {task.max_score}</span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Patient Stats Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100">
          <h3 className="text-lg font-semibold text-slate-700">患者平均分</h3>
        </div>
        {patientStats.length === 0 ? (
          <div className="p-12 text-center text-slate-400">
            暂无评估数据
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">患者</th>
                  <th className="px-6 py-3 text-center text-xs font-semibold text-slate-500 uppercase tracking-wider">评估次数</th>
                  <th className="px-6 py-3 text-center text-xs font-semibold text-slate-500 uppercase tracking-wider">总分</th>
                  <th className="px-6 py-3 text-center text-xs font-semibold text-slate-500 uppercase tracking-wider">Sprint</th>
                  <th className="px-6 py-3 text-center text-xs font-semibold text-slate-500 uppercase tracking-wider">Tracking</th>
                  <th className="px-6 py-3 text-center text-xs font-semibold text-slate-500 uppercase tracking-wider">League</th>
                  <th className="px-6 py-3 text-center text-xs font-semibold text-slate-500 uppercase tracking-wider">Boundary</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {patientStats.filter(p => p.session_count > 0).map((patient) => (
                  <tr key={patient.patient_id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-4 font-medium text-slate-800">{patient.patient_name}</td>
                    <td className="px-6 py-4 text-center text-slate-600">{patient.session_count}</td>
                    <td className="px-6 py-4 text-center">
                      {patient.avg_total_score != null ? (
                        <span className="inline-flex items-center justify-center w-14 h-7 bg-blue-100 text-blue-700 rounded-full text-sm font-semibold">
                          {patient.avg_total_score}
                        </span>
                      ) : '--'}
                    </td>
                    <td className="px-6 py-4 text-center text-slate-600">
                      {patient.avg_sprint_score ?? '--'}
                    </td>
                    <td className="px-6 py-4 text-center text-slate-600">
                      {patient.avg_tracking_score ?? '--'}
                    </td>
                    <td className="px-6 py-4 text-center text-slate-600">
                      {patient.avg_league_score ?? '--'}
                    </td>
                    <td className="px-6 py-4 text-center text-slate-600">
                      {patient.avg_boundary_score ?? '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}