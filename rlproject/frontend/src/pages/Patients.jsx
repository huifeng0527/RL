import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { getPatients, createPatient, deletePatient } from '../services/api';

export default function Patients() {
  const [patients, setPatients] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [toast, setToast] = useState(null);
  const toastTimerRef = useRef(null);
  const [formData, setFormData] = useState({
    name: '',
    gender: '',
    birth_date: '',
    diagnosis: '',
    notes: ''
  });

  useEffect(() => {
    loadPatients();
    return () => {
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    };
  }, []);

  const showToast = (message, type = 'info') => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    setToast({ message, type });
    toastTimerRef.current = setTimeout(() => setToast(null), 3000);
  };

  const loadPatients = async () => {
    try {
      const res = await getPatients();
      setPatients(res.data);
    } catch (error) {
      console.error('Failed to load patients:', error);
      showToast('加载患者列表失败', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await createPatient(formData);
      setShowModal(false);
      setFormData({ name: '', gender: '', birth_date: '', diagnosis: '', notes: '' });
      loadPatients();
      showToast('患者添加成功', 'success');
    } catch (error) {
      console.error('Failed to create patient:', error);
      showToast('添加患者失败', 'error');
    }
  };

  const handleDelete = async (id) => {
    if (!confirm('确定要删除该患者吗？所有相关评估记录也将被删除。')) return;
    try {
      await deletePatient(id);
      loadPatients();
      showToast('患者已删除', 'success');
    } catch (error) {
      console.error('Failed to delete patient:', error);
      showToast('删除患者失败', 'error');
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleDateString('zh-CN');
  };

  return (
    <div className="animate-fade-in-up">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">患者管理</h2>
          <p className="text-slate-500 mt-1">管理所有患者的评估记录</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="btn btn-primary"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          新增患者
        </button>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="card p-6">
              <div className="skeleton h-4 w-1/3 mb-4"></div>
              <div className="skeleton h-3 w-2/3 mb-2"></div>
              <div className="skeleton h-3 w-1/2"></div>
            </div>
          ))}
        </div>
      ) : patients.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-slate-700 mb-2">暂无患者记录</h3>
          <p className="text-slate-500 mb-6">添加第一位患者开始康复评估之旅</p>
          <button
            onClick={() => setShowModal(true)}
            className="btn btn-primary"
          >
            添加患者
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {patients.map((patient, index) => (
            <div
              key={patient.id}
              className="card-elevated p-6 group"
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center text-white text-xl font-bold shadow-lg shadow-blue-200">
                  {patient.name.charAt(0).toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg font-semibold text-slate-800 truncate">{patient.name}</h3>
                  <p className="text-sm text-slate-500">
                    {patient.gender === 'M' ? '男' : patient.gender === 'F' ? '女' : '未设置'} · {patient.diagnosis || '无诊断'}
                  </p>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-wide">出生日期</p>
                  <p className="text-sm font-medium text-slate-700">{formatDate(patient.birth_date)}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-wide">添加时间</p>
                  <p className="text-sm font-medium text-slate-700">{formatDate(patient.created_at)}</p>
                </div>
              </div>

              <div className="mt-4 flex items-center gap-2">
                <Link
                  to={`/patients/${patient.id}`}
                  className="flex-1 btn btn-primary py-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  查看详情
                </Link>
                <button
                  onClick={() => handleDelete(patient.id)}
                  className="btn btn-ghost text-slate-400 hover:text-red-500 p-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Add Patient Modal */}
      {showModal && (
        <>
          <div className="modal-backdrop" onClick={() => setShowModal(false)} />
          <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
            <div className="modal-content w-full max-w-md p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center">
                    <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-bold text-slate-800">新增患者</h3>
                </div>
                <button
                  onClick={() => setShowModal(false)}
                  className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
                >
                  <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <form onSubmit={handleSubmit} className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5">姓名 *</label>
                  <input
                    type="text"
                    required
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="input"
                    placeholder="请输入患者姓名"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1.5">性别</label>
                    <select
                      value={formData.gender}
                      onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
                      className="input"
                    >
                      <option value="">请选择</option>
                      <option value="M">男</option>
                      <option value="F">女</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1.5">出生日期</label>
                    <input
                      type="date"
                      value={formData.birth_date}
                      onChange={(e) => setFormData({ ...formData, birth_date: e.target.value })}
                      className="input"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5">诊断</label>
                  <textarea
                    value={formData.diagnosis}
                    onChange={(e) => setFormData({ ...formData, diagnosis: e.target.value })}
                    rows={2}
                    className="input resize-none"
                    placeholder="请输入诊断信息"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5">备注</label>
                  <textarea
                    value={formData.notes}
                    onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                    rows={2}
                    className="input resize-none"
                    placeholder="其他备注信息"
                  />
                </div>

                <div className="flex justify-end gap-3 pt-2">
                  <button
                    type="button"
                    onClick={() => setShowModal(false)}
                    className="btn btn-ghost"
                  >
                    取消
                  </button>
                  <button
                    type="submit"
                    className="btn btn-primary"
                  >
                    保存
                  </button>
                </div>
              </form>
            </div>
          </div>
        </>
      )}

      {/* Toast Notification */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.type === 'success' && (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )}
          {toast.type === 'error' && (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          )}
          {toast.message}
        </div>
      )}
    </div>
  );
}
