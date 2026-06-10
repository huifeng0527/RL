export const EVALUATION_TASKS = [
  {
    id: 'rapid_reach',
    name: 'Rapid Reach',
    shortName: 'Reach',
    subtitle: '反应、启动、快速到达',
    icon: 'M13 10V3L4 14h7v7l9-11h-7z',
    color: 'blue',
    gradient: 'from-blue-500 to-blue-600',
    shadow: 'shadow-blue-200',
  },
  {
    id: 'continuous_tracking',
    name: 'Continuous Tracking',
    shortName: 'Tracking',
    subtitle: '连续协调、平滑控制',
    icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10',
    color: 'emerald',
    gradient: 'from-emerald-500 to-emerald-600',
    shadow: 'shadow-emerald-200',
  },
  {
    id: 'moving_target_interception',
    name: 'Moving Target Interception',
    shortName: 'Interception',
    subtitle: '预测、时机控制、拦截',
    icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0a9 9 0 0118 0z',
    color: 'orange',
    gradient: 'from-orange-500 to-orange-600',
    shadow: 'shadow-orange-200',
  },
  {
    id: 'adaptive_boundary_challenge',
    name: 'Adaptive Boundary Challenge',
    shortName: 'Boundary',
    subtitle: '工作空间、边界控制',
    icon: 'M4 5a1 1 0 011-1h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5z',
    color: 'purple',
    gradient: 'from-purple-500 to-purple-600',
    shadow: 'shadow-purple-200',
  },
  {
    id: 'rhythmic_switching',
    name: 'Rhythmic Switching',
    shortName: 'Rhythm',
    subtitle: '节律同步、交替协调',
    icon: 'M7 8h10M7 12h10M7 16h10',
    color: 'rose',
    gradient: 'from-rose-500 to-rose-600',
    shadow: 'shadow-rose-200',
  },
  {
    id: 'mirror_mapping_reach',
    name: 'Mirror Mapping Reach',
    shortName: 'Mirror',
    subtitle: '视觉-运动转换、镜像映射',
    icon: 'M8 7h8M8 12h8M8 17h8M4 4v16M20 4v16',
    color: 'cyan',
    gradient: 'from-cyan-500 to-cyan-600',
    shadow: 'shadow-cyan-200',
  },
];

export const getTaskScore = (source, taskId, fallback = 0) => {
  if (!source) return fallback;
  return source[taskId] ?? source[`${taskId}_score`] ?? fallback;
};
