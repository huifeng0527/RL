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
    icon: 'M3 12h4l3-9 4 18 3-9h4',
    color: 'emerald',
    gradient: 'from-emerald-500 to-emerald-600',
    shadow: 'shadow-emerald-200',
  },
  {
    id: 'workspace_exploration',
    name: 'Workspace Exploration',
    shortName: 'Workspace',
    subtitle: '可及空间、边界控制',
    icon: 'M4 5a1 1 0 011-1h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5z',
    color: 'purple',
    gradient: 'from-purple-500 to-purple-600',
    shadow: 'shadow-purple-200',
  },
  {
    id: 'rhythmic_synchronization',
    name: 'Rhythmic Synchronization',
    shortName: 'Rhythm',
    subtitle: '节律同步、时间协调',
    icon: 'M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z',
    color: 'rose',
    gradient: 'from-rose-500 to-rose-600',
    shadow: 'shadow-rose-200',
  },
  {
    id: 'constrained_line_tracing',
    name: 'Constrained Line Tracing',
    shortName: 'Line Trace',
    subtitle: '精细路径控制',
    icon: 'M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z',
    color: 'cyan',
    gradient: 'from-cyan-500 to-cyan-600',
    shadow: 'shadow-cyan-200',
  },
];

export const getTaskScore = (source, taskId, fallback = 0) => {
  if (!source) return fallback;
  return source[taskId] ?? source[`${taskId}_score`] ?? fallback;
};
