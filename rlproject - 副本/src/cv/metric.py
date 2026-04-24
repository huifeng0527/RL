import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class PatientTrajectoryAnalyzer:
    def __init__(self):
        # 存储格式: [t, hand_x, hand_y]
        self.raw_data = [] 
        self.last_analysis_result = None 
        
    def add_point(self, t, x, y):
        """
        记录数据点 
        t: 秒 (float)
        x, y: 厘米 (float)
        """
        self.raw_data.append([t, x, y])

    def save_csv(self, filename="patient_data.csv"):
        if not self.raw_data:
            print("[Warning] 没有数据可保存")
            return
        df = pd.DataFrame(self.raw_data, columns=["time", "x", "y"])
        df.to_csv(filename, index=False)
        print(f"原始数据已保存至 {filename}")

    def compute_clinical_metrics(self, smooth_window=11):
        """计算指标并更新内部状态"""
        if len(self.raw_data) < 20:
            print("[Warning] 数据点不足 (<20)，无法分析")
            return None

        # --- 1. 数据预处理 ---
        data = np.array(self.raw_data, dtype=float)
        t = data[:, 0]
        t = t - t[0] # 从0秒开始
        pos = data[:, 1:3]

        # [新增] 简单的单位检查
        if t[-1] > 3600: # 如果持续时间大于1小时，可能是毫秒
            print("[Auto-Correction] 检测到时间单位可能是毫秒，转换为秒")
            t /= 1000.0

        # --- 2. 平滑处理 (Savgol Filter) ---
        window_length = min(smooth_window, len(t) if len(t)%2==1 else len(t)-1)
        if window_length < 3: window_length = 3
        
        pos_smooth = np.zeros_like(pos)
        try:
            pos_smooth[:, 0] = savgol_filter(pos[:, 0], window_length, 3)
            pos_smooth[:, 1] = savgol_filter(pos[:, 1], window_length, 3)
        except:
            pos_smooth = pos

        # --- 3. 微分计算 (关键物理量) ---
        # 时间差
        dt = np.gradient(t)
        dt[dt < 1e-5] = 1e-5 # 防止除零
        
        # 位移向量 (用于计算空间曲率)
        # 注意：np.gradient 计算的是中心差分，形状与输入一致
        deltas = np.gradient(pos_smooth, axis=0) 
        
        # 速度
        vel = deltas / dt[:, None]
        speed = np.linalg.norm(vel, axis=1)
        
        # 加速度
        acc = np.gradient(vel, axis=0) / dt[:, None]
        acc_mag = np.linalg.norm(acc, axis=1)
        
        # 加加速度 (Jerk)
        jerk = np.gradient(acc, axis=0) / dt[:, None]
        jerk_mag = np.linalg.norm(jerk, axis=1)

        # --- 4. 指标计算 ---
        
        # A. 时间平滑度 (Temporal Smoothness)
        # 使用对数均方 Jerk，取负数使其变大为好，或者保留原始值越小越好
        # 这里为了直观，直接用 Mean Squared Jerk
        smoothness_temporal = -np.log(np.sqrt(np.mean(jerk_mag**2)) + 1e-5)
        
        # B. 震颤 (Tremor)
        tremor_score = np.std(acc_mag)
        
        # C. 路径效率 (Efficiency)
        path_len = np.sum(np.sqrt(np.sum(np.diff(pos_smooth, axis=0)**2, axis=1)))
        straight_len = np.linalg.norm(pos_smooth[-1] - pos_smooth[0])
        path_efficiency = path_len / (straight_len + 1e-5) if straight_len > 1.0 else 1.0

        # D. 空间平滑度/曲率 (Spatial Smoothness)
        # 计算每一帧的运动方向角 (弧度)
        angles = np.arctan2(vel[:, 1], vel[:, 0])
        # 解卷绕 (处理 -180 到 180 的跳变)
        angles_unwrapped = np.unwrap(angles)
        # 计算角度变化
        angle_changes = np.gradient(angles_unwrapped)
        # 计算每厘米转过的角度 (曲率近似 k = dθ/ds)
        step_lengths = np.linalg.norm(deltas, axis=1)
        step_lengths = np.maximum(step_lengths, 1e-5) # 防止静止时除零
        curvature = np.abs(angle_changes) / step_lengths
        # 取平均曲率 (越小越直)
        smoothness_spatial = np.mean(curvature)

        # E. 运动范围 (ROM)
        try:
            hull = ConvexHull(pos_smooth)
            rom_area = hull.volume
        except:
            rom_area = 0.0

        metrics = {
            "duration": float(round(t[-1], 2)),
            "smoothness_temp": float(round(smoothness_temporal, 2)), # 越大越平滑(因为取了负对数)
            "smoothness_spatial": float(round(smoothness_spatial, 2)), # 越小越平滑(曲率)
            "tremor": float(round(tremor_score, 2)),
            "efficiency": float(round(path_efficiency, 2)),
            "rom_area": float(round(rom_area, 2)),
            "mean_speed": float(round(np.mean(speed), 2)),
            "max_speed": float(round(np.max(speed), 2))
        }

        self.last_analysis_result = {
            "time": t,
            "pos": pos_smooth,
            "speed": speed,
            "jerk": jerk_mag,
            "curvature": curvature,
            "metrics": metrics
        }

        return metrics
    
    def get_vel(self):
        """获取最近几帧的速度数据(用于实时控制)"""
        data = np.array(self.raw_data, dtype=float)
        if len(data) < 4:
            return 0.0, 0.0
        else:
            # 只取最后8帧计算实时速度
            recent_data = data[-64:,:]
            t = recent_data[:, 0]
            pos = recent_data[:, 1:3]

            dt = np.gradient(t)
            dt[dt==0] = 1e-5
            
            vel = np.gradient(pos, axis=0) / dt[:, None]
            speed = np.linalg.norm(vel, axis=1)
            return np.mean(speed), np.max(speed)

    def plot_report(self, save_path="report.png"):
        """绘制 2x2 的临床分析图表"""
        data = self.last_analysis_result
        
        if data is None:
            print("[Error] 请先调用 compute_clinical_metrics() 进行计算")
            return

        t = data['time']
        speed = data['speed']
        jerk = data['jerk']
        curvature = data['curvature'] # 空间曲率
        pos = data['pos']
        metrics = data['metrics']

        # 设置绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.3)
        
        # --- 图1: 轨迹与区域 (Trajectory & ROM) ---
        axs[0, 0].plot(pos[:, 0], pos[:, 1], 'b-', linewidth=1.5, label='Path')
        axs[0, 0].plot(pos[0, 0], pos[0, 1], 'go', markersize=8, label='Start')
        axs[0, 0].plot(pos[-1, 0], pos[-1, 1], 'rx', markersize=8, label='End')
        # 尝试画凸包边界
        try:
            hull = ConvexHull(pos)
            for simplex in hull.simplices:
                axs[0, 0].plot(pos[simplex, 0], pos[simplex, 1], 'k--', alpha=0.3)
        except:
            pass
        axs[0, 0].set_title(f"Trajectory (ROM Area: {metrics['rom_area']} $cm^2$)", fontsize=12)
        axs[0, 0].set_xlabel("X (cm)")
        axs[0, 0].set_ylabel("Y (cm)")
        axs[0, 0].legend()
        axs[0, 0].axis('equal') # 保持比例，不然圆会变成椭圆

        # --- 图2: 速度曲线 (Velocity Profile) ---
        axs[0, 1].plot(t, speed, color='#2ecc71', linewidth=1.5)
        axs[0, 1].fill_between(t, speed, color='#2ecc71', alpha=0.1)
        axs[0, 1].set_title(f"Velocity  (Avg: {metrics['mean_speed']} cm/s)", fontsize=12)
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("Speed (cm/s)")

        # --- 图3: 时间平滑度 (Jerk / Temporal Smoothness) ---
        axs[1, 0].plot(t, jerk, color='#e74c3c', linewidth=1)
        axs[1, 0].set_title(f"Temporal Smoothness (Log Jerk Score: {metrics['smoothness_temp']})", fontsize=12)
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("Jerk ($cm/s^3$)")
        # 添加一条参考线，显示哪里抖动最厉害
        axs[1, 0].axhline(y=np.mean(jerk), color='gray', linestyle='--', alpha=0.5, label='Mean Jerk')
        axs[1, 0].legend()

        # --- 图4: 空间平滑度 (Curvature / Spatial Smoothness) ---
        # 过滤掉极端值以便绘图好看
        disp_curvature = np.clip(curvature, 0, 10) 
        axs[1, 1].plot(t, disp_curvature, color='#9b59b6', linewidth=1)
        axs[1, 1].set_title(f"Spatial Smoothness (Mean Curvature: {metrics['smoothness_spatial']})", fontsize=12)
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("Curvature (rad/cm)")
        # axs[1, 1].text(0.05, 0.9, "Lower is Straighter", transform=axs[1, 1].transAxes, fontsize=10, color='gray')

        # 保存
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        except Exception as e:
            print(f"保存失败: {e}")
            
        plt.close(fig)