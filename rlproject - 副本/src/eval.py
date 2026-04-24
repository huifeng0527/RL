import traceback
import numpy as np
import cv2
import time
import os
import math
from collections import deque
from matplotlib import pyplot as plt
import pygame

from stable_baselines3 import PPO
from ultralytics import YOLO

# ========================================================
# 导入自定义库
# ========================================================
from custom_env import EnvironmentRenderer
from camera_calibration.camera_calibration import CameraCalibration
from robot_control.ur_control import URControl  
from cv.hand_detect import HandDetection
from cv.get_workspace import get_workspace

# ========================================================
# 1. 系统与测评参数配置
# ========================================================
w_env, h_env = 15, 10  # 环境物理尺寸 (单位)
CONTROL_FREQ = 25.0
RX_C, RY_C, RZ_C = 0.193, 0.067, 5.3 # 机械臂固定姿态

# 评估任务配置 (状态机)
EVAL_TASKS =['Sprint', 'Tracking', 'LeagueGame', 'Boundary']
current_task_idx = 0

# 记录评估数据的字典
eval_results = {
    'Sprint': {'catch_times': [], 'peak_vels':[]},
    'Tracking': {'rmse_list': [], 'jerk_list':[]},
    'LeagueGame': {'is_caught': False, 'survival_time': 0.0, 'dist_list':[]},
    'Boundary': {'min_x': 999, 'max_x': 0, 'min_y': 999, 'max_y': 0, 'vel_list':[]}
}

# ========================================================
# 2. 硬件与模型初始化
# ========================================================
pygame.init()
screen = pygame.display.set_mode((int(10 * 50 * 1.5), int(10 * 50)))

print("[初始化] 加载视觉模型...")
cv_model = YOLO(r'C:\Users\admin\Desktop\huifeng\rlproject\src\runs\detect\train3\weights\best.onnx')
hand_detector = HandDetection()
cali = CameraCalibration()

print("[初始化] 连接机械臂...")
robot_control = URControl("192.168.1.2")

print("[初始化] 加载 RL 决策大模型...")
# ⚠️ 替换为你训练好的纯 RL PPO 模型
rl_model_path = r"C:\Users\admin\Desktop\huifeng\RL\src\logs\ablation_study_0409_0922\2_MLP_LSTM\best_model.zip" 
try:
    rl_model = PPO.load(rl_model_path, custom_objects={'learning_rate': 0.0, 'optimizer_class': None})
except Exception as e:
    print(f"PPO 模型加载失败, 请检查路径: {e}")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
cv2.namedWindow('M-HECS Evaluation', cv2.WINDOW_NORMAL)  

if not cap.isOpened(): exit()
ret, frame = cap.read()
undistorted_frame = get_workspace(cali.undistort_frame(frame))
undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
h_px, w_px = undistorted_frame.shape[:2]

render = EnvironmentRenderer(grid_size=10, cell_size=50)

def safe_normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8: return np.zeros_like(v)
    return v / norm

# ========================================================
# 3. 辅助函数：安全复位与过渡
# ========================================================
def safe_transition_to_center(task_name):
    print(f"\n{'='*40}")
    print(f"即将开始评估任务: 【{task_name}】")
    print(f"{'='*40}")
    
    robot_control.rtde_c.servoStop()
    center_pixel = np.array([w_px/2, h_px/2])
    center_world = cali.pixel_to_world(center_pixel.astype(int))
    target_pose = [center_world[0], center_world[1], 0.116, RX_C, RY_C, RZ_C]
    
    robot_control.rtde_c.moveL(target_pose, 0.2, 0.2, asynchronous=False)
    
    for i in range(3, 0, -1):
        print(f">>> 请受试者准备，{i} 秒后开始...")
        time.sleep(1)
    print(">>> 评估开始！")
    return center_pixel.astype(np.float64)


# ========================================================
# [新增] 预生成 Task 2 的理想轨迹点云 (用于计算高精度轮廓误差 Cross-Track Error)
# ========================================================
t_vals = np.linspace(0, 2 * math.pi, 500) # 生成 500 个密集点
ideal_trajectories = {
    'Circle': np.column_stack((
        w_env/2 + (w_env/3) * np.cos(t_vals), 
        h_env/2 + (h_env/3) * np.sin(t_vals)
    )),
    'Figure-8': np.column_stack((
        w_env/2 + (w_env/3) * np.sin(t_vals), 
        h_env/2 + (h_env/3) * np.sin(t_vals) * np.cos(t_vals)
    )),
    'Line': np.column_stack((
        w_env/2 + (w_env/3) * np.sin(t_vals), 
        np.full_like(t_vals, h_env/2)
    ))
}
# ========================================================
# 4. 主测评循环 (含全局安全速度限制)
# ========================================================
try:
    center_pixel = safe_transition_to_center(EVAL_TASKS[current_task_idx])
    
    virtual_target_pixel = center_pixel.copy()
    desired_virtual_target = center_pixel.copy() 
    MAX_SAFE_STRIDE = 0.6 
    
    hand_history_buffer = deque([np.zeros(2)] * 16, maxlen=16)
    trajectory_robot = deque(maxlen=40)
    last_hand_env = np.zeros(2)
    fixed_point =[10, 10]
    pixel_per_cm = 10.0
    
    task_start_time = time.time()
    last_control_time = time.time()
    start_time = time.perf_counter()
    
    sprint_target_env = np.array([3.0, 3.0]) 
    sprint_catch_count = 0
    sprint_target_spawn_time = time.time()
    
    while current_task_idx < len(EVAL_TASKS):
        current_task = EVAL_TASKS[current_task_idx]
        t_now = time.time()
        task_elapsed = t_now - task_start_time
        
        # --- A. 视觉感知 ---
        ret, frame = cap.read()
        if not ret: break
        
        undistorted_frame = cv2.rotate(get_workspace(cali.undistort_frame(frame)), cv2.ROTATE_90_COUNTERCLOCKWISE)
        results = cv_model.predict(undistorted_frame, conf=0.7, save=False, imgsz=640, verbose=False)
        undistorted_frame, hand_positions = hand_detector.process_frame(undistorted_frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x2 - x1 > 100: continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trajectory_robot.append([int(cx), int(cy)])
                cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (255, 0, 255), 4)
                pixel_per_cm = ((x2 + y2 - x1 - y1) / 2) / 2

        if len(trajectory_robot) >= 2:
            for j in range(1, len(trajectory_robot)):
                thickness = int((j / len(trajectory_robot)) * 4) + 1
                cv2.line(undistorted_frame, trajectory_robot[j - 1], trajectory_robot[j], (0, 255, 255), thickness)

        position_hand_env = np.array([0, 0], dtype=np.float32)
        inst_vel = 0.0
        dt_vision = max((t_now - last_control_time), 0.01)
        
        if hand_positions:
            position_hand_env = hand_positions[0] / np.array([w_px/w_env, h_px/h_env])
            hand_move = position_hand_env - last_hand_env
            inst_vel = np.linalg.norm(hand_move) / dt_vision
            
            hand_history_buffer.append(hand_move)
            last_hand_env = position_hand_env

        *position_robot_world, _, _, _, _ = robot_control.get_robot_pose()
        real_robot_pixel = cali.world_to_pixel(position_robot_world)
        position_robot_env = real_robot_pixel[0]*w_env/w_px, real_robot_pixel[1]*h_env/h_px
        dist_hand_robot = np.linalg.norm(np.array(position_robot_env) - position_hand_env)

        hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 150, 255]))
        ys, xs = np.where(mask > 0)
        try: fixed_point =[xs[np.argmax(ys)].item()*w_env/w_px, h_env]
        except ValueError: pass

        # --- B. 任务状态机逻辑 ---
        task_finished = False
        
        # ----------------------------------------------------
        # 任务 1: Sprint (反应与爆发力)
        # ----------------------------------------------------
        if current_task == 'Sprint':
            cv2.putText(undistorted_frame, f"Task 1: ({sprint_catch_count}/5)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)
            
            desired_virtual_target[0] = sprint_target_env[0] * w_px / w_env
            desired_virtual_target[1] = sprint_target_env[1] * h_px / h_env
            
            if len(eval_results['Sprint']['peak_vels']) <= sprint_catch_count:
                eval_results['Sprint']['peak_vels'].append(inst_vel)
            else:
                eval_results['Sprint']['peak_vels'][sprint_catch_count] = max(eval_results['Sprint']['peak_vels'][sprint_catch_count], inst_vel)
            
            dist_to_target = np.linalg.norm(sprint_target_env - position_hand_env)
            if dist_to_target < 1.5: 
                catch_time = t_now - sprint_target_spawn_time
                eval_results['Sprint']['catch_times'].append(catch_time)
                print(f"  -> Target {sprint_catch_count+1} caught in {catch_time:.2f}s!")
                sprint_catch_count += 1
                if sprint_catch_count >= 5:
                    task_finished = True
                else:
                    min_travel_distance = 3.0 
                    sprint_target_env = np.random.uniform(2.0,[w_env-2.0, h_env-2.0])
                    while np.linalg.norm(sprint_target_env - position_hand_env) < min_travel_distance:
                        sprint_target_env = np.random.uniform(2.0,[w_env-2.0, h_env-2.0])
                    sprint_target_spawn_time = t_now
                    
        # ----------------------------------------------------
        # 任务 2: Tracking (多轨迹平滑追踪) - 修改为 3 个阶段
        # ----------------------------------------------------
        elif current_task == 'Tracking':
            # 任务 2: 平滑追踪 (采用交叉轨迹误差 Cross-Track Error)
            time_left = 20 - task_elapsed
            t = task_elapsed * 1.2 # 基础速度
            
            # 判断当前处于哪种轨迹阶段，并控制虚拟目标点的移动
            if task_elapsed < 10.0:
                shape_name = 'Circle'
                target_x = w_env/2 + (w_env/3) * math.cos(t)
                target_y = h_env/2 + (h_env/3) * math.sin(t)
            elif task_elapsed < 20.0:
                shape_name = 'Figure-8'
                t_8 = (task_elapsed - 10.0) * 1.2
                target_x = w_env/2 + (w_env/3) * math.sin(t_8)
                target_y = h_env/2 + (h_env/3) * math.sin(t_8) * math.cos(t_8)
                
            cv2.putText(undistorted_frame, f"Task 2: Tracking [{shape_name}] ({time_left:.1f}s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)
            
            # 赋给期望目标
            desired_virtual_target[0] = target_x * w_px / w_env
            desired_virtual_target[1] = target_y * h_px / h_env
            
            # ========================================================
            # 🌟 [核心修复] 计算真正的“轮廓误差 (Cross-Track Error)”
            # 不是算到机器人的距离，而是算手到理想轨迹线的垂直距离！
            # ========================================================
            # 拿到当前阶段的 500 个理想点
            path_points = ideal_trajectories[shape_name] 
            # 巧妙利用 numpy 广播机制，瞬间算出人手到 500 个点的距离
            distances_to_path = np.linalg.norm(path_points - position_hand_env, axis=1)
            # 取最小距离，即为到曲线的最短距离
            cross_track_error = np.min(distances_to_path)
            
            eval_results['Tracking']['rmse_list'].append(cross_track_error)
            eval_results['Tracking']['jerk_list'].append(inst_vel) 
            
            if task_elapsed >= 20: task_finished = True
            
        # ----------------------------------------------------
        # 任务 3: LeagueGame (对抗与安全距离) - 修改为抓到即结束
        # ----------------------------------------------------
        elif current_task == 'LeagueGame':
            cv2.putText(undistorted_frame, f"Task 3: catching ({30 - task_elapsed:.1f}s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)
            
            robot_obs = np.array([position_robot_env], dtype=np.float32).flatten()
            hand_obs = np.array([position_hand_env], dtype=np.float32).flatten()
            distance_obs = np.array([dist_hand_robot], dtype=np.float32)
            boundary_obs = np.array([robot_obs[0], w_env-robot_obs[0], robot_obs[1], h_env-robot_obs[1]])
            flat_history = np.array(hand_history_buffer).flatten()
            
            vec_arm = fixed_point - hand_obs
            dist_arm = np.linalg.norm(vec_arm)
            to_shoulder = safe_normalize(vec_arm)
            blocking_point = hand_obs + to_shoulder * min(1, dist_arm)
            
            obs_array = np.concatenate((
                robot_obs, hand_obs, distance_obs, boundary_obs, 
                np.array([0.6]), flat_history
            ))
            
            action, _ = rl_model.predict(obs_array, deterministic=True)
            action_pixel = action * np.array([w_px/w_env, h_px/h_env]) * 0.6
            desired_virtual_target += action_pixel
            
            # 记录动态距离
            eval_results['LeagueGame']['dist_list'].append(dist_hand_robot)
            
            # 抓取判定 (抓到立刻结束！)
            if dist_hand_robot < 1.5:
                eval_results['LeagueGame']['is_caught'] = True
                eval_results['LeagueGame']['survival_time'] = task_elapsed
                print(f"  -> Robot CAUGHT by patient at {task_elapsed:.2f}s!")
                task_finished = True
            elif task_elapsed >= 30.0:
                eval_results['LeagueGame']['is_caught'] = False
                eval_results['LeagueGame']['survival_time'] = 30.0
                print(f"  -> Robot survived the full 30s!")
                task_finished = True

        # ----------------------------------------------------
        # 任务 4: Boundary (活动范围与抖动) - 增加速度记录用于算抖动
        # ----------------------------------------------------
        elif current_task == 'Boundary':
            cv2.putText(undistorted_frame, f"Task 4: ROM ({20 - task_elapsed:.1f}s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)
            
            perimeter = 2 * (w_env - 4) + 2 * (h_env - 4)
            progress = (task_elapsed / 20.0) * perimeter
            if progress < w_env - 4:
                tx, ty = 1 + progress, 1
            elif progress < w_env - 4 + h_env - 4:
                tx, ty = w_env - 1, 1 + (progress - (w_env - 4))
            elif progress < 2 * (w_env - 4) + h_env - 4:
                tx, ty = w_env - 1 - (progress - (w_env - 4) - (h_env - 4)), h_env - 1
            else:
                tx, ty = 1, h_env - 1 - (progress - 2*(w_env - 4) - (h_env - 4))
                
            desired_virtual_target[0] = tx * w_px / w_env
            desired_virtual_target[1] = ty * h_px / h_env
            
            # 记录边界极值
            res = eval_results['Boundary']
            res['min_x'] = min(res['min_x'], position_hand_env[0])
            res['max_x'] = max(res['max_x'], position_hand_env[0])
            res['min_y'] = min(res['min_y'], position_hand_env[1])
            res['max_y'] = max(res['max_y'], position_hand_env[1])
            
            # 记录瞬时速度，最后用于计算 Jerk(加加速度/抖动)
            res['vel_list'].append(inst_vel)
            
            if task_elapsed >= 20.0: task_finished = True

        # ========================================================
        # --- C. 物理执行与安全限幅 ---
        # ========================================================
        desired_virtual_target[0] = np.clip(desired_virtual_target[0], 50, w_px-50)
        desired_virtual_target[1] = np.clip(desired_virtual_target[1], 50, h_px-50)
        
        max_pixel_step = MAX_SAFE_STRIDE * (w_px / w_env)
        diff_vec = desired_virtual_target - virtual_target_pixel
        dist_pixel = np.linalg.norm(diff_vec)
        
        if dist_pixel > max_pixel_step:
            virtual_target_pixel += (diff_vec / dist_pixel) * max_pixel_step
        else:
            virtual_target_pixel = desired_virtual_target.copy()
        
        target_position_world = cali.pixel_to_world(virtual_target_pixel.astype(int))
        target_pose = [target_position_world[0], target_position_world[1], 0.116, RX_C, RY_C, RZ_C]

        actual_dt = time.time() - last_control_time
        safe_dt = np.clip(actual_dt, 0.01, 0.2) 
        last_control_time = time.time()
        
        robot_control.servo_robot(target_pose, dt=safe_dt)
        
        # --- D. 渲染与任务切换 ---
        cv2.circle(undistorted_frame, (int(real_robot_pixel[0]), int(real_robot_pixel[1])), 20, (0, 255, 0), -1) 
        cv2.circle(undistorted_frame, (int(virtual_target_pixel[0]), int(virtual_target_pixel[1])), 10, (255, 0, 255), 2)
        cv2.imshow('M-HECS Evaluation', undistorted_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        if task_finished:
            current_task_idx += 1
            if current_task_idx < len(EVAL_TASKS):
                center_p = safe_transition_to_center(EVAL_TASKS[current_task_idx])
                virtual_target_pixel = center_p.copy()
                desired_virtual_target = center_p.copy()
                task_start_time = time.time()

except Exception as e:
    print(f"\n[Error] 测评异常: {e}")
    traceback.print_exc()

finally:
    print("\n[系统] 测评结束，正在安全关闭...")
    if 'robot_control' in locals():
        robot_control.rtde_c.servoStop()
        time.sleep(0.5)
        robot_control.disconnect()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
import matplotlib.gridspec as gridspec

# ========================================================
# 5. M-HECS 多维度雷达图报告生成 (基于 Min-Max 临床边界缩放)
# ========================================================
def normalize_score(value, bound_0, bound_100):
    """
    通用临床分数归一化函数 (Min-Max Scaling 变体)
    - value: 实际测量值
    - bound_0: 0分对应的极差物理边界
    - bound_100: 100分对应的理想物理边界
    如果 bound_100 > bound_0，说明指标越大越好 (如面积)；
    如果 bound_100 < bound_0，说明指标越小越好 (如时间、误差)。
    """
    # 核心映射公式
    score = 100.0 * (value - bound_0) / (bound_100 - bound_0)
    # 强行截断在 0 到 100 之间，防止越界
    return max(0.0, min(100.0, score))

def generate_radar_report(results):
    print("\n[系统] 正在基于临床边界生成 M-HECS 评估报告...")
    
    # ---------------------------------------------------------
    # 1. 严格的上下边界映射 (Clinical Anchors Mapping)
    # ---------------------------------------------------------
    
    # 【Task 1: Shooting】指标：平均捕捉耗时 (秒) -> 越小越好
    # 边界设定：健康人 0.8秒内抓到(100分)，超过 4.0秒视为重度迟缓(0分)
    avg_catch_time = np.mean(results['Sprint']['catch_times']) if results['Sprint']['catch_times'] else 4.0
    score_shooting = normalize_score(avg_catch_time, bound_0=3, bound_100=0.8)
    
    # 【Task 2: Tracking】指标：交叉轨迹误差 RMSE (环境单位) -> 越小越好
    # 边界设定：完全贴合为 0.0(100分)，严重脱轨偏离超过 2.5(约10厘米)为(0分)
    avg_rmse = np.mean(results['Tracking']['rmse_list']) if results['Tracking']['rmse_list'] else 2.5
    score_tracking = normalize_score(avg_rmse, bound_0=2, bound_100=0.0)
    
    # 【Task 3: Catching/LeagueGame】复合指标：患者视角
    survival_t = results['LeagueGame']['survival_time']
    dist_list = results['LeagueGame']['dist_list']
    avg_game_dist = np.mean(dist_list) if dist_list else 10.0
    
    # 子指标 3a：捕捉时间 -> 越小越好 (占60分)
    # 边界：秒抓 2.0秒(100分)，被机器人秀满 30.0秒没抓到(0分)
    time_score_normalized = normalize_score(survival_t, bound_0=10, bound_100=2.0)
    
    # 子指标 3b：压迫距离 -> 越小越好 (占40分)
    # 边界：死死咬住距离 2.0(100分)，被彻底甩开距离 8.0(0分)
    dist_score_normalized = normalize_score(avg_game_dist, bound_0=8.0, bound_100=3)
    
    score_catching = (time_score_normalized * 0.6) + (dist_score_normalized * 0.4)
    
    # 【Task 4: ROM】复合指标：面积与抖动
    b = results['Boundary']
    area = max(0, b['max_x'] - b['min_x']) * max(0, b['max_y'] - b['min_y'])
    max_area = (w_env - 4) * (h_env - 4) 
    
    # 子指标 4a：面积覆盖率 -> 越大越好 (占50分)
    # 边界：覆盖率为 0(0分)，完全覆盖 max_area(100分)
    area_score_normalized = normalize_score(area, bound_0=0.0, bound_100=max_area)
    
    # 子指标 4b：运动抖动 (Mean Jerk 近似) -> 越小越好 (占50分)
    vel_list = b['vel_list']
    mean_jerk = np.mean(np.abs(np.diff(vel_list))) if len(vel_list) > 1 else 3.0
    # 边界：丝滑无抖动 0.0(100分)，剧烈震颤 3.0(0分)
    jerk_score_normalized = normalize_score(mean_jerk, bound_0=3.0, bound_100=0.0)
    
    score_rom = (area_score_normalized * 0.5) + (jerk_score_normalized * 0.5)
    
    # ---------------------------------------------------------
    # 2. 临床量表等效推算 (Clinical Equivalency Mapping)
    # ---------------------------------------------------------
    s_shoot = score_shooting / 100.0
    s_track = score_tracking / 100.0
    s_catch = score_catching / 100.0
    s_rom = score_rom / 100.0

    mhecs_total = 100.0 * (0.20 * s_shoot + 0.30 * s_track + 0.30 * s_catch + 0.20 * s_rom)
    # FMA-UE (满分66)：侧重协同(Tracking)和范围(ROM)
    est_fma = 66.0 * (0.10 * s_shoot + 0.45 * s_track + 0.05 * s_catch + 0.40 * s_rom)
    # ARAT (满分57)：侧重爆发力(Shooting)和功能抓取(Catching)
    est_arat = 57.0 * (0.40 * s_shoot + 0.15 * s_track + 0.35 * s_catch + 0.10 * s_rom)
    
    # ================= 3. 绘图与排版美化 =================
    labels =['Shooting\n(Power & Reaction)', 'Tracking\n(Synergy & Smoothness)', 'Catching\n(Cognitive Interception)', 'ROM\n(Workspace & Stability)']
    stats = np.array([score_shooting, score_tracking, score_catching, score_rom])
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]
    
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2]) 
    
    # -- 雷达图 --
    ax = plt.subplot(gs[0], polar=True)
    ax.set_facecolor('#f8f9fa')
    ax.grid(color='#bdc3c7', linewidth=1.0, linestyle='--')
    ax.spines['polar'].set_visible(False)
    ax.plot(angles, stats, color='#2980b9', linewidth=2.5, linestyle='solid', marker='o', markersize=6)
    ax.fill(angles, stats, color='#3498db', alpha=0.3)
    ax.plot(angles, [60]*len(angles), color='#e74c3c', linewidth=1.5, linestyle=':') # 及格警戒线
    
    ax.set_yticklabels([])
    ax.set_ylim(0, 100) 
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold', color='#2c3e50')
    
    # -- 文本报告框 --
    ax_text = plt.subplot(gs[1])
    ax_text.axis('off') 
    
    report_text = (
        f"M-HECS Digital Report\n"
        f"{'-'*25}\n\n"
        f"[ Sub-Task Scores ]\n"
        f"1. Shooting:   {score_shooting:5.1f} / 100\n"
        f"2. Tracking:   {score_tracking:5.1f} / 100\n"
        f"3. Catching:   {score_catching:5.1f} / 100\n"
        f"4. ROM:        {score_rom:5.1f} / 100\n\n"
        f"{'-'*25}\n"
        f"OVERALL:       {mhecs_total:5.1f} / 100\n\n\n"
        f"[ Clinical Estimation ]\n"
        f"Est. FMA-UE:   {est_fma:5.1f} / 66\n"
        f"(Motor Synergy & Coord.)\n\n"
        f"Est. ARAT:     {est_arat:5.1f} / 57\n"
        f"(Handling & Reaching)"
    )
    
    ax_text.text(0.0, 0.9, report_text, fontsize=12, family='monospace', 
                 verticalalignment='top', color='#2c3e50',
                 bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.8))
    
    plt.suptitle('M-HECS: Magnetically-actuated Hand-Eye Coordination Scale', size=16, fontweight='bold', color='#2c3e50', y=0.95)
    plt.tight_layout()
    
    save_path = 'm_hecs_radar_clinical_report.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 临床级测评报告已保存为: {save_path}")
    plt.show()

# 运行结束后自动弹出雷达图
generate_radar_report(eval_results)