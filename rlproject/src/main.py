import traceback
import numpy as np
import cv2
import time
import os
import sys
from collections import deque
from matplotlib import pyplot as plt
import pygame
import mediapipe as mp
import gymnasium as gym
from gymnasium.spaces import Box

# Add RL root to sys.path so PPO can unpickle src.utils.feature_extractors
_rl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _rl_root not in sys.path:
    sys.path.insert(0, _rl_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback # [新增] 用于回调
from stable_baselines3.common.logger import configure       # [新增] 用于配置日志
from ultralytics import YOLO

# 导入自定义库
from custom_env import EnvironmentRenderer
from camera_calibration.camera_calibration import CameraCalibration
from robot_control.ur_control import URControl  
from cv.hand_detect import HandDetection
from cv.metric import PatientTrajectoryAnalyzer
from cv.get_workspace import get_workspace

# ========================================================
# 1. PPO 真机微调超参数
# ========================================================
FINE_TUNE_MODE = True
FINE_TUNE_LR = 5e-5        # 极低学习率
PPO_N_STEPS = 512          # 每 512 步更新一次
PPO_BATCH_SIZE = 256
PPO_CLIP_RANGE = 0.1       # 限制策略更新幅度
# PPO_ENT_COEF = 0.0         # 关闭盲目探索
MAX_STEPS_PER_EPISODE = 200
CONTROL_FREQ = 25.0        # 强制固定真机控制频率 25Hz (dt=0.04s)

# ========================================================
# [新增] 实时监控回调函数 (RealWorldMonitorCallback)
# ========================================================
class RealWorldMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_diffs = []
        self.zpd_hits =[]

    def _on_step(self) -> bool:
        # 获取环境 step 吐出的 info (由于使用了 DummyVecEnv，info在列表中)
        info = self.locals["infos"][0] 
        
        if "action_diff" in info:
            self.action_diffs.append(info["action_diff"])
            self.zpd_hits.append(info["is_in_zpd"])
        return True

    def _on_rollout_end(self) -> None:
        # 每当 PPO 收集完一个 buffer (n_steps) 准备更新网络时，触发此函数
        if len(self.action_diffs) > 0:
            mean_action_diff = np.mean(self.action_diffs)
            mean_zpd_rate = np.mean(self.zpd_hits) * 100.0 # 转为百分比
            
            # 推送到 TensorBoard
            self.logger.record("real_world_metrics/action_jitter", mean_action_diff)
            self.logger.record("real_world_metrics/zpd_maintenance_rate_%", mean_zpd_rate)
            
            # 打印在终端，方便肉眼判断何时停止
            print(f"\n📊 [实时监控] 正在进行网络反向传播...")
            print(f"   ➤ ZPD保持率: {mean_zpd_rate:.1f}% (目标:>75%)")
            print(f"   ➤ 动作抖动量: {mean_action_diff:.4f} (越低越平滑)")
            print(f"   [提示] 当抖动量触底平稳，且ZPD达标时，即可按 Ctrl+C 结束微调！\n")
            
            # 清空缓存，准备记录下一轮
            self.action_diffs.clear()
            self.zpd_hits.clear()

# 全局共享变量
pygame.init()
grid_s, cell_s = 10, 50
w_env, h_env = 15, 10
screen = pygame.display.set_mode((int(grid_s * cell_s * 1.5), int(grid_s * cell_s)))

cv_model = YOLO(r'C:\Users\admin\Desktop\huifeng\rlproject\src\runs\detect\train3\weights\best.onnx')
hand_detector = HandDetection()
cali = CameraCalibration()
analyzer = PatientTrajectoryAnalyzer()
render = EnvironmentRenderer(grid_size=10, cell_size=50)

robot_ip = "192.168.1.2"
robot_control = URControl(robot_ip)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  

if not cap.isOpened(): exit()
ret, frame = cap.read()
undistorted_frame = get_workspace(cali.undistort_frame(frame))
w,h = undistorted_frame.shape[:2]


# ========================================================
# 2. 真机硬件封装为标准 Gym 环境
# ========================================================
class RealWorldRehabEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.history_length = 16
        self.obs_dim = 10 + self.history_length * 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        self.stride_robot = 0.6
        self.rx_c, self.ry_c, self.rz_c = 0.193, 0.067, 5.3
        
        self.pixel_per_cm = 10.0
        self.fixed_point =[10, 10]
        self.last_hand = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32) # [新增] 用于计算动作抖动
        
        self.hand_history_buffer = deque(maxlen=self.history_length)
        for _ in range(self.history_length): self.hand_history_buffer.append(np.zeros(2))
        
        self.trajectory_robot = deque(maxlen=40)
        self.trajectory = deque(maxlen=60)
        
        self.start_time = time.perf_counter()
        self.last_control_time = time.time()
        self.ep_steps = 0
        self.virtual_target_pixel = np.array([w/2, h/2], dtype=np.float64) 
        
        self.target_dt = 1.0 / CONTROL_FREQ 

    def reset(self, seed=None, options=None):
        # time.sleep(1)
        print("\n[系统] 正在将机器人复位至桌面中心...")
        robot_control.rtde_c.servoStop()
        for i in range(2, 0, -1):
            print(f"[系统] 请受试者准备，{i} 秒后开始新回合...")
            time.sleep(1)
        center_pixel = np.array([w/2, h/2])
        center_world = cali.pixel_to_world(center_pixel.astype(int))
        target_pose =[center_world[0], center_world[1], 0.116, self.rx_c, self.ry_c, self.rz_c]
        robot_control.rtde_c.moveL(target_pose, 0.2, 0.2, asynchronous=False)
        
        
            
        self.virtual_target_pixel = center_pixel.astype(np.float64)
        self.ep_steps = 0
        self.last_action = np.zeros(2, dtype=np.float32) # [新增] 复位上一帧动作
        
        self.hand_history_buffer.clear()
        for _ in range(self.history_length): self.hand_history_buffer.append(np.zeros(2))
        self.trajectory.clear()
        
        obs = self._get_obs_and_render()
        self.last_control_time = time.time()
        return obs, {}

    def step(self, action):
        step_start_time = time.time() 
        
        # ==========================================
        # [新增] 提取平滑度与 ZPD 指标
        # ==========================================
        action_diff = np.linalg.norm(action - self.last_action)
        self.last_action = action.copy()

        # 1. 虚拟目标点累加控制
        action_pixel = action * np.array([w/w_env, h/h_env]) * self.stride_robot
        self.virtual_target_pixel += action_pixel
        self.virtual_target_pixel[0] = np.clip(self.virtual_target_pixel[0], 50, w-100)
        self.virtual_target_pixel[1] = np.clip(self.virtual_target_pixel[1], 50, h-100)
        
        target_position_world = cali.pixel_to_world(self.virtual_target_pixel.astype(int))
        target_pose =[target_position_world[0], target_position_world[1], 0.118, self.rx_c, self.ry_c, self.rz_c]

        # 2. 动态时间计算与发指令
        now = time.time()
        actual_dt = now - self.last_control_time
        safe_dt = np.clip(actual_dt, 0.01, 0.2) 
        self.last_control_time = now
        
        robot_control.servo_robot(target_pose, dt=safe_dt)
        
        # 3. 读取画面并计算 Obs
        obs = self._get_obs_and_render()
        
        # 4. 计算 Reward 和 Done
        distance_to_object = obs[4]
        z_min, z_max = 4.0, 6.0
        
        # [新增] 判断是否在 ZPD 区间内
        is_in_zpd = 1.0 if (z_min <= distance_to_object <= z_max) else 0.0
        
        if distance_to_object < z_min:
            reward = -0.1 * np.exp(1.5 * (z_min - distance_to_object) * 2)
        elif distance_to_object < z_max:
            reward = 0.4
        else:
            reward = -0.3 * (distance_to_object - z_max)
            
        reward = np.clip(reward, -10, 10)
        # reward -= 0.3 * np.linalg.norm(action)
        reward -= 1.5 * action_diff
        
        terminated = False
        truncated = False
        if distance_to_object < 2:
            terminated = True
            reward -= 20.0
        elif self.ep_steps >= MAX_STEPS_PER_EPISODE:
            truncated = True
            # reward += 10.0
            
        self.ep_steps += 1
        
        # 5. 强制频率锁定
        elapsed_time = time.time() - step_start_time
        if elapsed_time < self.target_dt:
            time.sleep(self.target_dt - elapsed_time)
            
        # ==========================================
        # [新增] 将指标塞入 info 返回给 Callback
        # ==========================================
        info = {
            "action_diff": action_diff,
            "is_in_zpd": is_in_zpd
        }
            
        return obs, reward, terminated, truncated, info

    def _get_obs_and_render(self):
        ret, frame = cap.read()
        if not ret: return np.zeros(self.obs_dim, dtype=np.float32)
        
        undistorted_frame = get_workspace(cali.undistort_frame(frame))
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        results = cv_model.predict(undistorted_frame, conf=0.7, save=False, imgsz=640, verbose=False)
        undistorted_frame, hand_positions = hand_detector.process_frame(undistorted_frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x2 - x1 > 100: continue
                self.trajectory_robot.append([(x1 + x2) // 2, (y1 + y2) // 2])
                cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (255, 0, 255), 4)
                self.pixel_per_cm = ((x2 + y2 - x1 - y1) / 2) / 2

        position_hand_env = np.array([0,0], dtype=np.float32)
        if hand_positions:
            position_hand_env = hand_positions[0] / np.array([w/w_env, h/h_env])
            analyzer.add_point(time.perf_counter() - self.start_time, position_hand_env[0], position_hand_env[1])

        hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 150, 255]))
        ys, xs = np.where(mask > 0)
        try:
            self.fixed_point =[xs[np.argmax(ys)].item()*w_env/w, h_env]
        except ValueError:
            pass

        *position_robot_world, _, _, _, _ = robot_control.get_robot_pose()
        real_robot_pixel = cali.world_to_pixel(position_robot_world)
        position_robot_env = real_robot_pixel[0]*w_env/w, real_robot_pixel[1]*h_env/h
        
        self.trajectory.append(position_robot_env)

        robot_obs = np.array([position_robot_env], dtype=np.float32).flatten()
        hand_obs = np.array([position_hand_env], dtype=np.float32).flatten()
        
        hand_move = hand_obs - self.last_hand
        distance_to_object = np.linalg.norm(robot_obs - hand_obs)
        distance_obs = np.array([distance_to_object], dtype=np.float32)
        boundary_obs = np.array([robot_obs[0], w_env-robot_obs[0], robot_obs[1], h_env-robot_obs[1]])
        
        vec_arm = self.fixed_point - hand_obs
        dist_arm = np.linalg.norm(vec_arm)
        to_shoulder = self.safe_normalize(vec_arm)
        blocking_point = hand_obs + to_shoulder * min(1, dist_arm)
        
        self.hand_history_buffer.append(hand_move)
        flat_history = np.array(self.hand_history_buffer).flatten()

        # obs_array = np.concatenate((
        #     robot_obs, hand_obs, distance_obs, boundary_obs, 
        #     np.array([self.stride_robot]), np.array(self.fixed_point, dtype=np.float32), 
        #     blocking_point.flatten(), np.zeros(2).flatten(), flat_history
        # ))
        obs_array = np.concatenate((
            robot_obs, hand_obs, distance_obs, boundary_obs, 
            np.array([self.stride_robot]), flat_history
        ))
        
        self.last_hand = hand_obs

        # === 画面渲染 ===
        # 当前距离
        cv2.putText(undistorted_frame, f"Distance: {distance_to_object:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        render.render(obs_array[:2], obs_array[2:4], self.fixed_point, self.trajectory, blocking_point.flatten())
        cv2.circle(undistorted_frame, (int(real_robot_pixel[0]), int(real_robot_pixel[1])), 20, (0, 255, 0), -1) 
        cv2.imshow('Frame', undistorted_frame)
        cv2.waitKey(1)

        return obs_array

    def safe_normalize(self, v):
        norm = np.linalg.norm(v)
        if norm < 1e-8: return np.zeros_like(v)
        return v / norm

import glob
def get_latest_model(model_dir, prefix="finetuned_ppo_real"):
    pattern = os.path.join(model_dir, f"{prefix}_*.zip")
    model_files = glob.glob(pattern)
    
    if not model_files:
        return None
    
    # 按修改时间排序（最新的在最后）
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model
# ========================================================
# 3. 主程序：加载模型并进行 PPO 在线微调
# ========================================================
if __name__ == '__main__':
    try:
        real_env = DummyVecEnv([
        lambda: Monitor(RealWorldRehabEnv())
    ])
        
        # # ⚠️ 请确保这里的路径是你最新仿真训练出来的 best_model
        # model_path = r"C:\Users\admin\Desktop\huifeng\RL\src\logs\ablation_study_0402_1321\2_MLP_LSTM\best_model.zip"
        
        # if FINE_TUNE_MODE:
        #     print(">>> 准备开启 PPO 真机在线微调 <<<")
            
        #     custom_args = {
        #         "n_steps": PPO_N_STEPS,
        #         "batch_size": PPO_BATCH_SIZE,
        #         "clip_range": lambda _: PPO_CLIP_RANGE, 
        #         "ent_coef": PPO_ENT_COEF
        #     }
            
        #     model = PPO.load(model_path, env=real_env, custom_objects=custom_args)
            
        #     # [新增] 设置 Logger，让监控数据推送到 Tensorboard
        #     tb_log_dir = f"logs/real_world_finetune_tb_{time.strftime('%m%d_%H%M')}"
        #     logger = configure(tb_log_dir, ["stdout", "tensorboard"])
        #     model.set_logger(logger)
            
        #     # 强行更新学习率
        #     model.lr_schedule = lambda _: FINE_TUNE_LR
        #     for param_group in model.policy.optimizer.param_groups:
        #         param_group['lr'] = FINE_TUNE_LR
                
        #     print(f"微调配置: LR={FINE_TUNE_LR}, n_steps={PPO_N_STEPS}")
        #     print("请受试者开始交互！(随时按 Ctrl+C 安全停止)")
            
        #     # [新增] 实例化并挂载我们刚刚写的监控 Callback
        #     monitor_cb = RealWorldMonitorCallback()
            
        #     # 微调 10000 步 (约 6 分钟)，期间观察终端输出，随时可以 Ctrl+C 停止
        #     model.learn(total_timesteps=10_000, callback=monitor_cb)
            
        #     save_name = f"logs/finetuned_ppo_real_{time.strftime('%H_%M')}.zip"
        #     model.save(save_name)
        #     print(f"✅ 微调自然结束，模型已保存: {save_name}")
            
        # else:
        #     # 纯部署模式，只预测不学习
        #     print(">>> 纯推理部署模式 (不进行微调) <<<")
        #     model = PPO.load(model_path, env=real_env)
        #     obs = real_env.reset()
        #     while True:
        #         action, _ = model.predict(obs, deterministic=True)
        #         obs, _, _, _ = real_env.step(action)

    # except KeyboardInterrupt:
    #     print("\n检测到 Ctrl+C，手动终止并保存模型...")
    #     if FINE_TUNE_MODE and 'model' in locals():
    #         save_name = f"logs/finetuned_ppo_real_{time.strftime('%H_%M')}_manual_stop.zip"
    #         model.save(save_name)
    #         print(f"✅ 紧急微调模型已保存: {save_name}")
        SIM_MODEL_PATH = r"C:\Users\admin\Desktop\huifeng\RL\src\logs\ablation_study_0409_0922\2_MLP_LSTM\best_model.zip" 
        # 2. 你上次微调到一半保存的真机模型 (如果没有就填 None)
        LAST_REAL_MODEL_PATH = get_latest_model("logs/model", prefix="finetuned_ppo_real")
        
        # --- 控制开关 ---
        RESUME_FROM_REAL_MODEL = True # True: 接着上次的真机模型训; False: 拿纯仿真模型开启第一次真机训
        
        if FINE_TUNE_MODE:
            print("\n" + "="*40)
            print(">>> 准备开启 PPO 真机在线微调 (多疗程模式) <<<")
            
            custom_args = {
                "n_steps": PPO_N_STEPS,
                "batch_size": PPO_BATCH_SIZE,
                "clip_range": lambda _: PPO_CLIP_RANGE, 
                # "ent_coef": PPO_ENT_COEF
            }
            
            # 🌟 [核心逻辑] 决定从哪个模型开始加载
            if RESUME_FROM_REAL_MODEL and LAST_REAL_MODEL_PATH and os.path.exists(LAST_REAL_MODEL_PATH):
                print(f"[*] 检测到历史微调模型，正在加载: {LAST_REAL_MODEL_PATH}")
                model = PPO.load(LAST_REAL_MODEL_PATH, env=real_env, custom_objects=custom_args)
                session_name = f"Session_Resume_{time.strftime('%m%d_%H%M')}"
            else:
                print(f"[*] 开始首次真机微调，加载仿真基础模型: {SIM_MODEL_PATH}")
                model = PPO.load(SIM_MODEL_PATH, env=real_env, custom_objects=custom_args)
                session_name = f"Session_New_{time.strftime('%m%d_%H%M')}"
            
            # [必须操作] 配置 Tensorboard，确保曲线能接上
            tb_log_dir = f"logs/real_world_finetune_tb/" # 保持同一个大文件夹
            logger = configure(tb_log_dir, ["stdout", "tensorboard"])
            model.set_logger(logger)
            
            # 🌟 [极其关键] 每次 load 完，必须再次强行压制学习率！
            model.lr_schedule = lambda _: FINE_TUNE_LR
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = FINE_TUNE_LR
                
            print(f"微调配置: LR={FINE_TUNE_LR}, n_steps={PPO_N_STEPS}")
            print("请受试者开始交互！(随时按 Ctrl+C 安全停止)")
            
            monitor_cb = RealWorldMonitorCallback()
            
            # 🌟 [核心秘诀] reset_num_timesteps=False 保证内部步数和曲线不断开
            model.learn(
                total_timesteps=2000, # 这次只训 5000 步 (比如 3 分钟)
                callback=monitor_cb, 
                reset_num_timesteps=not RESUME_FROM_REAL_MODEL, 
                tb_log_name="Real_World_Finetuning" # 保持名字一样，曲线就在同一张图里
            )
            
            # 保存这次微调的结果
            save_name = f"logs/model/finetuned_ppo_real_{session_name}.zip"
            model.save(save_name)
            print(f"✅ 本疗程微调结束，进度已保存: {save_name}")
    except Exception as e:
        print(f"\n[Error] 运行异常: {e}")
        traceback.print_exc()
        
    finally:
        print("断开机械臂，关闭系统...")
        if 'robot_control' in locals():
            robot_control.rtde_c.servoStop() # 强制刹车
            time.sleep(0.5)
            robot_control.disconnect()
        if 'cap' in locals(): cap.release()
        cv2.destroyAllWindows()
        pygame.quit()