"""Evaluation engine that wraps the original eval.py logic.

This module provides a clean interface for running the 4 evaluation tasks:
- Sprint: Reaction & explosive power
- Tracking: Multi-trajectory smooth tracking
- LeagueGame: Competition & cognitive interception
- Boundary: Range of motion & stability
"""
import sys
import os
import base64
import math
import time

_backend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_backend_dir)
_rl_root = os.path.dirname(_project_root)
_hw_src = os.path.join(_project_root, 'src')
for _p in [_hw_src, _rl_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cv.get_workspace import get_workspace

import numpy as np
import cv2
from collections import deque
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Container for evaluation results."""
    sprint: Optional[Dict] = None
    tracking: Optional[Dict] = None
    league: Optional[Dict] = None
    boundary: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            'sprint': self.sprint,
            'tracking': self.tracking,
            'league': self.league,
            'boundary': self.boundary
        }


@dataclass
class TaskProgress:
    """Track evaluation progress."""
    current_task: str = ""
    task_index: int = 0
    total_tasks: int = 4
    task_progress: float = 0.0
    message: str = ""


class EvalEngine:
    """Engine for running rehabilitation evaluation tasks."""

    def __init__(
        self,
        robot_ip: str = "192.168.1.2",
        yolo_model_path: str = None,
        rl_model_path: str = None,
        calibration_path: str = None,
        control_freq: float = 25.0,
        simulate: bool = True
    ):
        """
        Initialize evaluation engine.

        Args:
            robot_ip: UR robot IP address
            yolo_model_path: Path to YOLO ONNX model for robot detection
            rl_model_path: Path to PPO model for LeagueGame
            calibration_path: Path to calibration data
            control_freq: Control frequency in Hz
            simulate: If True, run in simulation mode without hardware
        """
        self.robot_ip = robot_ip
        self.control_freq = control_freq
        self.target_dt = 1.0 / control_freq
        self.simulate = simulate

        # Environment dimensions (from eval.py)
        self.w_env, self.h_env = 15, 10
        self.grid_size = 10
        self.MAX_SAFE_STRIDE = 0.6  # Max robot movement per step (from eval.py)

        # Paths
        _backend_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_backend_dir)
        _rl_root = os.path.dirname(_project_root)

        self.yolo_model_path = yolo_model_path or os.path.join(
            _project_root, 'src', 'runs', 'detect', 'train3', 'weights', 'best.onnx'
        )
        self.rl_model_path = rl_model_path or os.path.join(
            _rl_root, 'logs', 'ablation_study_0416_1050',
            '2_MLP_LSTM', 'best_model.zip'
        )

        # Hardware interfaces
        self.robot_control = None
        self.hand_detector = None
        self.cali = None
        self.rl_model = None
        self.cap = None

        # Camera dimensions
        self.w_px, self.h_px = 2592, 1944

        # Runtime state
        self._running = False
        self._current_task = None
        self._progress_callback: Optional[Callable] = None
        self._frame_callback: Optional[Callable] = None
        self._frame_broadcast_callback: Optional[Callable] = None

        # Robot pose constants (from eval.py)
        self.RX_C, self.RY_C, self.RZ_C = 0.193, 0.067, 5.3

        # FPS tracking
        self._fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0

        # Video recording
        self._video_writer = None
        self._video_path = None
        self._is_recording = False

        # Simulation state
        self._sim_hand_pos = None
        self._sim_robot_pos = None
        self._sim_frame_w = 640
        self._sim_frame_h = 480

        # For ideal trajectories (from eval.py)
        self.t_vals = np.linspace(0, 2 * math.pi, 500)
        self.ideal_trajectories = {
            'Circle': np.column_stack((
                self.w_env / 2 + (self.w_env / 3) * np.cos(self.t_vals),
                self.h_env / 2 + (self.h_env / 3) * np.sin(self.t_vals)
            )),
            'Figure-8': np.column_stack((
                self.w_env / 2 + (self.w_env / 3) * np.sin(self.t_vals),
                self.h_env / 2 + (self.h_env / 3) * np.sin(self.t_vals) * np.cos(self.t_vals)
            ))
        }

    def set_progress_callback(self, callback: Callable[[TaskProgress], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def set_frame_callback(self, callback: Callable[[np.ndarray, Dict], None]):
        """Set callback for receiving frames for visualization."""
        self._frame_callback = callback

    def set_frame_broadcast_callback(self, callback: Callable[[str], None]):
        """Set callback for broadcasting frames via WebSocket."""
        self._frame_broadcast_callback = callback

    def _update_progress(self, task: str, index: int, progress: float, message: str = ""):
        """Update progress and notify callback."""
        p = TaskProgress(
            current_task=task,
            task_index=index,
            total_tasks=4,
            task_progress=progress,
            message=message
        )
        if self._progress_callback:
            self._progress_callback(p)

    def _start_video_recording(self, session_id: int):
        """Start recording video to file."""
        if self._is_recording:
            return
        # Create videos directory
        videos_dir = os.path.join(os.path.dirname(_backend_dir), 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._video_path = os.path.join(videos_dir, f"eval_session_{session_id}_{timestamp}.mp4")
        # Use mp4v codec (more compatible)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(
            self._video_path, fourcc, 25.0, (self.w_px, self.h_px)
        )
        self._is_recording = True
        print(f"[EvalEngine] Started recording to {self._video_path}")

    def _stop_video_recording(self):
        """Stop recording video."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._is_recording = False
        print(f"[EvalEngine] Stopped recording")

    def get_video_path(self) -> Optional[str]:
        """Get the path to the recorded video."""
        return self._video_path

    def connect(self) -> bool:
        """Connect to hardware (robot, camera, models) or run in simulate mode."""
        if self.simulate:
            print("[EvalEngine] Running in SIMULATE mode (no hardware)")
            self._sim_hand_pos = np.array([self.w_env / 2, self.h_env / 2])
            self._sim_robot_pos = np.array([self.w_env / 2, self.h_env / 2])
            return True

        try:
            print("[EvalEngine] Connecting to robot...")

            from robot_control.ur_control import URControl
            from cv.hand_detect import HandDetection
            from camera_calibration.camera_calibration import CameraCalibration
            from stable_baselines3 import PPO

            self.robot_control = URControl(self.robot_ip)

            print("[EvalEngine] Loading hand detector...")
            self.hand_detector = HandDetection()

            print("[EvalEngine] Loading camera calibration...")
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.cali = CameraCalibration(
                calibration_matrix_path=os.path.join(_project_root, 'src', 'camera_calibration', 'calibration_data.npz'),
                homography_matrix_path=os.path.join(_project_root, 'src', 'camera_calibration', 'Homography_matrix.npy')
            )

            print("[EvalEngine] Loading RL model for LeagueGame...")
            self.rl_model = PPO.load(
                self.rl_model_path,
                custom_objects={'learning_rate': 0.0, 'optimizer_class': None}
            )

            print("[EvalEngine] Opening camera...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

            if not self.cap.isOpened():
                raise RuntimeError("Camera not accessible")

            # Get actual frame dimensions
            ret, frame = self.cap.read()
            if ret:
                undistorted_frame = get_workspace(self.cali.undistort_frame(frame))
                undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.h_px, self.w_px = undistorted_frame.shape[:2]

            print("[EvalEngine] Connected successfully!")
            return True

        except Exception as e:
            print(f"[EvalEngine] Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from hardware."""
        self._stop_video_recording()
        if self.cap:
            self.cap.release()
            self.cap = None

    def _get_frame_and_positions(self) -> tuple:
        """Capture frame and compute hand/robot positions."""
        if self.simulate:
            frame = self._generate_sim_frame()
            _, buffer = cv2.imencode('.jpg', frame)
            if self._frame_broadcast_callback:
                self._frame_broadcast_callback(base64.b64encode(buffer.tobytes()).decode('utf-8'))
            # Return in environment coordinates (matching w_env/h_env units)
            return None, self._sim_hand_pos.copy(), self._sim_robot_pos.copy()

        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        undistorted_frame = get_workspace(self.cali.undistort_frame(frame))
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h_px, w_px = undistorted_frame.shape[:2]

        # Detect hand position
        annotated_frame, hand_positions = self.hand_detector.process_frame(undistorted_frame)
        if hand_positions:
            hand_pixel = np.array(hand_positions[0], dtype=np.float64)
            # Convert pixel to environment coordinates (same unit as w_env/h_env)
            # hand_pixel is [x, y] in pixels, divide by pixel dimensions to get environment coords
            hand_env = np.array([hand_pixel[0] * self.w_env / w_px, hand_pixel[1] * self.h_env / h_px], dtype=np.float64)
        else:
            hand_pixel = None
            hand_env = None

        # Robot position from UR (world coords for LeagueGame RL model)
        *position_robot_world, _, _, _, _ = self.robot_control.get_robot_pose()
        real_robot_pixel = self.cali.world_to_pixel(position_robot_world)
        robot_world = np.array([position_robot_world[0], position_robot_world[1]])
        # Robot in environment coords for distance calculation
        robot_env = np.array([real_robot_pixel[0] * self.w_env / self.w_px, real_robot_pixel[1] * self.h_env / self.h_px], dtype=np.float64)

        if self._frame_callback:
            self._frame_callback(undistorted_frame, {
                'hand': hand_pixel,
                'robot': real_robot_pixel,
                'hand_env': hand_env,
                'robot_env': robot_env,
                'hand_world': hand_env,  # deprecated, use hand_env
                'robot_world': robot_world
            })
        if self._frame_broadcast_callback:
            _, buffer = cv2.imencode('.jpg', undistorted_frame)
            self._frame_broadcast_callback(base64.b64encode(buffer.tobytes()).decode('utf-8'))

        # Update FPS
        self._frame_count += 1
        elapsed = time.time() - self._fps_start_time
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = time.time()

        # Record frame to video
        if self._is_recording and self._video_writer is not None:
            self._video_writer.write(undistorted_frame)

        # Return hand_env and robot_env in environment coordinates (matching w_env/h_env units)
        return undistorted_frame, hand_env, robot_env

    def _generate_sim_frame(self) -> np.ndarray:
        """Generate a simulated camera frame for testing."""
        frame = np.zeros((self._sim_frame_h, self._sim_frame_w, 3), dtype=np.uint8)

        grid_color = (40, 40, 40)
        for i in range(0, self._sim_frame_w, 50):
            cv2.line(frame, (i, 0), (i, self._sim_frame_h), grid_color, 1)
        for i in range(0, self._sim_frame_h, 50):
            cv2.line(frame, (0, i), (self._sim_frame_w, i), grid_color, 1)

        robot_px = int(self._sim_robot_pos[0] / self.w_env * self._sim_frame_w)
        robot_py = int(self._sim_robot_pos[1] / self.h_env * self._sim_frame_h)
        cv2.circle(frame, (robot_px, robot_py), 15, (255, 0, 0), -1)
        cv2.putText(frame, "Robot", (robot_px - 25, robot_py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        hand_px = int(self._sim_hand_pos[0] / self.w_env * self._sim_frame_w)
        hand_py = int(self._sim_hand_pos[1] / self.h_env * self._sim_frame_h)
        cv2.circle(frame, (hand_px, hand_py), 12, (0, 255, 0), -1)
        cv2.putText(frame, "Hand", (hand_px - 20, hand_py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"Task: {self._current_task or 'Ready'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _move_to_center(self):
        """Move robot to center position."""
        if self.simulate:
            self._sim_robot_pos = np.array([self.w_env / 2, self.h_env / 2])
            return

        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        undistorted_frame = get_workspace(self.cali.undistort_frame(frame))
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_px, w_px = undistorted_frame.shape[:2]

        self.robot_control.rtde_c.servoStop()
        center_pixel = np.array([w_px / 2, h_px / 2])
        center_world = self.cali.pixel_to_world(center_pixel.astype(int))
        target_pose = [center_world[0], center_world[1], 0.116, self.RX_C, self.RY_C, self.RZ_C]
        self.robot_control.rtde_c.moveL(target_pose, 0.2, 0.2, asynchronous=False)

    def _send_robot_to_pixel(self, target_pixel: np.ndarray, dt: float = 0.04):
        """Send robot to target pixel position (from eval.py logic).

        Uses moveL for large jumps (point-to-point) and servoL for continuous tracking.
        """
        if self.simulate:
            # Convert pixel to env coordinates for simulation
            x = target_pixel[0] / self.w_px * self.w_env
            y = (self.h_px - target_pixel[1]) / self.h_px * self.h_env
            self._sim_robot_pos = np.array([x, y])
            return

        # Track last position to detect large jumps
        last_pixel = getattr(self, '_last_target_pixel', None)
        self._last_target_pixel = target_pixel.copy()

        if last_pixel is not None:
            pixel_jump = np.linalg.norm(target_pixel - last_pixel)
            jump_threshold = self.MAX_SAFE_STRIDE * (self.w_px / self.w_env) * 3  # 3x normal step

            if pixel_jump > jump_threshold:
                # Large jump detected - use moveL for smooth point-to-point motion
                target_world = self.cali.pixel_to_world(target_pixel.astype(int))
                target_pose = [target_world[0], target_world[1], 0.116, self.RX_C, self.RY_C, self.RZ_C]
                self.robot_control.rtde_c.moveL(target_pose, 0.2, 0.2, asynchronous=False)
                return

        # Convert pixel to world coordinates
        target_world = self.cali.pixel_to_world(target_pixel.astype(int))
        target_pose = [target_world[0], target_world[1], 0.116, self.RX_C, self.RY_C, self.RZ_C]
        self.robot_control.servo_robot(target_pose, dt=dt)

    def _countdown(self, seconds: int = 3):
        """Display countdown."""
        for i in range(seconds, 0, -1):
            self._update_progress(self._current_task, self._task_index, 0, f"准备中... {i}")
            time.sleep(1)

    def _generate_target_position(self, current_pos: np.ndarray, min_dist: float = 3.0) -> np.ndarray:
        """Generate random target position at least min_dist from current position."""
        while True:
            target = np.array([
                np.random.uniform(2.0, self.w_env - 2.0),
                np.random.uniform(2.0, self.h_env - 2.0)
            ])
            if np.linalg.norm(target - current_pos) >= min_dist:
                return target

    def safe_normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalize vector safely."""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return np.zeros_like(v)
        return v / norm
    
    def _moveto_sprint_target(self, target_env: np.ndarray, w_px: int, h_px: int):
        """
        将机器人用 moveL 直接移动到 Sprint 目标的世界坐标位置。
        仿真模式下直接更新模拟位置。
        """
        if self.simulate:
            self._sim_robot_pos = target_env.copy()
            return

        # env 坐标 → 像素坐标 → 世界坐标
        target_pixel = np.array([
            target_env[0] * w_px / self.w_env,
            target_env[1] * h_px / self.h_env
        ], dtype=np.float64)

        target_pixel[0] = np.clip(target_pixel[0], 50, w_px - 50)
        target_pixel[1] = np.clip(target_pixel[1], 50, h_px - 50)

        target_world = self.cali.pixel_to_world(target_pixel.astype(int))
        target_pose = [
            target_world[0], target_world[1], 0.116,
            self.RX_C, self.RY_C, self.RZ_C
        ]

        # 停止当前伺服，然后 moveL 直接到位
        self.robot_control.rtde_c.servoStop()
        self.robot_control.rtde_c.moveL(target_pose, 0.3, 0.3, asynchronous=False)
        print(f"  [Sprint] moveL → target_env={target_env}, world={target_world[:2]}")

    def run_sprint(self) -> Dict:
        """Run Sprint task: 5 target catches measuring reaction time and velocity.
        Robot moves to each target via moveL (direct), not servo accumulation.
        After being caught, robot immediately moveL to next target position.
        """
        self._current_task = "Sprint"
        self._task_index = 1
        self._running = True

        results = {'catch_times': [], 'peak_vels': []}

        frame, hand_env_init, _ = self._get_frame_and_positions()
        if frame is not None:
            h_px, w_px = frame.shape[:2]
        else:
            w_px, h_px = self.w_px, self.h_px

        self._countdown()

        sprint_catch_count = 0

        # ── 生成第一个目标并用 moveL 直接到位 ──
        sprint_target_env = self._generate_target_position(
            np.array([self.w_env / 2, self.h_env / 2]), min_dist=3.0
        )
        self._moveto_sprint_target(sprint_target_env, w_px, h_px)
        sprint_target_spawn_time = time.time()

        last_control_time = time.time()
        last_hand_env = np.zeros(2)
        inst_vel = 0.0

        while self._running and sprint_catch_count < 5:
            t_now = time.time()

            # 获取手部位置
            frame, hand_env, _ = self._get_frame_and_positions()
            if frame is not None:
                h_px, w_px = frame.shape[:2]

            if hand_env is not None:
                hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env
            else:
                hand_env = np.array([self.w_env / 2, self.h_env / 2])

            # 计算瞬时速度
            dt_vision = max(t_now - last_control_time, 0.01)
            hand_move = hand_env - last_hand_env
            inst_vel = np.linalg.norm(hand_move) / dt_vision
            last_hand_env = hand_env
            last_control_time = t_now

            # 记录峰值速度
            if len(results['peak_vels']) <= sprint_catch_count:
                results['peak_vels'].append(inst_vel)
            else:
                results['peak_vels'][sprint_catch_count] = max(
                    results['peak_vels'][sprint_catch_count], inst_vel
                )

            # 判断是否抓到
            dist_to_target = np.linalg.norm(sprint_target_env - hand_env)
            if dist_to_target < 1.5:
                catch_time = t_now - sprint_target_spawn_time
                results['catch_times'].append(catch_time)
                print(f"  -> Target {sprint_catch_count + 1} caught in {catch_time:.2f}s!")
                sprint_catch_count += 1

                if sprint_catch_count < 5:
                    # ── 直接 moveL 跳到新目标，不用伺服累加 ──
                    sprint_target_env = self._generate_target_position(
                        hand_env, min_dist=3.0
                    )
                    self._moveto_sprint_target(sprint_target_env, w_px, h_px)
                    sprint_target_spawn_time = time.time()
            else:
                self._update_progress(
                    "Sprint", 1, sprint_catch_count / 5,
                    f"第 {sprint_catch_count + 1}/5 次 - 距离: {dist_to_target:.2f}"
                )

            time.sleep(self.target_dt)

        return results



    def run_tracking(self) -> Dict:
        """Run Tracking task: Follow moving target along predefined paths (from eval.py)."""
        self._current_task = "Tracking"
        self._task_index = 2
        self._running = True

        results = {'rmse_list': [], 'jerk_list': []}

        frame, _, _ = self._get_frame_and_positions()
        if frame is not None:
            h_px, w_px = frame.shape[:2]
        else:
            w_px, h_px = self.w_px, self.h_px

        self._countdown()

        duration = 20
        start_time = time.time()
        virtual_target_pixel = np.array([w_px / 2, h_px / 2], dtype=np.float64)
        last_hand_env = None
        last_vel = None
        last_control_time = time.time()

        while self._running and (time.time() - start_time) < duration:
            t_now = time.time()
            elapsed = t_now - start_time
            progress = elapsed / duration

            # Get hand position
            frame, hand_env, _ = self._get_frame_and_positions()
            if frame is not None:
                h_px, w_px = frame.shape[:2]

            if hand_env is not None:
                hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env
            else:
                hand_env = None

            # Determine trajectory shape
            if elapsed < 10.0:
                shape_name = 'Circle'
                t = elapsed * 1.2
                target_x = self.w_env / 2 + (self.w_env / 3) * math.cos(t)
                target_y = self.h_env / 2 + (self.h_env / 3) * math.sin(t)
            else:
                shape_name = 'Figure-8'
                t_8 = (elapsed - 10.0) * 1.2
                target_x = self.w_env / 2 + (self.w_env / 3) * math.sin(t_8)
                target_y = self.h_env / 2 + (self.h_env / 3) * math.sin(t_8) * math.cos(t_8)

            target_env = np.array([target_x, target_y])

            # Calculate cross-track error (from eval.py)
            path_points = self.ideal_trajectories[shape_name]
            if hand_env is not None:
                distances_to_path = np.linalg.norm(path_points - hand_env, axis=1)
                cross_track_error = np.min(distances_to_path)
                results['rmse_list'].append(float(cross_track_error))

                # Calculate jerk
                vel = np.linalg.norm(hand_env - last_hand_env) / max(t_now - last_control_time, 0.01) if last_hand_env is not None else 0
                if last_vel is not None:
                    jerk = abs(vel - last_vel) / max(t_now - last_control_time, 0.01)
                    results['jerk_list'].append(float(jerk))
                last_vel = vel
                last_hand_env = hand_env

            # Set desired target
            desired_virtual_target = np.array([
                target_env[0] * w_px / self.w_env,
                target_env[1] * h_px / self.h_env
            ])

            # Safety clipping and step limitation
            desired_virtual_target[0] = np.clip(desired_virtual_target[0], 50, w_px - 50)
            desired_virtual_target[1] = np.clip(desired_virtual_target[1], 50, h_px - 50)

            max_pixel_step = self.MAX_SAFE_STRIDE * (w_px / self.w_env)
            diff_vec = desired_virtual_target - virtual_target_pixel
            dist_pixel = np.linalg.norm(diff_vec)

            if dist_pixel > max_pixel_step:
                virtual_target_pixel += (diff_vec / dist_pixel) * max_pixel_step
            else:
                virtual_target_pixel = desired_virtual_target.copy()

            target_world = self.cali.pixel_to_world(virtual_target_pixel.astype(int))
            target_pose = [target_world[0], target_world[1], 0.116, self.RX_C, self.RY_C, self.RZ_C]
            self.robot_control.servo_robot(target_pose, dt=safe_dt)
            last_control_time = t_now

            self._update_progress("Tracking", 2, progress, f"追踪中... {elapsed:.1f}s / {duration}s [{shape_name}] | FPS: {self._current_fps:.1f}")

        return results

    def run_league(self) -> Dict:
        """Run LeagueGame task: Avoid RL-controlled robot (from eval.py)."""
        self._current_task = "LeagueGame"
        self._task_index = 3
        self._running = True

        results = {'is_caught': False, 'survival_time': 0.0, 'dist_list': []}

        frame, _, _ = self._get_frame_and_positions()
        if frame is not None:
            h_px, w_px = frame.shape[:2]
        else:
            w_px, h_px = self.w_px, self.h_px

        self._countdown()

        duration = 30
        start_time = time.time()
        virtual_target_pixel = np.array([w_px / 2, h_px / 2], dtype=np.float64)
        last_control_time = time.time()
        hand_history = deque([np.zeros(2)] * 16, maxlen=16)
        last_hand_env = np.zeros(2)

        while self._running:
            t_now = time.time()
            elapsed = t_now - start_time

            if elapsed > duration:
                break

            progress = elapsed / duration
            self._update_progress("LeagueGame", 3, progress, f"对抗中... {elapsed:.1f}s / {duration}s | FPS: {self._current_fps:.1f}")

            # Get hand position (now returns hand_env and robot_env in env coords)
            frame, hand_env, robot_env = self._get_frame_and_positions()
            if frame is not None:
                h_px, w_px = frame.shape[:2]

            if hand_env is not None:
                hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env
            else:
                hand_env = np.array([self.w_env / 2, self.h_env / 2])

            if robot_env is not None:
                robot_env = robot_env[:2] if len(robot_env) >= 2 else robot_env
            else:
                robot_env = np.array([self.w_env / 2, self.h_env / 2])

            dist_hand_robot = np.linalg.norm(robot_env - hand_env)
            results['dist_list'].append(float(dist_hand_robot))

            # Check if caught
            if dist_hand_robot < 1.5:
                results['is_caught'] = True
                results['survival_time'] = elapsed
                print(f"  -> Robot CAUGHT at {elapsed:.2f}s!")
                break

            # Build observation for RL model (from eval.py)
            dist_obs = np.array([dist_hand_robot], dtype=np.float32)
            boundary_obs = np.array([robot_env[0], self.w_env - robot_env[0], robot_env[1], self.h_env - robot_env[1]])
            flat_history = np.array(hand_history).flatten()

            obs_array = np.concatenate((
                robot_env, hand_env, dist_obs, boundary_obs,
                np.array([0.6]), flat_history
            ))

            # Get action from RL model
            action, _ = self.rl_model.predict(obs_array, deterministic=True)

            # Convert action to target position (from eval.py)
            action_pixel = action * np.array([w_px / self.w_env, h_px / self.h_env]) * 0.6
            desired_virtual_target = virtual_target_pixel + action_pixel

            # Safety clipping and step limitation
            desired_virtual_target[0] = np.clip(desired_virtual_target[0], 50, w_px - 50)
            desired_virtual_target[1] = np.clip(desired_virtual_target[1], 50, h_px - 50)

            max_pixel_step = self.MAX_SAFE_STRIDE * (w_px / self.w_env)
            diff_vec = desired_virtual_target - virtual_target_pixel
            dist_pixel = np.linalg.norm(diff_vec)

            if dist_pixel > max_pixel_step:
                virtual_target_pixel += (diff_vec / dist_pixel) * max_pixel_step
            else:
                virtual_target_pixel = desired_virtual_target.copy()

            target_world = self.cali.pixel_to_world(virtual_target_pixel.astype(int))
            target_pose = [target_world[0], target_world[1], 0.116, self.RX_C, self.RY_C, self.RZ_C]
            self.robot_control.servo_robot(target_pose, dt=safe_dt)
            hand_move = hand_env - last_hand_env
            hand_history.append(hand_move)
            last_hand_env = hand_env
            last_control_time = time.time()

        if not results['is_caught']:
            results['survival_time'] = duration
            print(f"  -> Robot survived full {duration}s!")

        return results

    def run_boundary(self) -> Dict:
        """Run Boundary task: Track target along rectangular boundary (from eval.py)."""
        self._current_task = "Boundary"
        self._task_index = 4
        self._running = True

        results = {'min_x': 999, 'max_x': 0, 'min_y': 999, 'max_y': 0, 'vel_list': []}

        frame, _, _ = self._get_frame_and_positions()
        if frame is not None:
            h_px, w_px = frame.shape[:2]
        else:
            w_px, h_px = self.w_px, self.h_px

        self._countdown()

        duration = 20
        start_time = time.time()
        virtual_target_pixel = np.array([w_px / 2, h_px / 2], dtype=np.float64)
        last_hand_env = None
        last_control_time = time.time()

        while self._running and (time.time() - start_time) < duration:
            t_now = time.time()
            elapsed = t_now - start_time
            progress = elapsed / duration

            # Move target along rectangle perimeter (from eval.py)
            perimeter = 2 * (self.w_env - 4) + 2 * (self.h_env - 4)
            prog = (elapsed / duration) * perimeter

            x_min, x_max = 1, self.w_env - 1
            y_min, y_max = 1, self.h_env - 1

            if prog < self.w_env - 4:
                tx, ty = 1 + prog, y_min
            elif prog < self.w_env - 4 + self.h_env - 4:
                tx, ty = x_max, y_min + (prog - (self.w_env - 4))
            elif prog < 2 * (self.w_env - 4) + self.h_env - 4:
                tx, ty = x_max - (prog - (self.w_env - 4) - (self.h_env - 4)), y_max
            else:
                tx, ty = x_min, y_max - (prog - 2 * (self.w_env - 4) - (self.h_env - 4))

            target_env = np.array([tx, ty])

            # Get hand position and update bounds
            frame, hand_env, _ = self._get_frame_and_positions()
            if frame is not None:
                h_px, w_px = frame.shape[:2]

            if hand_env is not None:
                hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env

                results['min_x'] = min(results['min_x'], hand_env[0])
                results['max_x'] = max(results['max_x'], hand_env[0])
                results['min_y'] = min(results['min_y'], hand_env[1])
                results['max_y'] = max(results['max_y'], hand_env[1])

                # Calculate velocity
                if last_hand_env is not None:
                    vel = np.linalg.norm(hand_env - last_hand_env) / max(t_now - last_control_time, 0.01)
                    results['vel_list'].append(float(vel))
                last_hand_env = hand_env

            # Set desired target
            desired_virtual_target = np.array([
                target_env[0] * w_px / self.w_env,
                target_env[1] * h_px / self.h_env
            ])

            # Safety clipping and step limitation
            desired_virtual_target[0] = np.clip(desired_virtual_target[0], 50, w_px - 50)
            desired_virtual_target[1] = np.clip(desired_virtual_target[1], 50, h_px - 50)

            max_pixel_step = self.MAX_SAFE_STRIDE * (w_px / self.w_env)
            diff_vec = desired_virtual_target - virtual_target_pixel
            dist_pixel = np.linalg.norm(diff_vec)

            if dist_pixel > max_pixel_step:
                virtual_target_pixel += (diff_vec / dist_pixel) * max_pixel_step
            else:
                virtual_target_pixel = desired_virtual_target.copy()

            target_world = self.cali.pixel_to_world(virtual_target_pixel.astype(int))
            target_pose = [target_world[0], target_world[1], 0.116, self.RX_C, self.RY_C, self.RZ_C]
            self.robot_control.servo_robot(target_pose, dt=safe_dt)
            last_control_time = t_now

            self._update_progress("Boundary", 4, progress, f"边界追踪... {elapsed:.1f}s / {duration}s | FPS: {self._current_fps:.1f}")

        return results

    def run_all(self, task_order: List[str] = None, session_id: int = None) -> EvalResult:
        """Run all evaluation tasks."""
        if task_order is None:
            task_order = ['sprint', 'tracking', 'league', 'boundary']

        result = EvalResult()
        self._running = True

        # Start video recording
        if session_id is not None:
            self._start_video_recording(session_id)

        self._move_to_center()

        task_map = {
            'sprint': self.run_sprint,
            'tracking': self.run_tracking,
            'league': self.run_league,
            'boundary': self.run_boundary
        }

        for i, task_name in enumerate(task_order):
            if not self._running:
                break

            task_func = task_map.get(task_name.lower())
            if task_func:
                task_result = task_func()
                if task_name.lower() == 'sprint':
                    result.sprint = task_result
                elif task_name.lower() == 'tracking':
                    result.tracking = task_result
                elif task_name.lower() == 'league':
                    result.league = task_result
                elif task_name.lower() == 'boundary':
                    result.boundary = task_result

            if self._running:
                self._move_to_center()

        # Stop video recording
        self._stop_video_recording()

        return result

    def stop(self):
        """Stop the current evaluation."""
        self._running = False
        self._stop_video_recording()
