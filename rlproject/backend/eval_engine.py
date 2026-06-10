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


EVALUATION_TASKS = [
    ("rapid_reach", "Rapid Reach"),
    ("continuous_tracking", "Continuous Tracking"),
    ("moving_target_interception", "Moving Target Interception"),
    ("adaptive_boundary_challenge", "Adaptive Boundary Challenge"),
    ("rhythmic_switching", "Rhythmic Switching"),
    ("mirror_mapping_reach", "Mirror Mapping Reach"),
]
EVALUATION_TASK_KEYS = [task[0] for task in EVALUATION_TASKS]
EVALUATION_TASK_NAMES = dict(EVALUATION_TASKS)
LEGACY_TASK_KEY_MAP = {
    "sprint": "rapid_reach",
    "tracking": "continuous_tracking",
    "boundary": "adaptive_boundary_challenge",
}


@dataclass
class EvalResult:
    """Container for evaluation results."""
    rapid_reach: Optional[Dict] = None
    continuous_tracking: Optional[Dict] = None
    moving_target_interception: Optional[Dict] = None
    adaptive_boundary_challenge: Optional[Dict] = None
    rhythmic_switching: Optional[Dict] = None
    mirror_mapping_reach: Optional[Dict] = None
    legacy_league: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "rapid_reach": self.rapid_reach,
            "continuous_tracking": self.continuous_tracking,
            "moving_target_interception": self.moving_target_interception,
            "adaptive_boundary_challenge": self.adaptive_boundary_challenge,
            "rhythmic_switching": self.rhythmic_switching,
            "mirror_mapping_reach": self.mirror_mapping_reach,
            "legacy_league": self.legacy_league,
        }


@dataclass
class TaskProgress:
    """Track evaluation progress."""
    current_task: str = ""
    task_index: int = 0
    total_tasks: int = 6
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
        control_freq: float = 12,
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
        default_rl_model_path = os.path.join(
            _rl_root, 'logs', 'dual_iterative_0427_1314', 'iteration_14', 'robot', 'robot', 'best_model.zip'
        )
        self.rl_model_path = rl_model_path or os.getenv('RL_MODEL_PATH') or default_rl_model_path

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
        self.RX_C, self.RY_C, self.RZ_C,self.z = 0.193, 0.067, 5.3,0.115

        # FPS tracking
        self._fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        self._frame_broadcast_interval = 1.0 / 12.0
        self._last_frame_broadcast_time = 0.0

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
                self.h_env / 2 + (self.h_env / 3) * np.sin(self.t_vals) * np.cos(self.t_vals)  # FIX 2: was w_env/2, now h_env/2
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
            total_tasks=len(EVALUATION_TASKS),
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

    def _broadcast_frame(self, frame: np.ndarray):
        if not self._frame_broadcast_callback:
            return
        now = time.time()
        if now - self._last_frame_broadcast_time < self._frame_broadcast_interval:
            return
        self._last_frame_broadcast_time = now
        success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            self._frame_broadcast_callback(base64.b64encode(buffer.tobytes()).decode('utf-8'))

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
            if not os.path.exists(self.rl_model_path):
                raise FileNotFoundError(f"RL model not found: {self.rl_model_path}")
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
            self.disconnect()
            return False

    def disconnect(self):
        """Disconnect from hardware."""
        self._running = False
        self._stop_video_recording()
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[EvalEngine] Failed to release camera: {e}")
            self.cap = None
        if self.hand_detector:
            try:
                self.hand_detector.release()
            except Exception as e:
                print(f"[EvalEngine] Failed to release hand detector: {e}")
            self.hand_detector = None
        if self.robot_control:
            try:
                self.robot_control.disconnect()
            except Exception as e:
                print(f"[EvalEngine] Failed to disconnect robot: {e}")
            self.robot_control = None

    def _get_frame_and_positions(self) -> tuple:
        """Capture frame and compute hand/robot positions."""
        if self.simulate:
            frame = self._generate_sim_frame()
            self._broadcast_frame(frame)
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
        self._broadcast_frame(undistorted_frame)

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
        target_pose = [center_world[0], center_world[1], self.z, self.RX_C, self.RY_C, self.RZ_C]
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
                target_pose = [target_world[0], target_world[1], self.z, self.RX_C, self.RY_C, self.RZ_C]
                self.robot_control.rtde_c.moveL(target_pose, 0.2, 0.2, asynchronous=False)
                return

        # Convert pixel to world coordinates
        target_world = self.cali.pixel_to_world(target_pixel.astype(int))
        target_pose = [target_world[0], target_world[1], self.z, self.RX_C, self.RY_C, self.RZ_C]
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
            target_world[0], target_world[1], self.z,
            self.RX_C, self.RY_C, self.RZ_C
        ]

        # 停止当前伺服，然后 moveL 直接到位
        self.robot_control.rtde_c.servoStop()
        self.robot_control.rtde_c.moveL(target_pose, 3, 1, asynchronous=False)
        print(f"  [Sprint] moveL → target_env={target_env}, world={target_world[:2]}")

    def run_rapid_reach(self) -> Dict:
        """Run Rapid Reach: sudden target reaching with per-trial timeout."""
        self._current_task = "Rapid Reach"
        self._task_index = 1
        self._running = True

        trial_count = 8
        target_radius = 1.5
        max_trial_time = 6.0
        results = {
            'catch_times': [],
            'peak_vels': [],
            'successes': [],
            'target_positions': [],
            'reaction_times': [],
            'movement_times': [],
            'endpoint_errors': [],
        }

        frame, _, _ = self._get_frame_and_positions()
        if frame is not None:
            h_px, w_px = frame.shape[:2]
        else:
            w_px, h_px = self.w_px, self.h_px

        self._countdown()

        completed_trials = 0
        target_env = self._generate_target_position(np.array([self.w_env / 2, self.h_env / 2]), min_dist=3.0)
        results['target_positions'].append(target_env.tolist())
        self._moveto_sprint_target(target_env, w_px, h_px)
        trial_start = time.time()
        last_control_time = time.time()
        last_hand_env = np.array([self.w_env / 2, self.h_env / 2], dtype=np.float64)
        movement_onset = None

        while self._running and completed_trials < trial_count:
            loop_start = time.time()
            t_now = loop_start

            if self.simulate:
                direction = target_env - self._sim_hand_pos
                self._sim_hand_pos = self._sim_hand_pos + self.safe_normalize(direction) * min(np.linalg.norm(direction), 0.35)

            frame, hand_env, _ = self._get_frame_and_positions()
            if frame is not None:
                h_px, w_px = frame.shape[:2]

            if hand_env is not None:
                hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env
            else:
                hand_env = last_hand_env.copy()

            dt_vision = max(t_now - last_control_time, 0.01)
            inst_vel = np.linalg.norm(hand_env - last_hand_env) / dt_vision
            if movement_onset is None and inst_vel > 0.3:
                movement_onset = t_now - trial_start

            if len(results['peak_vels']) <= completed_trials:
                results['peak_vels'].append(float(inst_vel))
            else:
                results['peak_vels'][completed_trials] = max(results['peak_vels'][completed_trials], float(inst_vel))

            dist_to_target = float(np.linalg.norm(target_env - hand_env))
            elapsed = t_now - trial_start
            caught = dist_to_target <= target_radius
            timed_out = elapsed >= max_trial_time

            if caught or timed_out:
                results['successes'].append(bool(caught))
                results['reaction_times'].append(float(movement_onset if movement_onset is not None else max_trial_time))
                results['movement_times'].append(float(elapsed))
                results['endpoint_errors'].append(dist_to_target)
                if caught:
                    results['catch_times'].append(float(elapsed))
                completed_trials += 1

                if completed_trials < trial_count:
                    target_env = self._generate_target_position(target_env, min_dist=3.0)
                    results['target_positions'].append(target_env.tolist())
                    self._moveto_sprint_target(target_env, w_px, h_px)
                    trial_start = time.time()
                    movement_onset = None
            else:
                self._update_progress(
                    "Rapid Reach", 1, completed_trials / trial_count,
                    f"快速到达 {completed_trials + 1}/{trial_count} - 距离: {dist_to_target:.2f}"
                )

            last_hand_env = hand_env
            last_control_time = t_now
            sleep_time = self.target_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        return results

    def run_sprint(self) -> Dict:
        return self.run_rapid_reach()

    def run_continuous_tracking(self) -> Dict:
        """Run Continuous Tracking: follow moving target along predefined paths."""
        self._current_task = "Continuous Tracking"
        self._task_index = 2
        self._running = True

        results = {'rmse_list': [], 'jerk_list': [], 'trajectory_names': []}

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
        last_perception_time = time.time()

        while self._running and (time.time() - start_time) < duration:
            loop_start = time.time()
            t_now = loop_start
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
                target_y = self.h_env / 2 + (self.h_env / 3) * math.sin(t_8) * math.cos(t_8)  # FIX 2: h_env/2 (was w_env/2)

            target_env = np.array([target_x, target_y])

            # Calculate cross-track error (from eval.py)
            path_points = self.ideal_trajectories[shape_name]
            if hand_env is not None:
                distances_to_path = np.linalg.norm(path_points - hand_env, axis=1)
                cross_track_error = np.min(distances_to_path)
                results['rmse_list'].append(float(cross_track_error))
                if not results['trajectory_names'] or results['trajectory_names'][-1] != shape_name:
                    results['trajectory_names'].append(shape_name)

                # Calculate jerk
                dt_perception = max(t_now - last_perception_time, 0.01)
                vel = np.linalg.norm(hand_env - last_hand_env) / dt_perception if last_hand_env is not None else 0
                if last_vel is not None:
                    jerk = abs(vel - last_vel) / dt_perception
                    results['jerk_list'].append(float(jerk))
                last_vel = vel
                last_hand_env = hand_env
                last_perception_time = t_now

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

            control_now = time.time()
            actual_dt = max(control_now - last_control_time, 0.01)
            safe_dt = min(actual_dt, 0.2)
            self._send_robot_to_pixel(virtual_target_pixel, dt=safe_dt)
            last_control_time = control_now

            self._update_progress("Continuous Tracking", 2, progress, f"连续追踪... {elapsed:.1f}s / {duration}s [{shape_name}] | FPS: {self._current_fps:.1f}")

            sleep_time = self.target_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if results['rmse_list']:
            errors = np.array(results['rmse_list'], dtype=np.float64)
            results['mean_error'] = float(np.mean(errors))
            results['max_error'] = float(np.max(errors))
            results['target_loss_rate'] = float(np.mean(errors > 1.5))
        else:
            results['mean_error'] = None
            results['max_error'] = None
            results['target_loss_rate'] = None
        return results

    def run_tracking(self) -> Dict:
        return self.run_continuous_tracking()

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
        last_control_time = time.time()  # FIX 3: timestamp before loop
        hand_history = deque([np.zeros(2)] * 16, maxlen=16)
        last_hand_env = np.zeros(2)

        while self._running:
            loop_start = time.time()
            t_now = loop_start
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
            if dist_hand_robot < 2:
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

            control_now = time.time()
            actual_dt = max(control_now - last_control_time, 0.01)
            safe_dt = min(actual_dt, 0.2)
            self._send_robot_to_pixel(virtual_target_pixel, dt=safe_dt)

            # Update hand history
            hand_move = hand_env - last_hand_env
            hand_history.append(hand_move)
            last_hand_env = hand_env
            last_control_time = control_now

            sleep_time = self.target_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if not results['is_caught']:
            results['survival_time'] = duration
            print(f"  -> Robot survived full {duration}s!")

        return results

    def _env_to_pixel(self, pos_env: np.ndarray, w_px: int, h_px: int) -> np.ndarray:
        return np.array([pos_env[0] * w_px / self.w_env, pos_env[1] * h_px / self.h_env], dtype=np.float64)

    def _step_virtual_target(self, current_pixel: np.ndarray, target_env: np.ndarray, w_px: int, h_px: int) -> np.ndarray:
        desired = self._env_to_pixel(target_env, w_px, h_px)
        desired[0] = np.clip(desired[0], 50, w_px - 50)
        desired[1] = np.clip(desired[1], 50, h_px - 50)
        max_pixel_step = self.MAX_SAFE_STRIDE * (w_px / self.w_env)
        diff_vec = desired - current_pixel
        dist_pixel = np.linalg.norm(diff_vec)
        if dist_pixel > max_pixel_step:
            return current_pixel + (diff_vec / dist_pixel) * max_pixel_step
        return desired.copy()

    def run_moving_target_interception(self) -> Dict:
        """Run Moving Target Interception with straight paths through a fixed intercept zone."""
        self._current_task = "Moving Target Interception"
        self._task_index = 3
        self._running = True

        trial_count = 6
        trial_duration = 4.0
        intercept_center = np.array([self.w_env / 2, self.h_env / 2], dtype=np.float64)
        intercept_radius = 1.0
        time_window = 0.4
        directions = [
            (np.array([1.0, 0.0]), np.array([0.0, 0.0])),
            (np.array([-1.0, 0.0]), np.array([0.0, 0.0])),
            (np.array([0.0, 1.0]), np.array([0.0, 0.0])),
            (np.array([0.0, -1.0]), np.array([0.0, 0.0])),
            (self.safe_normalize(np.array([1.0, 1.0])), np.array([0.0, 0.0])),
            (self.safe_normalize(np.array([-1.0, 1.0])), np.array([0.0, 0.0])),
        ]
        results = {
            'total_trials': trial_count,
            'successes': [],
            'timing_errors': [],
            'spatial_errors': [],
            'early_count': 0,
            'late_count': 0,
            'reaction_times': [],
        }

        frame, _, _ = self._get_frame_and_positions()
        if frame is not None:
            h_px, w_px = frame.shape[:2]
        else:
            w_px, h_px = self.w_px, self.h_px
        self._countdown()

        virtual_target_pixel = np.array([w_px / 2, h_px / 2], dtype=np.float64)
        last_control_time = time.time()
        for trial_idx in range(trial_count):
            if not self._running:
                break
            direction = directions[trial_idx % len(directions)][0]
            start_env = intercept_center - direction * 5.0
            end_env = intercept_center + direction * 5.0
            trial_start = time.time()
            t_ball = trial_duration / 2.0
            t_hand = None
            movement_onset = None
            last_hand_env = None
            min_spatial_error = float('inf')

            while self._running and (time.time() - trial_start) < trial_duration:
                loop_start = time.time()
                elapsed = loop_start - trial_start
                alpha = min(max(elapsed / trial_duration, 0.0), 1.0)
                target_env = start_env * (1 - alpha) + end_env * alpha

                if self.simulate:
                    self._sim_hand_pos = self._sim_hand_pos + self.safe_normalize(intercept_center - self._sim_hand_pos) * 0.25

                frame, hand_env, _ = self._get_frame_and_positions()
                if frame is not None:
                    h_px, w_px = frame.shape[:2]
                if hand_env is not None:
                    hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env
                    min_spatial_error = min(min_spatial_error, float(np.linalg.norm(hand_env - intercept_center)))
                    if last_hand_env is not None and movement_onset is None:
                        vel = np.linalg.norm(hand_env - last_hand_env) / max(time.time() - last_control_time, 0.01)
                        if vel > 0.3:
                            movement_onset = elapsed
                    last_hand_env = hand_env
                    if t_hand is None and np.linalg.norm(hand_env - intercept_center) <= intercept_radius:
                        t_hand = elapsed

                virtual_target_pixel = self._step_virtual_target(virtual_target_pixel, target_env, w_px, h_px)
                control_now = time.time()
                self._send_robot_to_pixel(virtual_target_pixel, dt=min(max(control_now - last_control_time, 0.01), 0.2))
                last_control_time = control_now
                self._update_progress("Moving Target Interception", 3, (trial_idx + alpha) / trial_count, f"移动拦截 {trial_idx + 1}/{trial_count}")

                sleep_time = self.target_dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            timing_error = None if t_hand is None else t_hand - t_ball
            success = timing_error is not None and abs(timing_error) <= time_window
            if timing_error is not None and timing_error < -time_window:
                results['early_count'] += 1
            elif timing_error is None or timing_error > time_window:
                results['late_count'] += 1
            results['successes'].append(bool(success))
            results['timing_errors'].append(float(timing_error) if timing_error is not None else None)
            results['spatial_errors'].append(float(min_spatial_error if np.isfinite(min_spatial_error) else intercept_radius * 3))
            results['reaction_times'].append(float(movement_onset if movement_onset is not None else trial_duration))

        return results

    def run_adaptive_boundary_challenge(self) -> Dict:
        """Run directional boundary exploration and near-boundary control."""
        self._current_task = "Adaptive Boundary Challenge"
        self._task_index = 4
        self._running = True

        directions = [2 * math.pi * i / 8 for i in range(8)]
        center = np.array([self.w_env / 2, self.h_env / 2], dtype=np.float64)
        max_radius = min(self.w_env, self.h_env) / 2 - 1.0
        error_threshold = 1.5
        hold_loss_time = 0.5
        results = {
            'reachable_radii': [],
            'reachable_area': 0.0,
            'directional_asymmetry': 0.0,
            'boundary_control_times': [],
            'boundary_violation_count': 0,
            'recovery_times': [],
            'min_x': 999.0,
            'max_x': 0.0,
            'min_y': 999.0,
            'max_y': 0.0,
            'vel_list': [],
        }

        frame, _, _ = self._get_frame_and_positions()
        if frame is not None:
            h_px, w_px = frame.shape[:2]
        else:
            w_px, h_px = self.w_px, self.h_px
        self._countdown()

        virtual_target_pixel = np.array([w_px / 2, h_px / 2], dtype=np.float64)
        last_control_time = time.time()
        last_hand_env = None
        last_perception_time = time.time()

        for idx, angle in enumerate(directions):
            if not self._running:
                break
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            direction_start = time.time()
            loss_start = None
            reached_radius = 0.0
            control_time = 0.0
            while self._running and reached_radius < max_radius:
                loop_start = time.time()
                elapsed = loop_start - direction_start
                reached_radius = min(max_radius, elapsed * 0.8)
                target_env = center + direction * reached_radius
                target_env[0] = np.clip(target_env[0], 1.0, self.w_env - 1.0)
                target_env[1] = np.clip(target_env[1], 1.0, self.h_env - 1.0)

                if self.simulate:
                    self._sim_hand_pos = self._sim_hand_pos + self.safe_normalize(target_env - self._sim_hand_pos) * 0.2

                frame, hand_env, _ = self._get_frame_and_positions()
                if frame is not None:
                    h_px, w_px = frame.shape[:2]
                if hand_env is not None:
                    hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env
                    results['min_x'] = min(results['min_x'], float(hand_env[0]))
                    results['max_x'] = max(results['max_x'], float(hand_env[0]))
                    results['min_y'] = min(results['min_y'], float(hand_env[1]))
                    results['max_y'] = max(results['max_y'], float(hand_env[1]))
                    if last_hand_env is not None:
                        vel = np.linalg.norm(hand_env - last_hand_env) / max(loop_start - last_perception_time, 0.01)
                        results['vel_list'].append(float(vel))
                    last_hand_env = hand_env
                    last_perception_time = loop_start
                    error = np.linalg.norm(hand_env - target_env)
                    if error <= error_threshold:
                        control_time += self.target_dt
                        loss_start = None
                    elif loss_start is None:
                        loss_start = loop_start
                    elif loop_start - loss_start >= hold_loss_time:
                        results['boundary_violation_count'] += 1
                        break

                virtual_target_pixel = self._step_virtual_target(virtual_target_pixel, target_env, w_px, h_px)
                control_now = time.time()
                self._send_robot_to_pixel(virtual_target_pixel, dt=min(max(control_now - last_control_time, 0.01), 0.2))
                last_control_time = control_now
                self._update_progress("Adaptive Boundary Challenge", 4, (idx + reached_radius / max_radius) / len(directions), f"边界方向 {idx + 1}/{len(directions)}")

                sleep_time = self.target_dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            results['reachable_radii'].append(float(reached_radius))
            results['boundary_control_times'].append(float(control_time))

        radii = results['reachable_radii']
        if radii:
            area = 0.0
            for i, radius in enumerate(radii):
                next_radius = radii[(i + 1) % len(radii)]
                area += 0.5 * radius * next_radius * math.sin(2 * math.pi / len(radii))
            results['reachable_area'] = float(area)
            results['directional_asymmetry'] = float((max(radii) - min(radii)) / max(max(radii), 1e-6))
        if results['min_x'] == 999.0:
            results['min_x'] = results['max_x'] = center[0]
            results['min_y'] = results['max_y'] = center[1]
        return results

    def run_boundary(self) -> Dict:
        return self.run_adaptive_boundary_challenge()

    def run_rhythmic_switching(self) -> Dict:
        """Run Rhythmic Switching between left and right targets."""
        self._current_task = "Rhythmic Switching"
        self._task_index = 5
        self._running = True

        beat_interval = 1.5
        beat_count = 16
        target_radius = 1.0
        time_window = 0.4
        left_target = np.array([self.w_env / 2 - 3.0, self.h_env / 2], dtype=np.float64)
        right_target = np.array([self.w_env / 2 + 3.0, self.h_env / 2], dtype=np.float64)
        targets = [left_target, right_target]
        results = {
            'beat_times': [],
            'target_sequence': [],
            'response_times': [],
            'timing_errors': [],
            'correct_count': 0,
            'early_count': 0,
            'late_count': 0,
            'miss_count': 0,
            'rhythm_variability': None,
        }

        frame, _, _ = self._get_frame_and_positions()
        if frame is not None:
            h_px, w_px = frame.shape[:2]
        else:
            w_px, h_px = self.w_px, self.h_px
        self._countdown()

        virtual_target_pixel = np.array([w_px / 2, h_px / 2], dtype=np.float64)
        last_control_time = time.time()
        task_start = time.time()
        last_response_target = None

        for beat_idx in range(beat_count):
            if not self._running:
                break
            target = targets[beat_idx % 2]
            target_name = 'L' if beat_idx % 2 == 0 else 'R'
            beat_time = beat_idx * beat_interval
            beat_abs = task_start + beat_time
            results['beat_times'].append(float(beat_time))
            results['target_sequence'].append(target_name)
            response_time = None

            while self._running and time.time() < beat_abs + beat_interval:
                loop_start = time.time()
                elapsed = loop_start - task_start

                if self.simulate:
                    self._sim_hand_pos = self._sim_hand_pos + self.safe_normalize(target - self._sim_hand_pos) * 0.25

                frame, hand_env, _ = self._get_frame_and_positions()
                if frame is not None:
                    h_px, w_px = frame.shape[:2]
                if hand_env is not None:
                    hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env
                    if response_time is None and last_response_target != target_name and np.linalg.norm(hand_env - target) <= target_radius:
                        response_time = elapsed
                        last_response_target = target_name

                virtual_target_pixel = self._step_virtual_target(virtual_target_pixel, target, w_px, h_px)
                control_now = time.time()
                self._send_robot_to_pixel(virtual_target_pixel, dt=min(max(control_now - last_control_time, 0.01), 0.2))
                last_control_time = control_now
                self._update_progress("Rhythmic Switching", 5, (beat_idx + min((elapsed - beat_time) / beat_interval, 1.0)) / beat_count, f"节律切换 {beat_idx + 1}/{beat_count} -> {target_name}")

                sleep_time = self.target_dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if response_time is None:
                results['response_times'].append(None)
                results['timing_errors'].append(None)
                results['miss_count'] += 1
                continue
            timing_error = response_time - beat_time
            results['response_times'].append(float(response_time))
            results['timing_errors'].append(float(timing_error))
            if abs(timing_error) <= time_window:
                results['correct_count'] += 1
            elif timing_error < -time_window:
                results['early_count'] += 1
            else:
                results['late_count'] += 1

        valid_errors = [e for e in results['timing_errors'] if e is not None]
        if valid_errors:
            results['rhythm_variability'] = float(np.std(valid_errors))
        return results

    def run_mirror_mapping_reach(self) -> Dict:
        """Run Mirror Mapping Reach with left/right mirrored cue-response zones."""
        self._current_task = "Mirror Mapping Reach"
        self._task_index = 6
        self._running = True

        y_offsets = [1.8, 0.6, -0.6, -1.8]
        left_x = self.w_env / 2 - 3.0
        right_x = self.w_env / 2 + 3.0
        cue_zones = []
        for y_offset in y_offsets:
            cue_zones.append(np.array([left_x, self.h_env / 2 + y_offset], dtype=np.float64))
            cue_zones.append(np.array([right_x, self.h_env / 2 + y_offset], dtype=np.float64))
        target_radius = 1.0
        max_trial_time = 5.0
        results = {
            'cue_zones': [],
            'response_zones': [],
            'successes': [],
            'wrong_side_count': 0,
            'wrong_target_count': 0,
            'timeouts': 0,
            'reaction_times': [],
            'movement_times': [],
            'spatial_errors': [],
            'path_efficiencies': [],
        }

        frame, _, _ = self._get_frame_and_positions()
        if frame is not None:
            h_px, w_px = frame.shape[:2]
        else:
            w_px, h_px = self.w_px, self.h_px
        self._countdown()

        virtual_target_pixel = np.array([w_px / 2, h_px / 2], dtype=np.float64)
        last_control_time = time.time()
        center = np.array([self.w_env / 2, self.h_env / 2], dtype=np.float64)

        for trial_idx, cue in enumerate(cue_zones):
            if not self._running:
                break
            response = np.array([self.w_env - cue[0], cue[1]], dtype=np.float64)
            results['cue_zones'].append(cue.tolist())
            results['response_zones'].append(response.tolist())
            self._moveto_sprint_target(cue, w_px, h_px)
            trial_start = time.time()
            movement_onset = None
            last_hand_env = center.copy()
            actual_path = 0.0
            final_error = float(np.linalg.norm(response - center))
            success = False
            wrong_side = False
            wrong_target = False

            while self._running and (time.time() - trial_start) < max_trial_time:
                loop_start = time.time()
                elapsed = loop_start - trial_start

                if self.simulate:
                    self._sim_hand_pos = self._sim_hand_pos + self.safe_normalize(response - self._sim_hand_pos) * 0.25

                frame, hand_env, _ = self._get_frame_and_positions()
                if frame is not None:
                    h_px, w_px = frame.shape[:2]
                if hand_env is not None:
                    hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env
                    step_dist = float(np.linalg.norm(hand_env - last_hand_env))
                    actual_path += step_dist
                    if movement_onset is None and step_dist / max(time.time() - last_control_time, 0.01) > 0.3:
                        movement_onset = elapsed
                    final_error = float(np.linalg.norm(hand_env - response))
                    if np.linalg.norm(hand_env - cue) <= target_radius:
                        wrong_side = True
                    if final_error <= target_radius:
                        success = True
                        break
                    other_distances = [np.linalg.norm(hand_env - other) for other in cue_zones if not np.allclose(other, cue) and not np.allclose(other, response)]
                    if other_distances and min(other_distances) <= target_radius:
                        wrong_target = True
                    last_hand_env = hand_env

                virtual_target_pixel = self._step_virtual_target(virtual_target_pixel, cue, w_px, h_px)
                control_now = time.time()
                self._send_robot_to_pixel(virtual_target_pixel, dt=min(max(control_now - last_control_time, 0.01), 0.2))
                last_control_time = control_now
                self._update_progress("Mirror Mapping Reach", 6, (trial_idx + min(elapsed / max_trial_time, 1.0)) / len(cue_zones), f"镜像到达 {trial_idx + 1}/{len(cue_zones)}")

                sleep_time = self.target_dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            movement_time = time.time() - trial_start
            results['successes'].append(bool(success))
            if wrong_side:
                results['wrong_side_count'] += 1
            if wrong_target:
                results['wrong_target_count'] += 1
            if not success:
                results['timeouts'] += 1
            results['reaction_times'].append(float(movement_onset if movement_onset is not None else max_trial_time))
            results['movement_times'].append(float(movement_time))
            results['spatial_errors'].append(float(final_error))
            shortest_path = float(np.linalg.norm(response - center))
            results['path_efficiencies'].append(float(shortest_path / max(actual_path, shortest_path, 1e-6)))

        return results

    def run_all(self, task_order: List[str] = None, session_id: int = None) -> EvalResult:
        """Run all evaluation tasks."""
        if task_order is None:
            task_order = EVALUATION_TASK_KEYS

        result = EvalResult()
        self._running = True

        if session_id is not None:
            self._start_video_recording(session_id)

        self._move_to_center()

        task_map = {
            'rapid_reach': self.run_rapid_reach,
            'continuous_tracking': self.run_continuous_tracking,
            'moving_target_interception': self.run_moving_target_interception,
            'adaptive_boundary_challenge': self.run_adaptive_boundary_challenge,
            'rhythmic_switching': self.run_rhythmic_switching,
            'mirror_mapping_reach': self.run_mirror_mapping_reach,
            'sprint': self.run_rapid_reach,
            'tracking': self.run_continuous_tracking,
            'boundary': self.run_adaptive_boundary_challenge,
            'league': self.run_league,
        }

        for task_name in task_order:
            if not self._running:
                break

            task_key = LEGACY_TASK_KEY_MAP.get(task_name.lower(), task_name.lower())
            task_func = task_map.get(task_name.lower()) or task_map.get(task_key)
            if task_func:
                task_result = task_func()
                if task_name.lower() == 'league':
                    result.legacy_league = task_result
                elif hasattr(result, task_key):
                    setattr(result, task_key, task_result)

            if self._running:
                self._move_to_center()

        self._stop_video_recording()

        return result

    def stop(self):
        """Stop the current evaluation."""
        self._running = False
        self._stop_video_recording()