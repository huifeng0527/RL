"""Evaluation engine that wraps the original eval.py logic.

This module provides a clean interface for running the 5 evaluation tasks:
- T1 Rapid Reach: reaction and large-range reaching
- T2 Continuous Tracking: dynamic tracking
- T3 Workspace Exploration: reachable workspace
- T4 Rhythmic Synchronization: temporal coordination
- T5 Constrained Line Tracing: fine path control
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
    ("workspace_exploration", "Workspace Exploration"),
    ("rhythmic_synchronization", "Rhythmic Synchronization"),
    ("constrained_line_tracing", "Constrained Line Tracing"),
]
EVALUATION_TASK_KEYS = [task[0] for task in EVALUATION_TASKS]
EVALUATION_TASK_NAMES = dict(EVALUATION_TASKS)
LEGACY_TASK_KEY_MAP = {
    "sprint": "rapid_reach",
    "tracking": "continuous_tracking",
    "boundary": "workspace_exploration",
    "adaptive_boundary_challenge": "workspace_exploration",
    "rhythmic_switching": "rhythmic_synchronization",
    "line_tracing": "constrained_line_tracing",
}


@dataclass
class EvalResult:
    """Container for evaluation results."""
    rapid_reach: Optional[Dict] = None
    continuous_tracking: Optional[Dict] = None
    workspace_exploration: Optional[Dict] = None
    rhythmic_synchronization: Optional[Dict] = None
    constrained_line_tracing: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "rapid_reach": self.rapid_reach,
            "continuous_tracking": self.continuous_tracking,
            "workspace_exploration": self.workspace_exploration,
            "rhythmic_synchronization": self.rhythmic_synchronization,
            "constrained_line_tracing": self.constrained_line_tracing,
        }


@dataclass
class TaskProgress:
    """Track evaluation progress."""
    current_task: str = ""
    task_index: int = 0
    total_tasks: int = 5
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
        self.robot_ip = robot_ip
        self.control_freq = control_freq
        self.target_dt = 1.0 / control_freq
        self.simulate = simulate

        self.w_env, self.h_env = 15, 10
        self.grid_size = 10
        self.MAX_SAFE_STRIDE = 0.6

        _backend_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_backend_dir)
        _rl_root = os.path.dirname(_project_root)

        self.yolo_model_path = yolo_model_path or os.path.join(
            _project_root, 'src', 'runs', 'detect', 'train3', 'weights', 'best.onnx'
        )

        self.robot_control = None
        self.hand_detector = None
        self.cali = None
        self.cap = None

        self.w_px, self.h_px = 2592, 1944

        self._running = False
        self._current_task = None
        self._progress_callback: Optional[Callable] = None
        self._frame_callback: Optional[Callable] = None
        self._frame_broadcast_callback: Optional[Callable] = None

        self.RX_C, self.RY_C, self.RZ_C, self.z = 0.107, 0.049, 4.747, 0.112

        self._fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        self._frame_broadcast_interval = 1.0 / 12.0
        self._last_frame_broadcast_time = 0.0

        self._video_writer = None
        self._video_path = None
        self._is_recording = False

        self._sim_hand_pos = None
        self._sim_robot_pos = None
        self._sim_frame_w = 640
        self._sim_frame_h = 480

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
        self._progress_callback = callback

    def set_frame_callback(self, callback: Callable[[np.ndarray, Dict], None]):
        self._frame_callback = callback

    def set_frame_broadcast_callback(self, callback: Callable[[str], None]):
        self._frame_broadcast_callback = callback

    def _update_progress(self, task: str, index: int, progress: float, message: str = ""):
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
        if self._is_recording:
            return
        videos_dir = os.path.join(os.path.dirname(_backend_dir), 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._video_path = os.path.join(videos_dir, f"eval_session_{session_id}_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(
            self._video_path, fourcc, 25.0, (self.w_px, self.h_px)
        )
        self._is_recording = True
        print(f"[EvalEngine] Started recording to {self._video_path}")

    def _stop_video_recording(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._is_recording = False
        print(f"[EvalEngine] Stopped recording")

    def get_video_path(self) -> Optional[str]:
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
        if self.simulate:
            print("[EvalEngine] Running in SIMULATE mode (no hardware)")
            self._sim_hand_pos = np.array([self.w_env / 2, self.h_env / 2])
            self._sim_robot_pos = np.array([self.w_env / 2, self.h_env / 2])
            self.w_px, self.h_px = self._sim_frame_w, self._sim_frame_h
            return True

        try:
            print("[EvalEngine] Connecting to robot...")

            from robot_control.ur_control import URControl
            from cv.hand_detect import HandDetection
            from camera_calibration.camera_calibration import CameraCalibration

            self.robot_control = URControl(self.robot_ip)
            

            print("[EvalEngine] Loading hand detector...")
            self.hand_detector = HandDetection()

            print("[EvalEngine] Loading camera calibration...")
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.cali = CameraCalibration(
                calibration_matrix_path=os.path.join(_project_root, 'src', 'camera_calibration', 'calibration_data.npz'),
                homography_matrix_path=os.path.join(_project_root, 'src', 'camera_calibration', 'Homography_matrix.npy')
            )

            print("[EvalEngine] Opening camera...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

            if not self.cap.isOpened():
                raise RuntimeError("Camera not accessible")

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
        if self.simulate:
            frame = self._generate_sim_frame()
            self._broadcast_frame(frame)
            return frame, self._sim_hand_pos.copy(), self._sim_robot_pos.copy()

        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        undistorted_frame = get_workspace(self.cali.undistort_frame(frame))
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h_px, w_px = undistorted_frame.shape[:2]

        annotated_frame, hand_positions = self.hand_detector.process_frame(undistorted_frame)
        if hand_positions:
            hand_pixel = np.array(hand_positions[0], dtype=np.float64)
            hand_env = np.array([hand_pixel[0] * self.w_env / w_px, hand_pixel[1] * self.h_env / h_px], dtype=np.float64)
        else:
            hand_pixel = None
            hand_env = None

        *position_robot_world, _, _, _, _ = self.robot_control.get_robot_pose()
        real_robot_pixel = self.cali.world_to_pixel(position_robot_world)
        robot_world = np.array([position_robot_world[0], position_robot_world[1]])
        robot_env = np.array([real_robot_pixel[0] * self.w_env / self.w_px, real_robot_pixel[1] * self.h_env / self.h_px], dtype=np.float64)

        if self._frame_callback:
            self._frame_callback(undistorted_frame, {
                'hand': hand_pixel,
                'robot': real_robot_pixel,
                'hand_env': hand_env,
                'robot_env': robot_env,
                'hand_world': hand_env,
                'robot_world': robot_world
            })
        self._broadcast_frame(undistorted_frame)

        self._frame_count += 1
        elapsed = time.time() - self._fps_start_time
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = time.time()

        if self._is_recording and self._video_writer is not None:
            self._video_writer.write(undistorted_frame)

        return undistorted_frame, hand_env, robot_env

    def _generate_sim_frame(self) -> np.ndarray:
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
        if self.simulate:
            self._sim_robot_pos = np.array([self.w_env / 2, self.h_env / 2])
            return
        print(22222222222222222)
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
        if self.simulate:
            x = target_pixel[0] / self.w_px * self.w_env
            y = target_pixel[1] / self.h_px * self.h_env
            self._sim_robot_pos = np.array([x, y])
            return

        last_pixel = getattr(self, '_last_target_pixel', None)
        self._last_target_pixel = target_pixel.copy()

        if last_pixel is not None:
            pixel_jump = np.linalg.norm(target_pixel - last_pixel)
            jump_threshold = self.MAX_SAFE_STRIDE * (self.w_px / self.w_env) * 3

            if pixel_jump > jump_threshold:
                target_world = self.cali.pixel_to_world(target_pixel.astype(int))
                target_pose = [target_world[0], target_world[1], self.z, self.RX_C, self.RY_C, self.RZ_C]
                self.robot_control.rtde_c.moveL(target_pose, 0.2, 0.2, asynchronous=False)
                return

        target_world = self.cali.pixel_to_world(target_pixel.astype(int))
        target_pose = [target_world[0], target_world[1], self.z, self.RX_C, self.RY_C, self.RZ_C]
        self.robot_control.servo_robot(target_pose, dt=dt)

    def _countdown(self, seconds: int = 3):
        for i in range(seconds, 0, -1):
            self._update_progress(self._current_task, self._task_index, 0, f"准备中... {i}")
            time.sleep(1)

    def _generate_target_position(self, current_pos: np.ndarray, min_dist: float = 3.0) -> np.ndarray:
        while True:
            target = np.array([
                np.random.uniform(2, self.w_env - 2),
                np.random.uniform(3, self.h_env - 3)
            ])
            if np.linalg.norm(target - current_pos) >= min_dist:
                return target

    def safe_normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return np.zeros_like(v)
        return v / norm

    def _moveto_sprint_target(self, target_env: np.ndarray, w_px: int, h_px: int):
        if self.simulate:
            self._sim_robot_pos = target_env.copy()
            return

        target_pixel = self._env_to_pixel(target_env, w_px, h_px)
        target_pixel[0] = np.clip(target_pixel[0], 50, w_px - 50)
        target_pixel[1] = np.clip(target_pixel[1], 50, h_px - 50)

        target_world = self.cali.pixel_to_world(target_pixel.astype(int))
        target_pose = [
            target_world[0], target_world[1], self.z,
            self.RX_C, self.RY_C, self.RZ_C
        ]

        self.robot_control.rtde_c.servoStop()
        self.robot_control.rtde_c.moveL(target_pose, 3, 1, asynchronous=False)
        print(f"  [Sprint] moveL -> target_env={target_env}, world={target_world[:2]}")

    # =========================================================================
    # T1: Rapid Reach
    # =========================================================================
    def run_rapid_reach(self) -> Dict:
        self._current_task = "Rapid Reach"
        self._task_index = 1
        self._running = True

        trial_count = 8
        target_radius = 2.0
        max_trial_time = 6.0
        loop_dt = 1.0 / 20
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
            sleep_time = loop_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        return results

    def run_sprint(self) -> Dict:
        return self.run_rapid_reach()

    # =========================================================================
    # T2: Continuous Tracking
    # =========================================================================
    def run_continuous_tracking(self) -> Dict:
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

            frame, hand_env, _ = self._get_frame_and_positions()
            if frame is not None:
                h_px, w_px = frame.shape[:2]

            if hand_env is not None:
                hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env
            else:
                hand_env = None

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

            path_points = self.ideal_trajectories[shape_name]
            if hand_env is not None:
                distances_to_path = np.linalg.norm(path_points - hand_env, axis=1)
                cross_track_error = np.min(distances_to_path)
                results['rmse_list'].append(float(cross_track_error))
                if not results['trajectory_names'] or results['trajectory_names'][-1] != shape_name:
                    results['trajectory_names'].append(shape_name)

                dt_perception = max(t_now - last_perception_time, 0.01)
                vel = np.linalg.norm(hand_env - last_hand_env) / dt_perception if last_hand_env is not None else 0
                if last_vel is not None:
                    jerk = abs(vel - last_vel) / dt_perception
                    results['jerk_list'].append(float(jerk))
                last_vel = vel
                last_hand_env = hand_env
                last_perception_time = t_now

            desired_virtual_target = np.array([
                target_env[0] * w_px / self.w_env,
                target_env[1] * h_px / self.h_env
            ])

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

    # =========================================================================
    # T3: Workspace Exploration
    # =========================================================================
    def run_workspace_exploration(self) -> Dict:
        """T3: Robot traces rectangular perimeter; measures hand range of motion and stability."""
        self._current_task = "Workspace Exploration"
        self._task_index = 3
        self._running = True

        duration = 20.0
        margin = 1.0
        wx, wy = self.w_env - 2 * margin, self.h_env - 2 * margin
        perimeter = 2 * wx + 2 * wy

        results = {
            'min_x': 999.0, 'max_x': 0.0,
            'min_y': 999.0, 'max_y': 0.0,
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
        task_start = time.time()

        while self._running:
            loop_start = time.time()
            elapsed = loop_start - task_start
            if elapsed >= duration:
                break

            progress = (elapsed / duration) * perimeter
            if progress < wx:
                tx, ty = margin + progress, margin
            elif progress < wx + wy:
                tx, ty = self.w_env - margin, margin + (progress - wx)
            elif progress < 2 * wx + wy:
                tx, ty = self.w_env - margin - (progress - wx - wy), self.h_env - margin
            else:
                tx, ty = margin, self.h_env - margin - (progress - 2 * wx - wy)
            target_env = np.array([tx, ty])

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

            virtual_target_pixel = self._step_virtual_target(virtual_target_pixel, target_env, w_px, h_px)
            control_now = time.time()
            self._send_robot_to_pixel(virtual_target_pixel, dt=min(max(control_now - last_control_time, 0.01), 0.2))
            last_control_time = control_now
            self._update_progress("Workspace Exploration", 3, elapsed / duration, f"空间探索 {elapsed:.1f}s / {duration}s")

            sleep_time = self.target_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if results['min_x'] == 999.0:
            center = np.array([self.w_env / 2, self.h_env / 2])
            results['min_x'] = results['max_x'] = center[0]
            results['min_y'] = results['max_y'] = center[1]
        return results

    def run_adaptive_boundary_challenge(self) -> Dict:
        return self.run_workspace_exploration()

    def run_boundary(self) -> Dict:
        return self.run_workspace_exploration()

    # =========================================================================
    # T4: Rhythmic Synchronization
    # =========================================================================
    def run_rhythmic_synchronization(self) -> Dict:
        self._current_task = "Rhythmic Synchronization"
        self._task_index = 4
        self._running = True

        beat_interval = 1.5
        beat_count = 16
        target_radius = 1.5
        time_window = 0.4
        loop_dt = 1.0 / 20
        left_target = np.array([self.w_env / 2 - 3.0, self.h_env / 2], dtype=np.float64)
        center_target = np.array([self.w_env / 2, self.h_env / 2], dtype=np.float64)
        right_target = np.array([self.w_env / 2 + 3.0, self.h_env / 2], dtype=np.float64)
        target_positions = [left_target, center_target, right_target]
        target_names = ['L', 'C', 'R']
        rng = np.random.default_rng()
        results = {
            'beat_times': [],
            'target_sequence': [],
            'target_positions': [],
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
        current_target_index = 1

        for beat_idx in range(beat_count):
            if not self._running:
                break
            if current_target_index == 0:
                step = 1
            elif current_target_index == len(target_positions) - 1:
                step = -1
            else:
                step = int(rng.choice([-1, 1]))
            current_target_index += step
            target_name = target_names[current_target_index]
            target = target_positions[current_target_index]
            beat_time = beat_idx * beat_interval
            beat_abs = task_start + beat_time
            results['beat_times'].append(float(beat_time))
            results['target_sequence'].append(target_name)
            results['target_positions'].append(target.tolist())
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
                        break

                virtual_target_pixel = self._step_virtual_target(virtual_target_pixel, target, w_px, h_px)
                control_now = time.time()
                self._send_robot_to_pixel(virtual_target_pixel, dt=min(max(control_now - last_control_time, 0.01), 0.2))
                last_control_time = control_now
                self._update_progress("Rhythmic Synchronization", 4, (beat_idx + min((elapsed - beat_time) / beat_interval, 1.0)) / beat_count, f"节律同步 {beat_idx + 1}/{beat_count} -> {target_name}")

                sleep_time = loop_dt - (time.time() - loop_start)
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

    def run_rhythmic_switching(self) -> Dict:
        return self.run_rhythmic_synchronization()

    # =========================================================================
    # T5: Constrained Line Tracing
    # =========================================================================
    def _detect_marker_line(self, frame: np.ndarray) -> tuple:
        """Detect a black marker line drawn on the table.

        Returns (start_env, end_env, path_pixels, success).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None, False

        line_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(line_contour) < 500:
            return None, None, None, False

        points = line_contour.reshape(-1, 2).astype(np.float64)
        if len(points) < 2:
            return None, None, None, False

        idx_start = int(np.argmax(points[:, 0]))
        idx_end = int(np.argmin(points[:, 0]))
        p1 = points[idx_start]
        p2 = points[idx_end]

        h_px, w_px = frame.shape[:2]
        start_env = np.array([p1[0] * self.w_env / w_px, p1[1] * self.h_env / h_px])
        end_env = np.array([p2[0] * self.w_env / w_px, p2[1] * self.h_env / h_px])

        num_pts = max(10, int(np.linalg.norm(p1 - p2) / 20))
        xs = np.linspace(p1[0], p2[0], num_pts)
        ys = np.linspace(p1[1], p2[1], num_pts)
        path_pixels = np.column_stack([xs, ys])

        return start_env, end_env, path_pixels, True

    def _line_trace_metrics(self, point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple:
        segment = end - start
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq < 1e-8:
            return float(np.linalg.norm(point - start)), 0.0
        projection = float(np.dot(point - start, segment) / segment_len_sq)
        progress = min(max(projection, 0.0), 1.0)
        closest = start + progress * segment
        lateral_error = float(np.linalg.norm(point - closest))
        return lateral_error, progress

    def run_constrained_line_tracing(self) -> Dict:
        """T5: Detect black marker line on table, robot traces it, measure hand accuracy."""
        self._current_task = "Constrained Line Tracing"
        self._task_index = 5
        self._running = True

        tolerance = 0.8
        max_trial_time = 15.0
        target_radius = 0.8

        results = {
            'line_specs': [],
            'successes': [],
            'completion_times': [],
            'mean_lateral_errors': [],
            'max_lateral_errors': [],
            'off_line_rates': [],
            'path_smoothness': [],
            'path_lengths': [],
            'speed_accuracy_scores': [],
        }

        frame, _, _ = self._get_frame_and_positions()
        if frame is not None:
            h_px, w_px = frame.shape[:2]
        else:
            w_px, h_px = self.w_px, self.h_px

        # Detect the marker line
        if self.simulate:
            start_env = np.array([self.w_env * 0.2, self.h_env * 0.75], dtype=np.float64)
            end_env = np.array([self.w_env * 0.8, self.h_env * 0.25], dtype=np.float64)
            detected = True
        else:
            self._update_progress("Constrained Line Tracing", 5, 0, "正在检测马克笔线条...")
            for _ in range(5):
                ret, det_frame = self.cap.read()
                if ret:
                    undist = get_workspace(self.cali.undistort_frame(det_frame))
                    undist = cv2.rotate(undist, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    break
            else:
                undist = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            start_env, end_env, _, detected = self._detect_marker_line(undist)

        if not detected:
            self._update_progress("Constrained Line Tracing", 5, 0, "未检测到线条，请用黑色马克笔在桌面上画线")
            time.sleep(2)
            return results

        results['line_specs'].append({
            'name': 'Detected Line',
            'start': start_env.tolist(),
            'end': end_env.tolist(),
            'tolerance': tolerance,
        })

        # Draw detected line on frame for visualization
        vis_frame = undist if not self.simulate else self._generate_sim_frame()
        p1 = self._env_to_pixel(start_env, w_px, h_px).astype(int)
        p2 = self._env_to_pixel(end_env, w_px, h_px).astype(int)
        cv2.line(vis_frame, tuple(p1), tuple(p2), (0, 255, 0), 3)
        cv2.circle(vis_frame, tuple(p1), 8, (0, 0, 255), -1)
        cv2.circle(vis_frame, tuple(p2), 8, (0, 0, 255), -1)
        cv2.putText(vis_frame, "Detected Line", (p1[0], p1[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self._broadcast_frame(vis_frame)

        self._countdown()

        trial_start = time.time()
        last_hand_env = None
        last_vel = None
        last_perception_time = time.time()
        lateral_errors = []
        off_line_samples = 0
        sample_count = 0
        path_length = 0.0
        jerk_values = []
        completion_time = max_trial_time
        success = False

        virtual_target_pixel = np.array([w_px / 2, h_px / 2], dtype=np.float64)
        last_control_time = time.time()

        while self._running:
            loop_start = time.time()
            elapsed = loop_start - trial_start
            if elapsed >= max_trial_time:
                break

            alpha = min(elapsed / max_trial_time, 1.0)
            target_env = start_env * (1.0 - alpha) + end_env * alpha

            if self.simulate:
                self._sim_hand_pos = self._sim_hand_pos + self.safe_normalize(target_env - self._sim_hand_pos) * 0.22

            frame, hand_env, _ = self._get_frame_and_positions()
            if frame is not None:
                h_px, w_px = frame.shape[:2]
            if hand_env is not None:
                hand_env = hand_env[:2] if len(hand_env) >= 2 else hand_env
                lateral_error, progress = self._line_trace_metrics(hand_env, start_env, end_env)
                lateral_errors.append(lateral_error)
                sample_count += 1
                if lateral_error > tolerance:
                    off_line_samples += 1

                dt_perception = max(loop_start - last_perception_time, 0.01)
                if last_hand_env is not None:
                    step_dist = float(np.linalg.norm(hand_env - last_hand_env))
                    path_length += step_dist
                    vel = step_dist / dt_perception
                    if last_vel is not None:
                        jerk_values.append(float(abs(vel - last_vel) / dt_perception))
                    last_vel = vel
                last_hand_env = hand_env
                last_perception_time = loop_start

                if progress >= 0.98 and np.linalg.norm(hand_env - end_env) <= target_radius:
                    completion_time = elapsed
                    success = True
                    break

            virtual_target_pixel = self._step_virtual_target(virtual_target_pixel, target_env, w_px, h_px)
            control_now = time.time()
            self._send_robot_to_pixel(virtual_target_pixel, dt=min(max(control_now - last_control_time, 0.01), 0.2))
            last_control_time = control_now
            self._update_progress("Constrained Line Tracing", 5, alpha, f"直线描画 {elapsed:.1f}s / {max_trial_time}s")

            sleep_time = self.target_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        mean_error = float(np.mean(lateral_errors)) if lateral_errors else None
        max_error = float(np.max(lateral_errors)) if lateral_errors else None
        off_line_rate = float(off_line_samples / max(sample_count, 1))
        smoothness = float(np.mean(jerk_values)) if jerk_values else None
        speed_accuracy = None
        if mean_error is not None:
            speed_accuracy = float(completion_time * (1.0 + mean_error / max(tolerance, 1e-6)))

        results['successes'].append(bool(success))
        results['completion_times'].append(float(completion_time))
        results['mean_lateral_errors'].append(mean_error)
        results['max_lateral_errors'].append(max_error)
        results['off_line_rates'].append(off_line_rate)
        results['path_smoothness'].append(smoothness)
        results['path_lengths'].append(float(path_length))
        results['speed_accuracy_scores'].append(speed_accuracy)

        return results

    # =========================================================================
    # Helpers
    # =========================================================================
    def _env_to_pixel(self, pos_env: np.ndarray, w_px: int, h_px: int) -> np.ndarray:
        return np.array([
            pos_env[0] * w_px / self.w_env,
            pos_env[1] * h_px / self.h_env
        ], dtype=np.float64)

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

    # =========================================================================
    # Run All
    # =========================================================================
    def run_all(self, task_order: List[str] = None, session_id: int = None) -> EvalResult:
        if task_order is None:
            task_order = EVALUATION_TASK_KEYS

        result = EvalResult()
        self._running = True

        if session_id is not None:
            self._start_video_recording(session_id)

        self._move_to_center()
        print("\n[系统] 正在将机器人复位至桌面中心...")
        # time.sleep(30)

        task_map = {
            'rapid_reach': self.run_rapid_reach,
            'continuous_tracking': self.run_continuous_tracking,
            'workspace_exploration': self.run_workspace_exploration,
            'rhythmic_synchronization': self.run_rhythmic_synchronization,
            'constrained_line_tracing': self.run_constrained_line_tracing,
            'sprint': self.run_rapid_reach,
            'tracking': self.run_continuous_tracking,
            'boundary': self.run_workspace_exploration,
            'adaptive_boundary_challenge': self.run_workspace_exploration,
            'rhythmic_switching': self.run_rhythmic_synchronization,
            'line_tracing': self.run_constrained_line_tracing,
        }

        for task_name in task_order:
            if not self._running:
                break

            task_key = LEGACY_TASK_KEY_MAP.get(task_name.lower(), task_name.lower())
            task_func = task_map.get(task_name.lower()) or task_map.get(task_key)
            if task_func:
                task_result = task_func()
                if hasattr(result, task_key):
                    setattr(result, task_key, task_result)

            if self._running:
                self._move_to_center()

        self._stop_video_recording()

        return result

    def stop(self):
        self._running = False
        self._stop_video_recording()
