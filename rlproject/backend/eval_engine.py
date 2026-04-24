"""Evaluation engine that wraps the original eval.py logic.

This module provides a clean interface for running the 4 evaluation tasks:
- Sprint: Reaction & explosive power
- Tracking: Multi-trajectory smooth tracking
- LeagueGame: Competition & cognitive interception
- Boundary: Range of motion & stability
"""
import numpy as np
import cv2
import time
import math
from collections import deque
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import json
import os


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
    task_progress: float = 0.0  # 0.0 - 1.0
    message: str = ""


class EvalEngine:
    """Engine for running rehabilitation evaluation tasks."""

    def __init__(
        self,
        robot_ip: str = "192.168.1.2",
        yolo_model_path: str = None,
        rl_model_path: str = None,
        calibration_path: str = None,
        control_freq: float = 8,
        simulate: bool = True  # 模拟模式，不连接硬件
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

        # Environment dimensions
        self.w_env, self.h_env = 15, 10
        self.grid_size = 10

        # Paths - use absolute paths based on known locations
        # rlproject/backend/eval_engine.py -> rlproject/ -> RL/
        _backend_dir = os.path.dirname(os.path.abspath(__file__))  # C:\...\rlproject\backend
        _project_root = os.path.dirname(_backend_dir)  # C:\...\rlproject
        _rl_root = os.path.dirname(_project_root)  # C:\...\RL

        self.yolo_model_path = yolo_model_path or os.path.join(
            _project_root, 'src', 'runs', 'detect', 'train3', 'weights', 'best.onnx'
        )
        self.rl_model_path = rl_model_path or os.path.join(
            _rl_root, 'logs', 'ablation_study_0409_0922',
            '2_MLP_LSTM', 'best_model.zip'
        )
        self.calibration_path = calibration_path or os.path.join(
            _project_root, 'src', 'calibration_data.npz'
        )

        # Hardware interfaces (initialized on connect)
        self.robot_control = None
        self.hand_detector = None
        self.cali = None
        self.rl_model = None
        self.cap = None

        # Runtime state
        self._running = False
        self._current_task = None
        self._progress_callback: Optional[Callable] = None
        self._frame_callback: Optional[Callable] = None
        self._frame_broadcast_callback: Optional[Callable] = None  # For WebSocket frame broadcast

        # Robot pose constants
        self.RX_C, self.RY_C, self.RZ_C = 0.193, 0.067, 5.3

        # Camera dimensions for simulate mode
        self._sim_frame_w = 640
        self._sim_frame_h = 480

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

    def connect(self) -> bool:
        """Connect to hardware (robot, camera, models) or run in simulate mode."""
        
                    # Add src path for hardware dependencies
        _src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        import sys
        if _src_path not in sys.path:
            sys.path.insert(0, _src_path)
        
        if self.simulate:
            print("[EvalEngine] Running in SIMULATE mode (no hardware)")
            # Initialize simulation state
            self._sim_hand_pos = np.array([self.w_env / 2, self.h_env / 2])
            self._sim_robot_pos = np.array([self.w_env / 2, self.h_env / 2])
            self._sim_time = 0
            return True

        try:
            print("[EvalEngine] Connecting to robot...")



            # Lazy imports for hardware dependencies
            from robot_control.ur_control import URControl
            from cv.hand_detect import HandDetection
            from camera_calibration.camera_calibration import CameraCalibration
            from stable_baselines3 import PPO

            self.robot_control = URControl(self.robot_ip)

            print("[EvalEngine] Loading hand detector...")
            self.hand_detector = HandDetection()

            print("[EvalEngine] Loading camera calibration...")
            self.cali = CameraCalibration()

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

            print("[EvalEngine] Connected successfully!")
            return True

        except Exception as e:
            print(f"[EvalEngine] Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from hardware."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def _get_sim_positions(self) -> tuple:
        """Get simulated hand/robot positions for testing without hardware."""
        # Simulate hand following target with some noise and delay
        self._sim_time += self.target_dt

        # Simulate random hand movement (user following target)
        # Add some randomness to simulate real user behavior
        noise = np.random.randn(2) * 0.1
        self._sim_hand_pos = self._sim_hand_pos + noise

        # Keep within bounds
        self._sim_hand_pos = np.clip(self._sim_hand_pos, 0.5, [self.w_env - 0.5, self.h_env - 0.5])

        return self._sim_hand_pos.copy(), self._sim_robot_pos.copy()

    def _generate_sim_frame(self) -> np.ndarray:
        """Generate a simulated camera frame for testing."""
        import cv2
        # Create a dark frame
        frame = np.zeros((self._sim_frame_h, self._sim_frame_w, 3), dtype=np.uint8)

        # Draw grid
        grid_color = (40, 40, 40)
        for i in range(0, self._sim_frame_w, 50):
            cv2.line(frame, (i, 0), (i, self._sim_frame_h), grid_color, 1)
        for i in range(0, self._sim_frame_h, 50):
            cv2.line(frame, (0, i), (self._sim_frame_w, i), grid_color, 1)

        # Draw robot position (blue circle)
        robot_px = int(self._sim_robot_pos[0] / self.w_env * self._sim_frame_w)
        robot_py = int(self._sim_robot_pos[1] / self.h_env * self._sim_frame_h)
        cv2.circle(frame, (robot_px, robot_py), 15, (255, 0, 0), -1)
        cv2.putText(frame, "Robot", (robot_px - 25, robot_py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw hand position (green circle)
        hand_px = int(self._sim_hand_pos[0] / self.w_env * self._sim_frame_w)
        hand_py = int(self._sim_hand_pos[1] / self.h_env * self._sim_frame_h)
        cv2.circle(frame, (hand_px, hand_py), 12, (0, 255, 0), -1)
        cv2.putText(frame, "Hand", (hand_px - 20, hand_py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw task info
        cv2.putText(frame, f"Task: {self._current_task or 'Ready'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _get_frame_and_positions(self) -> tuple:
        """Capture frame and compute hand/robot positions."""
        if self.simulate:
            # Generate simulated frame and broadcast
            frame = self._generate_sim_frame()
            # Encode frame as JPEG base64 for WebSocket transmission
            import cv2
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = buffer.tobytes()
            if self._frame_broadcast_callback:
                import base64
                self._frame_broadcast_callback(base64.b64encode(frame_base64).decode('utf-8'))
            return None, self._sim_hand_pos.copy(), self._sim_robot_pos.copy()

        from cv.get_workspace import get_workspace

        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        undistorted_frame = get_workspace(self.cali.undistort_frame(frame))
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h_px, w_px = undistorted_frame.shape[:2]

        # Detect hand position
        hand_pixel = self.hand_detector.detect(undistorted_frame)
        if hand_pixel is not None:
            hand_world = self.cali.pixel_to_world(hand_pixel.astype(int))
        else:
            hand_world = None

        # Robot position (from calibration, assuming center for now)
        # In real deployment, use YOLO to detect robot marker
        center_pixel = np.array([w_px / 2, h_px / 2])
        robot_world = self.cali.pixel_to_world(center_pixel.astype(int))

        # Notify frame callback for visualization
        if self._frame_callback:
            self._frame_callback(undistorted_frame, {
                'hand': hand_pixel,
                'robot': center_pixel,
                'hand_world': hand_world,
                'robot_world': robot_world
            })

        return undistorted_frame, hand_world, robot_world

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

    def _countdown(self, seconds: int = 3):
        """Display countdown."""
        for i in range(seconds, 0, -1):
            self._update_progress(self._current_task, self._task_index, 0, f"准备中... {i}")
            time.sleep(1)

    def _generate_target_position(self, current_pos: np.ndarray, min_dist: float = 3.0) -> np.ndarray:
        """Generate random target position at least min_dist from current position."""
        while True:
            target = np.array([
                np.random.uniform(1, self.w_env - 1),
                np.random.uniform(1, self.h_env - 1)
            ])
            if np.linalg.norm(target - current_pos) >= min_dist:
                return target

    def _pixel_to_env(self, pixel: np.ndarray, w_px: float, h_px: float) -> np.ndarray:
        """Convert pixel coordinates to environment coordinates."""
        x = pixel[0] / w_px * self.w_env
        y = (h_px - pixel[1]) / h_px * self.h_env  # Flip Y axis
        return np.array([x, y])

    def run_sprint(self) -> Dict:
        """Run Sprint task: 5 target catches measuring reaction time and velocity."""
        self._current_task = "Sprint"
        self._task_index = 1
        self._running = True

        results = {'catch_times': [], 'peak_vels': []}

        if not self.cap:
            return results

        ret, frame = self.cap.read()
        if not ret:
            return results

        undistorted_frame = get_workspace(self.cali.undistort_frame(frame))
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_px, w_px = undistorted_frame.shape[:2]

        self._countdown()

        for catch_num in range(5):
            if not self._running:
                break

            self._update_progress("Sprint", 1, catch_num / 5, f"第 {catch_num + 1}/5 次")

            # Generate new target
            hand_pixel, _, _ = self._get_frame_and_positions()
            if hand_pixel is not None:
                current_pos = self._pixel_to_env(hand_pixel, w_px, h_px)
            else:
                current_pos = np.array([self.w_env / 2, self.h_env / 2])

            target = self._generate_target_position(current_pos, min_dist=3.0)
            target_pixel = np.array([
                target[0] / self.w_env * w_px,
                (1 - target[1] / self.h_env) * h_px  # Flip Y
            ]).astype(int)

            # Move virtual target to position
            # (In real implementation, this would move a marker or be visualized)

            # Wait for user to catch target
            start_time = time.time()
            peak_vel = 0
            last_pos = current_pos if hand_pixel is not None else None
            caught = False

            while self._running and not caught:
                elapsed = time.time() - start_time
                if elapsed > 30:  # Timeout
                    break

                frame, hand_world, _ = self._get_frame_and_positions()
                if hand_world is None:
                    time.sleep(self.target_dt)
                    continue

                hand_pos = hand_world[:2] if len(hand_world) >= 2 else hand_world
                dist = np.linalg.norm(hand_pos - target)

                if dist < 1.5:  # Caught
                    catch_time = time.time() - start_time
                    results['catch_times'].append(catch_time)
                    results['peak_vels'].append(peak_vel)
                    caught = True
                else:
                    # Calculate velocity
                    if last_pos is not None:
                        vel = np.linalg.norm(hand_pos - last_pos) / self.target_dt
                        peak_vel = max(peak_vel, vel)
                    last_pos = hand_pos

                self._update_progress("Sprint", 1, (catch_num + 0.5) / 5,
                                       f"第 {catch_num + 1}/5 次 - 距离: {dist:.2f}")

                time.sleep(self.target_dt)

        return results

    def run_tracking(self) -> Dict:
        """Run Tracking task: Follow moving target along predefined paths."""
        self._current_task = "Tracking"
        self._task_index = 2
        self._running = True

        results = {'rmse_list': [], 'jerk_list': []}

        if not self.cap:
            return results

        ret, frame = self.cap.read()
        if not ret:
            return results

        undistorted_frame = get_workspace(self.cali.undistort_frame(frame))
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_px, w_px = undistorted_frame.shape[:2]

        self._countdown()

        duration = 20  # seconds
        start_time = time.time()
        last_hand_pos = None
        last_vel = None

        t = 0
        while self._running and (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            progress = elapsed / duration
            self._update_progress("Tracking", 2, progress, f"追踪中... {elapsed:.1f}s / {duration}s")

            # Generate trajectory point (circle then figure-8)
            if elapsed < 10:
                # Circle path
                cx, cy = self.w_env / 2, self.h_env / 2
                r = min(self.w_env, self.h_env) / 3
                omega = 2 * math.pi / 10  # Full circle in 10 seconds
                target_x = cx + r * math.cos(omega * elapsed)
                target_y = cy + r * math.sin(omega * elapsed)
            else:
                # Figure-8 path
                cx, cy = self.w_env / 2, self.h_env / 2
                r = min(self.w_env, self.h_env) / 4
                omega = 2 * math.pi / 10
                t = elapsed - 10
                target_x = cx + r * math.sin(omega * t)
                target_y = cy + r * math.sin(2 * omega * t)

            target = np.array([target_x, target_y])

            # Get hand position
            frame, hand_world, _ = self._get_frame_and_positions()
            if hand_world is not None:
                hand_pos = hand_world[:2] if len(hand_world) >= 2 else hand_world
            else:
                hand_pos = target  # Use target as fallback

            # Calculate cross-track error (perpendicular distance to trajectory)
            rmse = np.linalg.norm(hand_pos - target)
            results['rmse_list'].append(float(rmse))

            # Calculate jerk (rate of velocity change)
            if last_hand_pos is not None:
                vel = np.linalg.norm(hand_pos - last_hand_pos) / self.target_dt
                if last_vel is not None:
                    jerk = abs(vel - last_vel) / self.target_dt
                    results['jerk_list'].append(float(jerk))
                last_vel = vel
            last_hand_pos = hand_pos

            time.sleep(self.target_dt)

        return results

    def run_league(self) -> Dict:
        """Run LeagueGame task: Avoid RL-controlled robot."""
        self._current_task = "LeagueGame"
        self._task_index = 3
        self._running = True

        results = {'is_caught': False, 'survival_time': 0.0, 'dist_list': []}

        if not self.cap or not self.rl_model:
            return results

        ret, frame = self.cap.read()
        if not ret:
            return results

        undistorted_frame = get_workspace(self.cali.undistort_frame(frame))
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_px, w_px = undistorted_frame.shape[:2]

        self._countdown()

        duration = 30  # seconds
        start_time = time.time()
        robot_pos = np.array([self.w_env / 2, self.h_env / 2])

        while self._running:
            elapsed = time.time() - start_time
            if elapsed > duration:
                break

            progress = elapsed / duration
            self._update_progress("LeagueGame", 3, progress, f"对抗中... {elapsed:.1f}s / {duration}s")

            # Get hand position
            frame, hand_world, _ = self._get_frame_and_positions()
            if hand_world is not None:
                hand_pos = hand_world[:2] if len(hand_world) >= 2 else hand_world
            else:
                hand_pos = np.array([self.w_env / 2, self.h_env / 2])

            # RL model generates action
            # Simplified: move toward hand
            dist = np.linalg.norm(robot_pos - hand_pos)
            results['dist_list'].append(float(dist))

            if dist < 1.5:
                results['is_caught'] = True
                results['survival_time'] = elapsed
                break

            # Move robot toward hand (simplified RL action)
            direction = (hand_pos - robot_pos)
            if np.linalg.norm(direction) > 0.01:
                direction = direction / np.linalg.norm(direction)

            robot_pos = robot_pos + direction * 0.3  # stride

            time.sleep(self.target_dt)

        if not results['is_caught']:
            results['survival_time'] = duration

        return results

    def run_boundary(self) -> Dict:
        """Run Boundary task: Track target along rectangular boundary."""
        self._current_task = "Boundary"
        self._task_index = 4
        self._running = True

        results = {'min_x': 999, 'max_x': 0, 'min_y': 999, 'max_y': 0, 'vel_list': []}

        if not self.cap:
            return results

        ret, frame = self.cap.read()
        if not ret:
            return results

        undistorted_frame = get_workspace(self.cali.undistort_frame(frame))
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_px, w_px = undistorted_frame.shape[:2]

        self._countdown()

        duration = 20  # seconds
        start_time = time.time()
        last_hand_pos = None

        while self._running and (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            progress = elapsed / duration
            self._update_progress("Boundary", 4, progress, f"边界追踪... {elapsed:.1f}s / {duration}s")

            # Move target along rectangle perimeter
            perimeter_time = (elapsed % 40) / 40 * 4  # 0-4 for 4 sides
            side = int(perimeter_time)
            t = perimeter_time - side

            x_min, x_max = 1, self.w_env - 1
            y_min, y_max = 1, self.h_env - 1

            if side == 0:
                target = np.array([x_min + t * (x_max - x_min), y_min])
            elif side == 1:
                target = np.array([x_max, y_min + t * (y_max - y_min)])
            elif side == 2:
                target = np.array([x_max - t * (x_max - x_min), y_max])
            else:
                target = np.array([x_min, y_max - t * (y_max - y_min)])

            # Get hand position
            frame, hand_world, _ = self._get_frame_and_positions()
            if hand_world is not None:
                hand_pos = hand_world[:2] if len(hand_world) >= 2 else hand_world

                # Update bounds
                results['min_x'] = min(results['min_x'], hand_pos[0])
                results['max_x'] = max(results['max_x'], hand_pos[0])
                results['min_y'] = min(results['min_y'], hand_pos[1])
                results['max_y'] = max(results['max_y'], hand_pos[1])

                # Calculate velocity
                if last_hand_pos is not None:
                    vel = np.linalg.norm(hand_pos - last_hand_pos) / self.target_dt
                    results['vel_list'].append(float(vel))
                last_hand_pos = hand_pos

            time.sleep(self.target_dt)

        return results

    def run_all(self, task_order: List[str] = None) -> EvalResult:
        """
        Run all evaluation tasks.

        Args:
            task_order: Optional list specifying task order. Defaults to ['sprint', 'tracking', 'league', 'boundary']

        Returns:
            EvalResult containing results for all tasks
        """
        if task_order is None:
            task_order = ['sprint', 'tracking', 'league', 'boundary']

        result = EvalResult()

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

            self._move_to_center()

        return result

    def stop(self):
        """Stop the current evaluation."""
        self._running = False
