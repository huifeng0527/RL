import argparse
import importlib.util
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

cv2 = None
PPO = None
YOLO = None
CameraCalibration = None
DeploymentRolloutLogger = None
HandDetection = None
URControl = None
get_workspace = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TRAINING_SRC = REPO_ROOT / "src"
for path in [REPO_ROOT, SCRIPT_DIR, TRAINING_SRC]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.utils.cv_mpc_controller import ConstantVelocityMPCController


W_ENV = 15.0
H_ENV = 10.0
DEFAULT_CONTROL_FREQ = 20
DEFAULT_SERVO_FREQ = 125.0
DEFAULT_SERVO_LOOKAHEAD = 0.1
DEFAULT_SERVO_GAIN = 600
RX_C, RY_C, RZ_C,Z= 0.107, 0.049, 4.747,0.113
DEFAULT_STRIDE = 0.6
DEFAULT_MAX_SAFE_STRIDE = 1
WORKSPACE_MARGIN = 0.3
COMMON_SAFETY_GUARD = 0.2
MICROROBOT_MAX_AGE_S = 0.25
OBS_SCALAR_DIM = 12
MOTION_HISTORY_CHANNELS = 2
INTERACTION_HISTORY_CHANNELS = 8
VIRTUAL_HAND_HISTORY_LENGTH = 16
VIRTUAL_HAND_OBS_DIM = OBS_SCALAR_DIM + VIRTUAL_HAND_HISTORY_LENGTH * MOTION_HISTORY_CHANNELS
VIRTUAL_HAND_STRIDE_RANGE = (0.1, 0.4)
CONTROLLER_LABELS = {
    "league": "League RL",
    "cv_mpc": "CV-MPC",
}
CV_MPC_CONFIG = {
    "horizon": 5,
    "velocity_window": 3,
    "action_grid": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "discount": 0.95,
    "effort_weight": 0.01,
    "smoothness_weight": 0.03,
    "collision_penalty": 120.0,
    "oob_penalty": 240.0,
    "boundary_band": 1.0,
    "boundary_barrier_weight": 16.0,
    "collision_buffer": 1.0,
    "collision_barrier_weight": 8.0,
}

DEFAULT_POLICY_CANDIDATES = [
    REPO_ROOT / "logs" / "league_zpd35_55_noid_warm_entropy_10iter_r5m_h1m_gru_noaux" / "iteration_10" / "robot" / "robot" / "best_model.zip",
    REPO_ROOT / "logs" / "league_paper_gru_multistep_aux_pfsp_window_20iter" / "iteration_20" / "robot" / "robot" / "best_model.zip",
    REPO_ROOT / "rlproject" / "best_model.zip",
]
DEFAULT_HAND_MODEL_CANDIDATES = [
    REPO_ROOT / "logs" / "league_zpd35_55_noid_warm_entropy_10iter_r5m_h1m_gru_noaux" / "iteration_1" / "hand" / "hand" / "best_model.zip",
]
DEFAULT_VISION_MODEL = SCRIPT_DIR / "runs" / "detect" / "train3" / "weights" / "best.onnx"


@dataclass
class VisionResult:
    hand_positions: list
    hand_detected: bool
    microrobot_detected: bool
    robot_trajectory: list
    hand_env: np.ndarray
    microrobot_env: np.ndarray
    pixel_per_cm: float
    undistorted_frame: np.ndarray
    frame_id: int
    capture_t_perf: float
    processed_t_perf: float
    camera_dt_s: float | None
    camera_hz_inst: float | None


@dataclass(frozen=True)
class ServoTimingConfig:
    mode: str
    policy_hz: float
    servo_hz: float
    policy_period_s: float
    servo_period_s: float
    target_timeout_s: float
    max_speed_cm_s: float
    max_translation_per_tick_m: float
    lookahead_time_s: float
    gain: int


@dataclass
class ServoTarget:
    pose: np.ndarray
    policy_step: int
    sequence: int
    published_at_perf: float


@dataclass
class ServoSnapshot:
    servo_dt_s: float | None = None
    servo_loop_hz_inst: float | None = None
    target_age_s: float | None = None
    interpolation_phase: float = 0.0
    command_sequence: int = 0
    policy_step: int = -1
    commanded_pose: np.ndarray | None = None
    target_pose: np.ndarray | None = None
    deadline_overrun_s: float = 0.0
    deadline_overrun_count: int = 0
    watchdog_stopped: bool = False
    stop_reason: str = ""
    period_api_available: bool = False


class ServoLoopFailure(RuntimeError):
    pass


class ServoLoopSafetyStop(ServoLoopFailure):
    pass


class HandTracker:
    def __init__(self, w_env=W_ENV, h_env=H_ENV, vel_alpha=0.6):
        self.pos = np.array([w_env * 3 / 4, h_env / 3], dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.last_vision_time = None
        self.w_env = float(w_env)
        self.h_env = float(h_env)
        self.vel_alpha = float(vel_alpha)

    def update_vision(self, new_pos, t_now):
        new_pos = np.array(new_pos, dtype=np.float32)
        if self.last_vision_time is not None:
            dt = t_now - self.last_vision_time
            if dt > 1e-3:
                raw_vel = (new_pos - self.pos) / dt
                self.vel = self.vel_alpha * self.vel + (1 - self.vel_alpha) * raw_vel
        self.pos = new_pos.copy()
        self.last_vision_time = t_now

    def predict(self, t_now):
        if self.last_vision_time is None:
            return self.pos.copy()
        dt = t_now - self.last_vision_time
        predicted = self.pos + self.vel * dt
        predicted[0] = np.clip(predicted[0], 0.3, self.w_env - 0.3)
        predicted[1] = np.clip(predicted[1], 0.3, self.h_env - 0.3)
        return predicted.astype(np.float32)


class VisionThread(threading.Thread):
    def __init__(
        self,
        cap,
        cali,
        cv_model,
        w_px,
        h_px,
        w_env,
        h_env,
        result_queue,
        detect_hands=True,
        detect_microrobot=True,
    ):
        super().__init__(daemon=True)
        self.cap = cap
        self.cali = cali
        self.cv_model = cv_model
        self.w_px = int(w_px)
        self.h_px = int(h_px)
        self.w_env = float(w_env)
        self.h_env = float(h_env)
        self.result_queue = result_queue
        self.detect_microrobot = bool(detect_microrobot)
        if self.detect_microrobot and self.cv_model is None:
            raise ValueError(
                "cv_model is required when microrobot detection is enabled"
            )
        self.hand_detector = HandDetection() if detect_hands else None
        self.running = True
        self.frame_id = 0
        self.last_capture_t_perf = None
        self.frame_count = 0
        self.last_fps_time = time.perf_counter()

    def run(self):
        print("[vision] started")
        while self.running:
            capture_t_perf = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                continue

            camera_dt_s = None
            camera_hz_inst = None
            if self.last_capture_t_perf is not None:
                camera_dt_s = capture_t_perf - self.last_capture_t_perf
                if camera_dt_s > 1e-6:
                    camera_hz_inst = 1.0 / camera_dt_s
            self.last_capture_t_perf = capture_t_perf

            undistorted_frame = cv2.rotate(
                get_workspace(self.cali.undistort_frame(frame)),
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

            robot_trajectory = []
            microrobot_env = np.full(2, np.nan, dtype=np.float32)
            pixel_per_cm = 10.0
            if self.detect_microrobot:
                results = self.cv_model.predict(
                    undistorted_frame,
                    conf=0.7,
                    save=False,
                    imgsz=640,
                    verbose=False,
                )
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if x2 - x1 > 100:
                            continue
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        robot_trajectory.append((int(cx), int(cy)))
                        microrobot_env = np.array([
                            cx * self.w_env / self.w_px,
                            cy * self.h_env / self.h_px,
                        ], dtype=np.float32)
                        pixel_per_cm = ((x2 - x1) + (y2 - y1)) / 4
                        cv2.rectangle(
                            undistorted_frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 120, 20),
                            2,
                        )

            hand_positions = []
            hand_env = np.zeros(2, dtype=np.float32)
            if self.hand_detector is not None:
                undistorted_frame, hand_positions = self.hand_detector.process_frame(undistorted_frame)
                if hand_positions:
                    hand_env = np.array(hand_positions[0], dtype=np.float32) / np.array(
                        [self.w_px / self.w_env, self.h_px / self.h_env], dtype=np.float32
                    )

            if len(robot_trajectory) >= 2:
                for j in range(1, len(robot_trajectory)):
                    thickness = int((j / len(robot_trajectory)) * 4) + 1
                    cv2.line(undistorted_frame, robot_trajectory[j - 1], robot_trajectory[j], (0, 255, 255), thickness)

            processed_t_perf = time.perf_counter()
            self.frame_id += 1
            vision_result = VisionResult(
                hand_positions=hand_positions,
                hand_detected=len(hand_positions) > 0,
                microrobot_detected=len(robot_trajectory) > 0,
                robot_trajectory=robot_trajectory,
                hand_env=hand_env,
                microrobot_env=microrobot_env,
                pixel_per_cm=pixel_per_cm,
                undistorted_frame=undistorted_frame,
                frame_id=self.frame_id,
                capture_t_perf=capture_t_perf,
                processed_t_perf=processed_t_perf,
                camera_dt_s=camera_dt_s,
                camera_hz_inst=camera_hz_inst,
            )

            try:
                self.result_queue.put_nowait(vision_result)
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                self.result_queue.put_nowait(vision_result)

            self.frame_count += 1
            now = time.perf_counter()
            if now - self.last_fps_time >= 5.0:
                fps = self.frame_count / (now - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = now
                print(f"[vision] {fps:.1f} fps")

        if self.hand_detector is not None:
            self.hand_detector.release()
        print("[vision] stopped")

    def stop(self):
        self.running = False


def normalize_servo_timing(
    policy_hz,
    servo_mode="interpolated",
    servo_hz=DEFAULT_SERVO_FREQ,
    target_timeout_s=None,
    max_speed_cm_s=None,
    max_step_cm=DEFAULT_MAX_SAFE_STRIDE,
    lookahead_time_s=DEFAULT_SERVO_LOOKAHEAD,
    gain=DEFAULT_SERVO_GAIN,
):
    policy_hz = float(policy_hz)
    servo_hz = float(servo_hz)
    max_step_cm = float(max_step_cm)
    lookahead_time_s = float(lookahead_time_s)
    gain = int(gain)
    if servo_mode not in {"interpolated", "legacy"}:
        raise ValueError("servo mode must be 'interpolated' or 'legacy'")
    if policy_hz <= 0.0:
        raise ValueError("policy frequency must be positive")
    if servo_hz <= 0.0:
        raise ValueError("servo frequency must be positive")
    if servo_mode == "interpolated" and servo_hz < policy_hz:
        raise ValueError("servo frequency must be at least the policy frequency")
    if max_step_cm <= 0.0:
        raise ValueError("max step must be positive")
    if not 0.03 <= lookahead_time_s <= 0.2:
        raise ValueError("servo lookahead must be in [0.03, 0.2] seconds")
    if not 100 <= gain <= 2000:
        raise ValueError("servo gain must be in [100, 2000]")

    timeout = (
        max(3.0 / policy_hz, 0.15)
        if target_timeout_s is None
        else float(target_timeout_s)
    )
    if timeout <= 0.0:
        raise ValueError("servo target timeout must be positive")
    speed_cm_s = (
        max_step_cm * policy_hz
        if max_speed_cm_s is None
        else float(max_speed_cm_s)
    )
    if speed_cm_s <= 0.0:
        raise ValueError("servo maximum speed must be positive")

    effective_servo_hz = policy_hz if servo_mode == "legacy" else servo_hz
    servo_period_s = 1.0 / effective_servo_hz
    return ServoTimingConfig(
        mode=servo_mode,
        policy_hz=policy_hz,
        servo_hz=effective_servo_hz,
        policy_period_s=1.0 / policy_hz,
        servo_period_s=servo_period_s,
        target_timeout_s=timeout,
        max_speed_cm_s=speed_cm_s,
        max_translation_per_tick_m=(speed_cm_s / 100.0) * servo_period_s,
        lookahead_time_s=lookahead_time_s,
        gain=gain,
    )


def _as_finite_pose(pose, label="pose"):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{label} must be a finite 6D pose")
    return pose


def linear_interpolate_pose(start_pose, target_pose, phase):
    start = _as_finite_pose(start_pose, "start pose")
    target = _as_finite_pose(target_pose, "target pose")
    phase = float(np.clip(phase, 0.0, 1.0))
    return (start + phase * (target - start)).astype(np.float64)


def limit_pose_translation_step(previous_pose, requested_pose, max_step_m):
    previous = _as_finite_pose(previous_pose, "previous pose")
    requested = _as_finite_pose(requested_pose, "requested pose")
    max_step_m = float(max_step_m)
    if max_step_m <= 0.0:
        raise ValueError("maximum translation step must be positive")
    delta = requested[:3] - previous[:3]
    distance = float(np.linalg.norm(delta))
    limited = distance > max_step_m and distance > 1e-12
    command = requested.copy()
    if limited:
        command[:3] = previous[:3] + delta / distance * max_step_m
    return command, limited


def wait_until_high_resolution(
    deadline,
    clock=time.perf_counter,
    sleeper=time.sleep,
    spin_threshold_s=0.01,
):
    remaining = float(deadline) - clock()
    if remaining > spin_threshold_s:
        sleeper(remaining - spin_threshold_s)
    while clock() < deadline:
        sleeper(0)


class InterpolatedServoThread(threading.Thread):
    def __init__(
        self,
        robot_control,
        initial_pose,
        timing,
        clock=time.perf_counter,
        sleeper=time.sleep,
    ):
        super().__init__(daemon=True, name="interpolated-servo")
        if timing.mode != "interpolated":
            raise ValueError("InterpolatedServoThread requires interpolated mode")
        initial_pose = _as_finite_pose(initial_pose, "initial servo pose")
        now = float(clock())
        self.robot_control = robot_control
        self.timing = timing
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._target = ServoTarget(
            pose=initial_pose.copy(),
            policy_step=-1,
            sequence=0,
            published_at_perf=now,
        )
        self._snapshot = ServoSnapshot(
            command_sequence=0,
            policy_step=-1,
            commanded_pose=initial_pose.copy(),
            target_pose=initial_pose.copy(),
            period_api_available=bool(
                getattr(robot_control, "control_frequency_configured", False)
                and hasattr(robot_control.rtde_c, "initPeriod")
                and hasattr(robot_control.rtde_c, "waitPeriod")
            ),
        )
        self._failure = None
        self._requested_stop_reason = ""

    @property
    def period_api_available(self):
        return self._snapshot.period_api_available

    def publish_target(self, target_pose, policy_step, timestamp_perf=None):
        self.raise_if_failed()
        pose = _as_finite_pose(target_pose, "servo target")
        timestamp = self._clock() if timestamp_perf is None else float(timestamp_perf)
        with self._lock:
            sequence = self._target.sequence + 1
            self._target = ServoTarget(
                pose=pose.copy(),
                policy_step=int(policy_step),
                sequence=sequence,
                published_at_perf=timestamp,
            )
        return sequence

    def snapshot(self):
        with self._lock:
            snapshot = self._snapshot
            return ServoSnapshot(
                servo_dt_s=snapshot.servo_dt_s,
                servo_loop_hz_inst=snapshot.servo_loop_hz_inst,
                target_age_s=snapshot.target_age_s,
                interpolation_phase=snapshot.interpolation_phase,
                command_sequence=snapshot.command_sequence,
                policy_step=snapshot.policy_step,
                commanded_pose=(
                    None
                    if snapshot.commanded_pose is None
                    else snapshot.commanded_pose.copy()
                ),
                target_pose=(
                    None
                    if snapshot.target_pose is None
                    else snapshot.target_pose.copy()
                ),
                deadline_overrun_s=snapshot.deadline_overrun_s,
                deadline_overrun_count=snapshot.deadline_overrun_count,
                watchdog_stopped=snapshot.watchdog_stopped,
                stop_reason=snapshot.stop_reason,
                period_api_available=snapshot.period_api_available,
            )

    def request_stop(self, reason="requested_stop"):
        with self._lock:
            if not self._requested_stop_reason:
                self._requested_stop_reason = str(reason)
        self._stop_event.set()

    def stop_and_join(self, reason="requested_stop", timeout_s=2.0):
        self.request_stop(reason)
        if self.is_alive():
            self.join(timeout=float(timeout_s))
        if self.is_alive():
            raise RuntimeError("Servo thread did not stop within the timeout")
        self.raise_if_failed()

    def raise_if_failed(self):
        with self._lock:
            failure = self._failure
        if failure is not None:
            if isinstance(failure, ServoLoopFailure):
                raise failure
            raise ServoLoopFailure(f"Servo loop failed: {failure}") from failure

    def _set_failure(self, failure, stop_reason, watchdog=False):
        with self._lock:
            if self._failure is None:
                self._failure = failure
            self._snapshot.watchdog_stopped = bool(watchdog)
            self._snapshot.stop_reason = str(stop_reason)
        self._stop_event.set()

    def _safe_stop(self):
        try:
            self.robot_control.servo_stop()
        except Exception as exc:
            with self._lock:
                if self._failure is None:
                    self._failure = exc
                if not self._snapshot.stop_reason:
                    self._snapshot.stop_reason = "servo_stop_failed"

    def run(self):
        command_pose = self._target.pose.copy()
        segment_start_pose = command_pose.copy()
        segment_start_perf = self._target.published_at_perf
        active_sequence = self._target.sequence
        last_cycle_start = None
        next_deadline = self._clock()
        deadline_overrun_count = 0
        period_api = self.period_api_available

        try:
            while not self._stop_event.is_set():
                cycle_start = self._clock()
                servo_dt_s = (
                    None
                    if last_cycle_start is None
                    else cycle_start - last_cycle_start
                )
                last_cycle_start = cycle_start
                deadline_overrun_s = max(0.0, cycle_start - next_deadline)
                if deadline_overrun_s > 1e-3:
                    deadline_overrun_count += 1

                with self._lock:
                    target = ServoTarget(
                        pose=self._target.pose.copy(),
                        policy_step=self._target.policy_step,
                        sequence=self._target.sequence,
                        published_at_perf=self._target.published_at_perf,
                    )
                target_age_s = max(0.0, cycle_start - target.published_at_perf)
                if target_age_s > self.timing.target_timeout_s:
                    reason = (
                        f"stale_target_{target_age_s:.3f}s_"
                        f"over_{self.timing.target_timeout_s:.3f}s"
                    )
                    self._set_failure(
                        ServoLoopSafetyStop(reason),
                        reason,
                        watchdog=True,
                    )
                    break

                if target.sequence != active_sequence:
                    segment_start_pose = command_pose.copy()
                    segment_start_perf = target.published_at_perf
                    active_sequence = target.sequence
                phase = np.clip(
                    (cycle_start - segment_start_perf)
                    / self.timing.policy_period_s,
                    0.0,
                    1.0,
                )
                requested_pose = linear_interpolate_pose(
                    segment_start_pose,
                    target.pose,
                    phase,
                )
                command_pose, _ = limit_pose_translation_step(
                    command_pose,
                    requested_pose,
                    self.timing.max_translation_per_tick_m,
                )

                period_token = (
                    self.robot_control.rtde_c.initPeriod()
                    if period_api
                    else None
                )
                servo_ok = self.robot_control.servo_robot(
                    command_pose.tolist(),
                    dt=self.timing.servo_period_s,
                    lookahead_time=self.timing.lookahead_time_s,
                    gain=self.timing.gain,
                )
                if servo_ok is False:
                    raise RuntimeError("servoL returned failure")
                if period_api:
                    self.robot_control.rtde_c.waitPeriod(period_token)

                with self._lock:
                    self._snapshot = ServoSnapshot(
                        servo_dt_s=servo_dt_s,
                        servo_loop_hz_inst=(
                            None
                            if not servo_dt_s or servo_dt_s <= 1e-9
                            else 1.0 / servo_dt_s
                        ),
                        target_age_s=target_age_s,
                        interpolation_phase=float(phase),
                        command_sequence=target.sequence,
                        policy_step=target.policy_step,
                        commanded_pose=command_pose.copy(),
                        target_pose=target.pose.copy(),
                        deadline_overrun_s=deadline_overrun_s,
                        deadline_overrun_count=deadline_overrun_count,
                        watchdog_stopped=False,
                        stop_reason="",
                        period_api_available=period_api,
                    )

                next_deadline += self.timing.servo_period_s
                if not period_api:
                    if self._clock() > next_deadline:
                        next_deadline = self._clock() + self.timing.servo_period_s
                    wait_until_high_resolution(
                        next_deadline,
                        clock=self._clock,
                        sleeper=self._sleeper,
                    )
        except Exception as exc:
            self._set_failure(exc, "servo_exception")
        finally:
            with self._lock:
                if not self._snapshot.stop_reason:
                    self._snapshot.stop_reason = (
                        self._requested_stop_reason or "servo_loop_stopped"
                    )
            self._safe_stop()


def build_servo_log_fields(snapshot, actual_robot_pose=None):
    if snapshot is None:
        return {}
    actual_pose = (
        None
        if actual_robot_pose is None
        else _as_finite_pose(actual_robot_pose, "actual Robot pose")
    )
    commanded = snapshot.commanded_pose
    target = snapshot.target_pose
    tracking_error_cm = (
        None
        if commanded is None or actual_pose is None
        else float(np.linalg.norm(commanded[:3] - actual_pose[:3]) * 100.0)
    )
    target_error_cm = (
        None
        if target is None or actual_pose is None
        else float(np.linalg.norm(target[:3] - actual_pose[:3]) * 100.0)
    )
    return {
        "servo_dt_s": snapshot.servo_dt_s,
        "servo_loop_hz_inst": snapshot.servo_loop_hz_inst,
        "servo_target_age_s": snapshot.target_age_s,
        "servo_interpolation_phase": snapshot.interpolation_phase,
        "servo_command_sequence": snapshot.command_sequence,
        "servo_policy_step": snapshot.policy_step,
        "servo_commanded_world_x": None if commanded is None else float(commanded[0]),
        "servo_commanded_world_y": None if commanded is None else float(commanded[1]),
        "servo_commanded_world_z": None if commanded is None else float(commanded[2]),
        "servo_target_world_x": None if target is None else float(target[0]),
        "servo_target_world_y": None if target is None else float(target[1]),
        "servo_target_world_z": None if target is None else float(target[2]),
        "servo_tracking_error_cm": tracking_error_cm,
        "servo_target_error_cm": target_error_cm,
        "servo_deadline_overrun_s": snapshot.deadline_overrun_s,
        "servo_deadline_overrun_count": snapshot.deadline_overrun_count,
        "servo_watchdog_stopped": snapshot.watchdog_stopped,
        "servo_stop_reason": snapshot.stop_reason,
    }


def infer_observation_layout(model):
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if not shape:
        return 44, 16, MOTION_HISTORY_CHANNELS, "motion"
    obs_dim = int(shape[0])
    history_dim = obs_dim - OBS_SCALAR_DIM
    if history_dim > 0 and history_dim % INTERACTION_HISTORY_CHANNELS == 0 and obs_dim != 44:
        return obs_dim, history_dim // INTERACTION_HISTORY_CHANNELS, INTERACTION_HISTORY_CHANNELS, "interaction"
    if history_dim > 0 and history_dim % MOTION_HISTORY_CHANNELS == 0:
        return obs_dim, history_dim // MOTION_HISTORY_CHANNELS, MOTION_HISTORY_CHANNELS, "motion"
    raise ValueError(f"Unsupported policy observation dimension: {obs_dim}")


def fit_history(history_values, expected_size):
    flat = np.asarray(history_values, dtype=np.float32).flatten()
    if flat.size > expected_size:
        flat = flat[-expected_size:]
    elif flat.size < expected_size:
        flat = np.concatenate([np.zeros(expected_size - flat.size, dtype=np.float32), flat])
    return flat.astype(np.float32)


def find_default_policy():
    for path in DEFAULT_POLICY_CANDIDATES:
        if path.exists():
            return path
    return None


def find_default_hand_model():
    for path in DEFAULT_HAND_MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def validate_hand_model(model):
    obs_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if obs_shape != (VIRTUAL_HAND_OBS_DIM,):
        raise ValueError(
            f"Virtual Hand model observation shape must be ({VIRTUAL_HAND_OBS_DIM},), got {obs_shape}"
        )

    action_space = getattr(model, "action_space", None)
    action_shape = getattr(action_space, "shape", None)
    if action_shape != (2,):
        raise ValueError(f"Virtual Hand model action shape must be (2,), got {action_shape}")

    low = np.asarray(getattr(action_space, "low", []), dtype=np.float32)
    high = np.asarray(getattr(action_space, "high", []), dtype=np.float32)
    if low.shape != (2,) or high.shape != (2,) or np.any(low > -1.0) or np.any(high < 1.0):
        raise ValueError("Virtual Hand model action space must cover [-1, 1] in both dimensions")


def build_virtual_hand_observation(
    hand_position,
    robot_position,
    stride_hand,
    last_hand_actual_move,
    robot_history_buffer,
):
    hand_position = np.asarray(hand_position, dtype=np.float32)
    robot_position = np.asarray(robot_position, dtype=np.float32)
    boundary_distances = np.array([
        hand_position[0],
        W_ENV - hand_position[0],
        hand_position[1],
        H_ENV - hand_position[1],
    ], dtype=np.float32)
    flat_history = fit_history(
        robot_history_buffer,
        VIRTUAL_HAND_HISTORY_LENGTH * MOTION_HISTORY_CHANNELS,
    )
    obs = np.concatenate((
        hand_position,
        robot_position,
        np.array([np.linalg.norm(robot_position - hand_position)], dtype=np.float32),
        boundary_distances,
        np.array([stride_hand], dtype=np.float32),
        np.asarray(last_hand_actual_move, dtype=np.float32),
        flat_history,
    )).astype(np.float32)
    if obs.shape != (VIRTUAL_HAND_OBS_DIM,):
        raise RuntimeError(
            f"Built virtual Hand observation has {obs.shape[0]} dims, expected {VIRTUAL_HAND_OBS_DIM}"
        )
    return obs


def make_virtual_hand_delay_buffer(delay_frames):
    delay_frames = int(delay_frames)
    if delay_frames < 0:
        raise ValueError("Virtual Hand delay frames cannot be negative")
    if delay_frames == 0:
        return None
    return deque(
        [np.zeros(2, dtype=np.float32) for _ in range(delay_frames)],
        maxlen=delay_frames,
    )


def apply_virtual_hand_delay(hand_intent, delay_buffer):
    hand_intent = np.asarray(hand_intent, dtype=np.float32)
    if delay_buffer is None:
        return hand_intent.copy()
    delayed_move = np.asarray(delay_buffer.popleft(), dtype=np.float32)
    delay_buffer.append(hand_intent.copy())
    return delayed_move


def apply_virtual_hand_execution(
    hand_action,
    stride_hand,
    last_hand_actual_move,
    smoothing_alpha=1.0,
    delay_buffer=None,
):
    smoothing_alpha = float(smoothing_alpha)
    if not 0.0 <= smoothing_alpha <= 1.0:
        raise ValueError("Virtual Hand smoothing alpha must be in [0, 1]")

    clipped_action = np.clip(
        np.asarray(hand_action, dtype=np.float32),
        -1.0,
        1.0,
    )
    hand_intent = clipped_action * float(stride_hand)
    delayed_move = apply_virtual_hand_delay(hand_intent, delay_buffer)
    last_move = np.asarray(last_hand_actual_move, dtype=np.float32)
    smoothed_move = (
        smoothing_alpha * delayed_move
        + (1.0 - smoothing_alpha) * last_move
    ).astype(np.float32)
    delta_v = smoothed_move - last_move
    accel_magnitude = float(np.linalg.norm(delta_v))
    max_accel = 1.5 * float(stride_hand)
    accel_clipped = bool(accel_magnitude > max_accel and accel_magnitude > 1e-8)
    if accel_clipped:
        delta_v = (delta_v / accel_magnitude) * max_accel
    executed_move = (last_move + delta_v).astype(np.float32)
    diagnostics = {
        "smoothing_alpha": smoothing_alpha,
        "delay_frames": 0 if delay_buffer is None else int(delay_buffer.maxlen),
        "action": clipped_action,
        "action_norm": float(np.linalg.norm(clipped_action)),
        "command_move": hand_intent.astype(np.float32),
        "command_norm_cm": float(np.linalg.norm(hand_intent)),
        "delayed_move": delayed_move,
        "delayed_norm_cm": float(np.linalg.norm(delayed_move)),
        "smoothed_move": smoothed_move,
        "smoothed_norm_cm": float(np.linalg.norm(smoothed_move)),
        "executed_move": executed_move,
        "executed_norm_cm": float(np.linalg.norm(executed_move)),
        "accel_clipped": accel_clipped,
    }
    return executed_move, diagnostics


def build_virtual_hand_log_fields(diagnostics, stride_hand, inference_ms):
    if diagnostics is None:
        return {}
    action = diagnostics["action"]
    command_move = diagnostics["command_move"]
    delayed_move = diagnostics["delayed_move"]
    smoothed_move = diagnostics["smoothed_move"]
    executed_move = diagnostics["executed_move"]
    actual_move = diagnostics["actual_move"]
    return {
        "virtual_hand_stride_cm": float(stride_hand),
        "virtual_hand_smoothing_alpha": float(diagnostics["smoothing_alpha"]),
        "virtual_hand_delay_frames": int(diagnostics["delay_frames"]),
        "virtual_hand_action_x": float(action[0]),
        "virtual_hand_action_y": float(action[1]),
        "virtual_hand_action_norm": float(diagnostics["action_norm"]),
        "virtual_hand_command_dx_cm": float(command_move[0]),
        "virtual_hand_command_dy_cm": float(command_move[1]),
        "virtual_hand_command_norm_cm": float(diagnostics["command_norm_cm"]),
        "virtual_hand_delayed_dx_cm": float(delayed_move[0]),
        "virtual_hand_delayed_dy_cm": float(delayed_move[1]),
        "virtual_hand_delayed_norm_cm": float(diagnostics["delayed_norm_cm"]),
        "virtual_hand_smoothed_dx_cm": float(smoothed_move[0]),
        "virtual_hand_smoothed_dy_cm": float(smoothed_move[1]),
        "virtual_hand_smoothed_norm_cm": float(diagnostics["smoothed_norm_cm"]),
        "virtual_hand_exec_dx_cm": float(executed_move[0]),
        "virtual_hand_exec_dy_cm": float(executed_move[1]),
        "virtual_hand_exec_norm_cm": float(diagnostics["executed_norm_cm"]),
        "virtual_hand_actual_dx_cm": float(actual_move[0]),
        "virtual_hand_actual_dy_cm": float(actual_move[1]),
        "virtual_hand_actual_norm_cm": float(diagnostics["actual_norm_cm"]),
        "virtual_hand_accel_clipped": bool(diagnostics["accel_clipped"]),
        "virtual_hand_workspace_clipped": bool(diagnostics["workspace_clipped"]),
        "virtual_hand_policy_inference_ms": float(inference_ms),
    }


def sample_virtual_hand_start(robot_position, zpd_low, zpd_high, rng, margin=WORKSPACE_MARGIN):
    robot_position = np.asarray(robot_position, dtype=np.float32)
    initial_distance = 0.5 * (float(zpd_low) + float(zpd_high))
    if initial_distance <= 0:
        raise ValueError("Virtual Hand initial distance must be positive")

    initial_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    for offset in np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False):
        angle = initial_angle + float(offset)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        candidate = robot_position + direction * initial_distance
        if (
            margin <= candidate[0] <= W_ENV - margin
            and margin <= candidate[1] <= H_ENV - margin
        ):
            return candidate.astype(np.float32), float(angle % (2.0 * np.pi))

    raise RuntimeError("Could not place virtual Hand at the requested ZPD midpoint distance")


def parse_args():
    default_policy = find_default_policy()
    default_hand_model = find_default_hand_model()
    parser = argparse.ArgumentParser(description="Record one physical rollout with the League policy or tuned CV-MPC baseline.")
    parser.add_argument("--controller", choices=["league", "cv_mpc"], default="league", help="Robot controller used for this rollout.")
    parser.add_argument("--model", "--policy", dest="model", default=str(default_policy) if default_policy else None, help="PPO robot policy .zip used when --controller league.")
    parser.add_argument("--hand-source", choices=["camera", "virtual"], default="camera", help="Use the camera-tracked hand or an online virtual Hand policy.")
    parser.add_argument("--hand-model", default=str(default_hand_model) if default_hand_model else None, help="PPO Hand policy .zip used when --hand-source virtual.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for virtual Hand initialization and policy sampling.")
    parser.add_argument("--hand-stride", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hand-alpha", type=float, default=1.0, help="Virtual Hand motion smoothing coefficient in [0, 1].")
    parser.add_argument("--hand-delay-frames", type=int, default=0, help="Virtual Hand neural delay in control frames.")
    parser.add_argument("--vision-model", default=str(DEFAULT_VISION_MODEL), help=argparse.SUPPRESS)
    parser.add_argument(
        "--microrobot-vision",
        choices=["auto", "yolo", "none"],
        default="auto",
        help=(
            "Microrobot YOLO mode. auto disables YOLO for every virtual-Hand "
            "rollout because control and distance use UR RTDE TCP."
        ),
    )
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "deployment_rollouts"), help=argparse.SUPPRESS)
    parser.add_argument("--subject", default="pilot", help="Short subject/session label.")
    parser.add_argument("--condition", default=None, help="Condition label; defaults to the selected controller.")
    parser.add_argument("--seconds", "--duration", dest="duration", type=float, default=30.0, help="Rollout duration in seconds.")
    parser.add_argument("--zpd-low", type=float, default=3.5, help=argparse.SUPPRESS)
    parser.add_argument("--zpd-high", type=float, default=5.5, help=argparse.SUPPRESS)
    parser.add_argument(
        "--control-hz",
        "--policy-hz",
        dest="control_hz",
        type=float,
        default=DEFAULT_CONTROL_FREQ,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--servo-mode",
        choices=["interpolated", "legacy"],
        default="interpolated",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--servo-hz",
        type=float,
        default=DEFAULT_SERVO_FREQ,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--servo-target-timeout-s",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--servo-max-speed-cm-s",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--servo-lookahead",
        type=float,
        default=DEFAULT_SERVO_LOOKAHEAD,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--servo-gain",
        type=int,
        default=DEFAULT_SERVO_GAIN,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--stride", type=float, default=DEFAULT_STRIDE, help=argparse.SUPPRESS)
    parser.add_argument("--max-step", type=float, default=DEFAULT_MAX_SAFE_STRIDE, help=argparse.SUPPRESS)
    parser.add_argument("--robot-ip", default="192.168.1.2", help=argparse.SUPPRESS)
    parser.add_argument("--camera", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--camera-width", type=int, default=2592, help=argparse.SUPPRESS)
    parser.add_argument("--camera-height", type=int, default=1944, help=argparse.SUPPRESS)
    parser.add_argument("--save-video", action="store_true", help="Save annotated rollout video.")
    parser.add_argument("--snapshot-step", type=int, default=None, help=argparse.SUPPRESS)
    parser.set_defaults(stop_on_catch=True)
    parser.add_argument("--no-stop-on-catch", dest="stop_on_catch", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--catch-distance", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--no-display", action="store_true", help="Do not open the OpenCV preview window.")
    parser.add_argument("--countdown", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def require_file(path, label):
    if path is None:
        raise FileNotFoundError(f"No default {label} found; pass it explicitly.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_microrobot_vision_mode(args):
    requested_mode = getattr(args, "microrobot_vision", "auto")
    if requested_mode not in {"auto", "yolo", "none"}:
        raise ValueError(
            "microrobot vision mode must be one of: auto, yolo, none"
        )
    if requested_mode != "auto":
        return requested_mode, None
    if args.hand_source == "virtual":
        return "none", "virtual_hand_uses_ur_rtde_tcp"
    return "yolo", None


def resolve_vision_model_path(path, yolo_enabled):
    if not yolo_enabled:
        return None
    return require_file(path, "vision model")


def load_microrobot_model(path, yolo_enabled, yolo_class=None):
    if not yolo_enabled:
        return None
    model_class = yolo_class or YOLO
    if model_class is None:
        raise RuntimeError("YOLO dependency was not loaded")
    return model_class(str(path))


def load_runtime_dependencies(load_hand_detection=True, load_yolo=True):
    global cv2, PPO, YOLO, CameraCalibration, DeploymentRolloutLogger, HandDetection, URControl, get_workspace

    import cv2 as cv2_module
    from stable_baselines3 import PPO as ppo_class

    yolo_class = None
    if load_yolo:
        from ultralytics import YOLO as yolo_class

    from camera_calibration.camera_calibration import CameraCalibration as calibration_class
    logger_path = SCRIPT_DIR / "callbacks" / "deployment_rollout_logger.py"
    spec = importlib.util.spec_from_file_location("deployment_rollout_logger", logger_path)
    logger_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logger_module)
    logger_class = logger_module.DeploymentRolloutLogger
    hand_detection_class = None
    if load_hand_detection:
        from cv.hand_detect import HandDetection as hand_detection_class
    from cv.get_workspace import get_workspace as workspace_func
    from robot_control.ur_control import URControl as ur_control_class

    cv2 = cv2_module
    PPO = ppo_class
    YOLO = yolo_class
    CameraCalibration = calibration_class
    DeploymentRolloutLogger = logger_class
    HandDetection = hand_detection_class
    URControl = ur_control_class
    get_workspace = workspace_func


def safe_transition_to_center(robot_control, cali, w_px, h_px, countdown):
    print("[robot] moving to workspace center")
    robot_control.servo_stop()
    center_pixel = np.array([w_px / 2, h_px / 2], dtype=np.float64)
    center_world = cali.pixel_to_world(center_pixel.astype(int))
    target_pose = [center_world[0], center_world[1], Z, RX_C, RY_C, RZ_C]
    robot_control.move_robot(target_pose, speed=0.2, acceleration=0.2)
    for remaining in range(int(countdown), 0, -1):
        print(f"[start] {remaining}")
        time.sleep(1)
    return center_pixel


def get_robot_env_position(robot_control, cali, w_px, h_px):
    pose = robot_control.get_robot_pose()
    position_robot_world = pose[:2]
    real_robot_pixel = cali.world_to_pixel(position_robot_world)
    position_robot_env = np.array([
        real_robot_pixel[0] * W_ENV / w_px,
        real_robot_pixel[1] * H_ENV / h_px,
    ], dtype=np.float32)
    return pose, real_robot_pixel, position_robot_env


def limit_vector_norm(vector, max_norm):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > float(max_norm) and norm > 1e-8:
        return (vector / norm * float(max_norm)).astype(np.float32), True
    return vector.astype(np.float32), False


def workspace_clearance_env(position, margin=WORKSPACE_MARGIN):
    position = np.asarray(position, dtype=np.float32)
    return float(min(
        position[0] - margin,
        W_ENV - margin - position[0],
        position[1] - margin,
        H_ENV - margin - position[1],
    ))


def controller_label(controller):
    return CONTROLLER_LABELS.get(controller, str(controller))


def format_rollout_configuration(args, virtual_hand_stride, servo_timing=None):
    if servo_timing is None:
        servo_timing = normalize_servo_timing(
            policy_hz=args.control_hz,
            servo_mode=getattr(args, "servo_mode", "legacy"),
            servo_hz=getattr(args, "servo_hz", args.control_hz),
            target_timeout_s=getattr(args, "servo_target_timeout_s", None),
            max_speed_cm_s=getattr(args, "servo_max_speed_cm_s", None),
            max_step_cm=args.max_step,
            lookahead_time_s=getattr(
                args,
                "servo_lookahead",
                DEFAULT_SERVO_LOOKAHEAD,
            ),
            gain=getattr(args, "servo_gain", DEFAULT_SERVO_GAIN),
        )
    lines = [
        "-" * 78,
        "ROLLOUT RUNTIME CONFIGURATION",
        "-" * 78,
        f"Controller   : {controller_label(args.controller)}",
        (
            f"Robot stride: {float(args.stride):.3f} cm/action  |  "
            f"max step: {float(args.max_step):.3f} cm/control step"
        ),
        f"Hand source  : {args.hand_source}",
    ]
    if args.hand_source == "virtual":
        lines.extend([
            f"Hand stride : {float(virtual_hand_stride):.3f} cm/action",
            (
                f"Hand DR     : alpha={float(args.hand_alpha):.3f}  |  "
                f"delay={int(args.hand_delay_frames)} frame(s) "
                f"({1000.0 * args.hand_delay_frames / args.control_hz:.0f} ms nominal)"
            ),
        ])
    lines.extend([
        (
            f"Task         : duration={float(args.duration):.1f} s  |  "
            f"policy={servo_timing.policy_hz:.1f} Hz  |  "
            f"catch={float(args.catch_distance):.2f} cm"
        ),
        (
            f"Servo        : mode={servo_timing.mode}  |  "
            f"rate={servo_timing.servo_hz:.1f} Hz  |  "
            f"watchdog={servo_timing.target_timeout_s:.3f} s"
        ),
        f"Seed         : {int(args.seed)}",
        "-" * 78,
    ])
    return "\n".join(lines)


def resolve_common_target_env(
    action,
    measured_robot_env,
    stride,
    max_step,
    margin=WORKSPACE_MARGIN,
    safety_guard=COMMON_SAFETY_GUARD,
):
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    raw_delta_env = action * float(stride)
    limited_delta_env, target_step_limited = limit_vector_norm(raw_delta_env, max_step)
    desired_target_env = np.asarray(measured_robot_env, dtype=np.float32) + limited_delta_env
    low = np.array([margin + safety_guard, margin + safety_guard], dtype=np.float32)
    high = np.array([
        W_ENV - margin - safety_guard,
        H_ENV - margin - safety_guard,
    ], dtype=np.float32)
    target_env = np.clip(desired_target_env, low, high).astype(np.float32)
    target_clipped = not np.allclose(desired_target_env, target_env)
    return target_env, desired_target_env, target_clipped, target_step_limited


def main():
    args = parse_args()
    if args.zpd_low >= args.zpd_high:
        raise ValueError("--zpd-low must be smaller than --zpd-high")
    if args.hand_stride is not None and args.hand_stride <= 0:
        raise ValueError("--hand-stride must be positive")
    if not 0.0 <= args.hand_alpha <= 1.0:
        raise ValueError("--hand-alpha must be in [0, 1]")
    if args.hand_delay_frames < 0:
        raise ValueError("--hand-delay-frames cannot be negative")
    if args.duration <= 0:
        raise ValueError("--seconds must be positive")
    servo_timing = normalize_servo_timing(
        policy_hz=args.control_hz,
        servo_mode=args.servo_mode,
        servo_hz=args.servo_hz,
        target_timeout_s=args.servo_target_timeout_s,
        max_speed_cm_s=args.servo_max_speed_cm_s,
        max_step_cm=args.max_step,
        lookahead_time_s=args.servo_lookahead,
        gain=args.servo_gain,
    )

    virtual_hand_rng = np.random.default_rng(args.seed)
    virtual_hand_stride = None
    if args.hand_source == "virtual":
        virtual_hand_stride = (
            float(args.hand_stride)
            if args.hand_stride is not None
            else float(virtual_hand_rng.uniform(*VIRTUAL_HAND_STRIDE_RANGE))
        )

    microrobot_vision_mode, microrobot_vision_disabled_reason = (
        resolve_microrobot_vision_mode(args)
    )
    microrobot_yolo_enabled = microrobot_vision_mode == "yolo"

    model_path = require_file(args.model, "PPO policy") if args.controller == "league" else None
    hand_model_path = require_file(args.hand_model, "PPO Hand policy") if args.hand_source == "virtual" else None
    vision_model_path = resolve_vision_model_path(
        args.vision_model,
        microrobot_yolo_enabled,
    )
    load_runtime_dependencies(
        load_hand_detection=args.hand_source == "camera",
        load_yolo=microrobot_yolo_enabled,
    )

    print(
        format_rollout_configuration(
            args,
            virtual_hand_stride,
            servo_timing=servo_timing,
        ),
        flush=True,
    )
    print("MODEL FILES")
    if model_path is not None:
        print(f"Robot policy : {model_path}")
    else:
        print("Robot policy : Constant-Velocity MPC (no learned model)")
    if hand_model_path is not None:
        print(f"Hand policy  : {hand_model_path}")
    if microrobot_yolo_enabled:
        print("Microrobot YOLO: enabled")
        print(f"Vision model : {vision_model_path}")
    else:
        print(
            "Microrobot YOLO: disabled"
            + (
                f" ({microrobot_vision_disabled_reason})"
                if microrobot_vision_disabled_reason
                else ""
            )
        )
        print("Vision model : not loaded")
    print("-" * 78, flush=True)

    cv_model = load_microrobot_model(
        vision_model_path,
        microrobot_yolo_enabled,
    )
    cali = CameraCalibration()
    robot_control = None
    cap = None
    vision_thread = None
    servo_thread = None
    logger = None
    done_reason = "unknown"

    try:
        robot_control = URControl(
            args.robot_ip,
            control_frequency=(
                servo_timing.servo_hz
                if servo_timing.mode == "interpolated"
                else None
            ),
        )
        rl_model = None
        hand_model = None
        mpc_controller = None
        expected_obs_dim, history_length, history_channels, history_mode = 44, 16, MOTION_HISTORY_CHANNELS, "motion"
        if args.controller == "league":
            rl_model = PPO.load(str(model_path), custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0, "optimizer_class": None})
            expected_obs_dim, history_length, history_channels, history_mode = infer_observation_layout(rl_model)
            print(
                f"[model] Robot observation: {expected_obs_dim} dims "
                f"({history_mode}, length={history_length}, channels={history_channels})"
            )
        else:
            mpc_controller = ConstantVelocityMPCController(**CV_MPC_CONFIG)

        if args.hand_source == "virtual":
            hand_model = PPO.load(
                str(hand_model_path),
                custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0, "optimizer_class": None},
            )
            validate_hand_model(hand_model)
            if hasattr(hand_model, "set_random_seed"):
                hand_model.set_random_seed(args.seed)
            print(
                f"[model] Hand observation: {VIRTUAL_HAND_OBS_DIM} dims "
                f"(motion, length={VIRTUAL_HAND_HISTORY_LENGTH})"
            )

        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {args.camera} could not be opened")

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read initial camera frame")
        workspace_frame = cv2.rotate(get_workspace(cali.undistort_frame(frame)), cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_px, w_px = workspace_frame.shape[:2]

        if not args.no_display:
            cv2.namedWindow("Deployment Chase", cv2.WINDOW_NORMAL)

        safe_transition_to_center(robot_control, cali, w_px, h_px, args.countdown)
        initial_robot_pose, _, initial_robot_env = get_robot_env_position(
            robot_control,
            cali,
            w_px,
            h_px,
        )
        virtual_hand_env = None
        virtual_hand_initial_angle = None
        if args.hand_source == "virtual":
            virtual_hand_env, virtual_hand_initial_angle = sample_virtual_hand_start(
                initial_robot_env,
                args.zpd_low,
                args.zpd_high,
                virtual_hand_rng,
            )
            print(
                f"[hand] initial position: ({virtual_hand_env[0]:.2f}, {virtual_hand_env[1]:.2f}), "
                f"angle={virtual_hand_initial_angle:.3f} rad"
            )

        metadata = {
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "task": "DeploymentChase",
            "controller": args.controller,
            "policy_path": str(model_path) if model_path is not None else None,
            "cv_mpc_config": CV_MPC_CONFIG if args.controller == "cv_mpc" else None,
            "cv_mpc_observation_inputs": (
                "current absolute Robot/Hand positions, workspace boundaries, "
                "previous measured Robot move, and completed observed Hand moves "
                "from the shared 8-channel interaction history"
                if args.controller == "cv_mpc"
                else None
            ),
            "cv_mpc_hand_velocity_source": (
                "interaction_history channels 6-7 (hand_move_x, hand_move_y)"
                if args.controller == "cv_mpc"
                else None
            ),
            "cv_mpc_interaction_history_length": (
                int(history_length) if args.controller == "cv_mpc" else None
            ),
            "privileged_information_used": False if args.controller == "cv_mpc" else None,
            "hand_source": args.hand_source,
            "hand_model_path": str(hand_model_path) if hand_model_path is not None else None,
            "virtual_hand_policy_deterministic": False if args.hand_source == "virtual" else None,
            "virtual_hand_obs_dim": VIRTUAL_HAND_OBS_DIM if args.hand_source == "virtual" else None,
            "virtual_hand_history_length": VIRTUAL_HAND_HISTORY_LENGTH if args.hand_source == "virtual" else None,
            "virtual_hand_seed": int(args.seed) if args.hand_source == "virtual" else None,
            "virtual_hand_stride_cm": virtual_hand_stride,
            "virtual_hand_stride_semantics": (
                "componentwise policy-action scale, not guaranteed displacement"
                if args.hand_source == "virtual"
                else None
            ),
            "virtual_hand_stride_sample_range_cm": list(VIRTUAL_HAND_STRIDE_RANGE) if args.hand_source == "virtual" else None,
            "virtual_hand_initial_distance_cm": 0.5 * (args.zpd_low + args.zpd_high) if args.hand_source == "virtual" else None,
            "virtual_hand_initial_angle_rad": virtual_hand_initial_angle,
            "virtual_hand_initial_position_cm": virtual_hand_env.tolist() if virtual_hand_env is not None else None,
            "virtual_hand_execution_max_accel_scale": 1.5 if args.hand_source == "virtual" else None,
            "virtual_hand_pathology_mode": None,
            "virtual_hand_smoothing_alpha": float(args.hand_alpha) if args.hand_source == "virtual" else None,
            "virtual_hand_smoothing_formula": (
                "smoothed = alpha * delayed_intent + (1 - alpha) * previous_actual_move"
                if args.hand_source == "virtual"
                else None
            ),
            "virtual_hand_delay_frames": int(args.hand_delay_frames) if args.hand_source == "virtual" else None,
            "virtual_hand_delay_ms_target": (
                1000.0 * args.hand_delay_frames / args.control_hz
                if args.hand_source == "virtual"
                else None
            ),
            "virtual_hand_delay_semantics": (
                "N control-frame FIFO delay with zero initialization and pop-before-append"
                if args.hand_source == "virtual"
                else None
            ),
            "virtual_hand_execution_order": (
                [
                    "policy_action_clip",
                    "stride_scale",
                    "neural_delay",
                    "motion_smoothing",
                    "acceleration_clip",
                    "workspace_clip",
                ]
                if args.hand_source == "virtual"
                else None
            ),
            "virtual_hand_domain_randomization_fields": (
                ["motion_smoothing_alpha", "neural_delay_frames"]
                if args.hand_source == "virtual"
                else None
            ),
            "virtual_hand_existing_variation_fields": (
                ["hand_stride_cm", "virtual_hand_seed", "initial_angle"]
                if args.hand_source == "virtual"
                else None
            ),
            "virtual_hand_rom_randomization_enabled": False if args.hand_source == "virtual" else None,
            "virtual_hand_observation_noise_enabled": False if args.hand_source == "virtual" else None,
            "mediapipe_hand_detection_enabled": args.hand_source == "camera",
            "camera_enabled": True,
            "calibration_required": True,
            "microrobot_max_age_s": MICROROBOT_MAX_AGE_S,
            "microrobot_vision_mode_requested": args.microrobot_vision,
            "microrobot_vision_mode_effective": microrobot_vision_mode,
            "microrobot_yolo_enabled": bool(microrobot_yolo_enabled),
            "microrobot_yolo_disabled_reason": (
                microrobot_vision_disabled_reason
            ),
            "microrobot_detection_source": (
                "ultralytics_yolo_onnx"
                if microrobot_yolo_enabled
                else "disabled"
            ),
            "microrobot_detection_used_for_control": False,
            "task_robot_state_source": "ur_rtde_tcp",
            "distance_source": "ur_rtde_tcp",
            "vision_model_configured_path": (
                str(args.vision_model) if args.vision_model is not None else None
            ),
            "vision_model_path": (
                str(vision_model_path)
                if vision_model_path is not None
                else None
            ),
            "vision_model_loaded": bool(microrobot_yolo_enabled),
            "control_freq_target_hz": servo_timing.policy_hz,
            "policy_freq_target_hz": servo_timing.policy_hz,
            "policy_dt_target_s": servo_timing.policy_period_s,
            "servo_mode": servo_timing.mode,
            "servo_freq_target_hz": servo_timing.servo_hz,
            "servo_dt_target_s": servo_timing.servo_period_s,
            "servo_interpolation_mode": (
                "linear_hold" if servo_timing.mode == "interpolated" else "legacy_direct"
            ),
            "servo_target_timeout_s": servo_timing.target_timeout_s,
            "servo_max_speed_cm_s": servo_timing.max_speed_cm_s,
            "servo_max_translation_per_tick_m": (
                servo_timing.max_translation_per_tick_m
            ),
            "servo_lookahead_time_s": servo_timing.lookahead_time_s,
            "servo_gain": servo_timing.gain,
            "servo_period_api_available": bool(
                getattr(robot_control, "control_frequency_configured", False)
                and hasattr(robot_control.rtde_c, "initPeriod")
                and hasattr(robot_control.rtde_c, "waitPeriod")
            ),
            "servo_control_frequency_configured": bool(
                getattr(robot_control, "control_frequency_configured", False)
            ),
            "rtde_control_owner": (
                "interpolated_servo_thread_after_safe_transition"
                if servo_timing.mode == "interpolated"
                else "policy_loop_legacy"
            ),
            "virtual_hand_timing_semantics": (
                "updated once per policy step; no servo-rate dead reckoning"
                if args.hand_source == "virtual"
                else None
            ),
            "camera_index": int(args.camera),
            "camera_width_requested": int(args.camera_width),
            "camera_height_requested": int(args.camera_height),
            "workspace_width_cm": W_ENV,
            "workspace_height_cm": H_ENV,
            "workspace_margin_cm": WORKSPACE_MARGIN,
            "common_safety_guard_cm": COMMON_SAFETY_GUARD,
            "target_anchored_to_measured_pose": True,
            "duration_target_s": float(args.duration),
            "policy_obs_dim": int(expected_obs_dim) if args.controller == "league" else None,
            "history_mode": history_mode if args.controller == "league" else None,
            "history_length": int(history_length) if args.controller == "league" else None,
            "history_channels": int(history_channels) if args.controller == "league" else None,
            "interaction_history_feature_order": (
                [
                    "hand_minus_robot_x",
                    "hand_minus_robot_y",
                    "distance",
                    "distance_delta",
                    "robot_move_x",
                    "robot_move_y",
                    "hand_move_x",
                    "hand_move_y",
                ]
                if (
                    args.controller == "cv_mpc"
                    or (
                        args.controller == "league"
                        and history_mode == "interaction"
                    )
                )
                else None
            ),
            "previous_robot_move_source": "measured_tcp_delta",
            "stride_cm": float(args.stride),
            "max_safe_step_cm": float(args.max_step),
            "stop_on_catch": bool(args.stop_on_catch),
            "catch_distance_cm": float(args.catch_distance),
        }
        logger = DeploymentRolloutLogger(
            out_root=args.out_dir,
            subject=args.subject,
            condition=args.condition or (
                args.controller if args.hand_source == "camera" else f"{args.controller}_virtual_hand"
            ),
            metadata=metadata,
            zpd_low_cm=args.zpd_low,
            zpd_high_cm=args.zpd_high,
            save_video=args.save_video,
        )
        print(f"[log] {logger.rollout_dir}")

        vision_queue = queue.Queue(maxsize=1)
        vision_thread = VisionThread(
            cap,
            cali,
            cv_model,
            w_px,
            h_px,
            W_ENV,
            H_ENV,
            vision_queue,
            detect_hands=args.hand_source == "camera",
            detect_microrobot=microrobot_yolo_enabled,
        )
        vision_thread.start()

        virtual_target_env = initial_robot_env.copy()
        desired_target_env = initial_robot_env.copy()
        virtual_target_pixel = np.array([
            virtual_target_env[0] * w_px / W_ENV,
            virtual_target_env[1] * h_px / H_ENV,
        ], dtype=np.float32)
        desired_virtual_target = virtual_target_pixel.copy()
        hand_tracker = (
            HandTracker(w_env=W_ENV, h_env=H_ENV, vel_alpha=0.6)
            if args.hand_source == "camera"
            else None
        )
        virtual_hand_robot_history = deque(
            [np.zeros(MOTION_HISTORY_CHANNELS, dtype=np.float32)] * VIRTUAL_HAND_HISTORY_LENGTH,
            maxlen=VIRTUAL_HAND_HISTORY_LENGTH,
        )
        last_hand_actual_move = np.zeros(2, dtype=np.float32)
        virtual_hand_delay_buffer = (
            make_virtual_hand_delay_buffer(args.hand_delay_frames)
            if args.hand_source == "virtual"
            else None
        )
        prev_robot_env_for_hand = initial_robot_env.copy() if args.hand_source == "virtual" else None
        motion_history_buffer = deque([np.zeros(MOTION_HISTORY_CHANNELS, dtype=np.float32)] * history_length, maxlen=history_length)
        interaction_history_buffer = deque([np.zeros(INTERACTION_HISTORY_CHANNELS, dtype=np.float32)] * history_length, maxlen=history_length)
        last_action = np.zeros(2, dtype=np.float32)
        mpc_state = SimpleNamespace(
            steps=0,
            robot_position=np.zeros(2, dtype=np.float32),
            hand_position=np.zeros(2, dtype=np.float32),
            last_robot_action=np.zeros(2, dtype=np.float32),
            stride_robot=float(args.stride),
            margin=0.3,
            env_width=W_ENV,
            env_height=H_ENV,
            zpd_min=float(args.zpd_low),
            zpd_max=float(args.zpd_high),
            reward_step=0.2,
            distance_threshold_collision=float(args.catch_distance),
        )
        prev_hand_env = None
        prev_robot_env = None
        prev_distance_cm = None
        cached_frame = workspace_frame.copy()
        cached_microrobot_env = np.full(2, np.nan, dtype=np.float32)
        cached_microrobot_detected = False
        cached_microrobot_t_perf = None

        if servo_timing.mode == "interpolated":
            initial_servo_pose = np.array([
                initial_robot_pose[0],
                initial_robot_pose[1],
                Z,
                RX_C,
                RY_C,
                RZ_C,
            ], dtype=np.float64)
            servo_thread = InterpolatedServoThread(
                robot_control,
                initial_servo_pose,
                servo_timing,
            )
            servo_thread.start()
            logger.record_event(
                "servo_thread_started",
                policy_hz=servo_timing.policy_hz,
                servo_hz=servo_timing.servo_hz,
                target_timeout_s=servo_timing.target_timeout_s,
                period_api_available=servo_thread.period_api_available,
            )

        start_perf = time.perf_counter()
        next_policy_deadline = start_perf
        last_loop_start = None
        last_status_second = -1
        terminal_step_recorded = False
        step = 0

        while True:
            loop_start = time.perf_counter()
            if servo_thread is not None:
                servo_thread.raise_if_failed()
            policy_deadline_overrun_s = max(
                0.0,
                loop_start - next_policy_deadline,
            )
            t_task_s = loop_start - start_perf
            if t_task_s >= args.duration:
                done_reason = "timeout"
                break

            control_dt_s = None if last_loop_start is None else loop_start - last_loop_start
            control_hz = None if not control_dt_s or control_dt_s <= 1e-6 else 1.0 / control_dt_s
            last_loop_start = loop_start

            try:
                new_vision = vision_queue.get_nowait()
            except queue.Empty:
                new_vision = None

            vision_frame_available = new_vision is not None
            hand_detected = False
            microrobot_detected = False
            vision_age_s = None
            camera_dt_s = None
            camera_hz_inst = None
            frame_id = None
            if new_vision is not None:
                hand_detected = bool(new_vision.hand_detected)
                microrobot_detected = bool(new_vision.microrobot_detected)
                vision_age_s = loop_start - new_vision.processed_t_perf
                camera_dt_s = new_vision.camera_dt_s
                camera_hz_inst = new_vision.camera_hz_inst
                frame_id = new_vision.frame_id
                cached_frame = new_vision.undistorted_frame.copy()
                if microrobot_detected:
                    cached_microrobot_env = new_vision.microrobot_env.copy()
                    cached_microrobot_detected = True
                    cached_microrobot_t_perf = new_vision.processed_t_perf
                if hand_detected and hand_tracker is not None:
                    hand_tracker.update_vision(new_vision.hand_env, loop_start)

            robot_pose, real_robot_pixel, position_robot_env = get_robot_env_position(robot_control, cali, w_px, h_px)
            robot_boundary_clearance = workspace_clearance_env(position_robot_env)
            if not np.all(np.isfinite(position_robot_env)) or robot_boundary_clearance <= 0.0:
                done_reason = "safety_robot_out_of_bounds"
                legacy_stop_error = None
                if servo_thread is not None:
                    servo_thread.request_stop(done_reason)
                else:
                    try:
                        robot_control.servo_stop()
                    except Exception as exc:
                        legacy_stop_error = exc
                logger.record_step({
                    "step": step,
                    "t_wall_s": time.time(),
                    "t_task_s": t_task_s,
                    "control_dt_s": control_dt_s,
                    "control_loop_hz_inst": control_hz,
                    "policy_dt_s": control_dt_s,
                    "policy_loop_hz_inst": control_hz,
                    "policy_deadline_overrun_s": policy_deadline_overrun_s,
                    "policy_deadline_overrun": policy_deadline_overrun_s > 1e-3,
                    "robot_x_cm": float(position_robot_env[0]),
                    "robot_y_cm": float(position_robot_env[1]),
                    "robot_world_x": float(robot_pose[0]),
                    "robot_world_y": float(robot_pose[1]),
                    "robot_world_z": float(robot_pose[2]),
                    **build_servo_log_fields(
                        servo_thread.snapshot() if servo_thread is not None else None,
                        robot_pose,
                    ),
                    "safety_stop": True,
                    "safety_reason": done_reason,
                    "task_finished": True,
                    "done_reason": done_reason,
                })
                terminal_step_recorded = True
                logger.record_event(
                    "safety_stop",
                    t_task_s=t_task_s,
                    reason=done_reason,
                    robot_position_env=position_robot_env.tolist(),
                    boundary_clearance_cm=robot_boundary_clearance,
                )
                if servo_thread is not None:
                    servo_thread.stop_and_join(done_reason)
                elif legacy_stop_error is not None:
                    raise legacy_stop_error
                break

            if args.hand_source == "camera":
                position_hand_env = hand_tracker.predict(loop_start)
                dead_reckoning_used = not hand_detected
                dead_reckoning_age_s = (
                    None
                    if hand_tracker.last_vision_time is None
                    else loop_start - hand_tracker.last_vision_time
                )
            else:
                position_hand_env = virtual_hand_env.copy()
                dead_reckoning_used = False
                dead_reckoning_age_s = None
                completed_robot_move = (
                    position_robot_env - prev_robot_env_for_hand
                ).astype(np.float32)
                virtual_hand_robot_history.append(completed_robot_move)
                prev_robot_env_for_hand = position_robot_env.copy()

            cached_microrobot_age_s = (
                None
                if cached_microrobot_t_perf is None
                else loop_start - cached_microrobot_t_perf
            )
            microrobot_fresh = bool(
                cached_microrobot_detected
                and cached_microrobot_age_s is not None
                and cached_microrobot_age_s <= MICROROBOT_MAX_AGE_S
            )
            distance_cm = float(np.linalg.norm(position_robot_env - position_hand_env))
            in_zpd = args.zpd_low <= distance_cm <= args.zpd_high
            if args.stop_on_catch and distance_cm < args.catch_distance:
                done_reason = "caught"
                legacy_stop_error = None
                if servo_thread is not None:
                    servo_thread.request_stop(done_reason)
                else:
                    try:
                        robot_control.servo_stop()
                    except Exception as exc:
                        legacy_stop_error = exc
                logger.record_step({
                    "step": step,
                    "t_wall_s": time.time(),
                    "t_task_s": t_task_s,
                    "control_dt_s": control_dt_s,
                    "control_loop_hz_inst": control_hz,
                    "policy_dt_s": control_dt_s,
                    "policy_loop_hz_inst": control_hz,
                    "policy_deadline_overrun_s": policy_deadline_overrun_s,
                    "policy_deadline_overrun": policy_deadline_overrun_s > 1e-3,
                    "vision_frame_available": vision_frame_available,
                    "vision_frame_id": frame_id,
                    "vision_age_s": vision_age_s,
                    "camera_dt_s": camera_dt_s,
                    "camera_hz_inst": camera_hz_inst,
                    "hand_detected": hand_detected,
                    "microrobot_detected": microrobot_detected,
                    "dead_reckoning_used": dead_reckoning_used,
                    "dead_reckoning_age_s": dead_reckoning_age_s,
                    "hand_x_cm": float(position_hand_env[0]),
                    "hand_y_cm": float(position_hand_env[1]),
                    "robot_x_cm": float(position_robot_env[0]),
                    "robot_y_cm": float(position_robot_env[1]),
                    "distance_cm": distance_cm,
                    "in_zpd": in_zpd,
                    "robot_world_x": float(robot_pose[0]),
                    "robot_world_y": float(robot_pose[1]),
                    "robot_world_z": float(robot_pose[2]),
                    **build_servo_log_fields(
                        servo_thread.snapshot() if servo_thread is not None else None,
                        robot_pose,
                    ),
                    "safety_stop": False,
                    "safety_reason": "",
                    "task_finished": True,
                    "done_reason": done_reason,
                })
                terminal_step_recorded = True
                logger.record_event(
                    "caught",
                    t_task_s=t_task_s,
                    distance_cm=distance_cm,
                    catch_distance_cm=float(args.catch_distance),
                    distance_source="ur_rtde_tcp",
                )
                if servo_thread is not None:
                    servo_thread.stop_and_join(done_reason)
                elif legacy_stop_error is not None:
                    raise legacy_stop_error
                break

            hand_move = (
                np.zeros(2, dtype=np.float32)
                if prev_hand_env is None
                else (position_hand_env - prev_hand_env).astype(np.float32)
            )
            robot_move = (
                np.zeros(2, dtype=np.float32)
                if prev_robot_env is None
                else (position_robot_env - prev_robot_env).astype(np.float32)
            )
            distance_delta = (
                0.0
                if prev_distance_cm is None
                else float(distance_cm - prev_distance_cm)
            )
            if (
                prev_hand_env is not None
                and prev_robot_env is not None
                and prev_distance_cm is not None
            ):
                motion_history_buffer.append(hand_move)
                interaction_history_buffer.append(np.array([
                    float(position_hand_env[0] - position_robot_env[0]),
                    float(position_hand_env[1] - position_robot_env[1]),
                    float(distance_cm),
                    distance_delta,
                    float(robot_move[0]),
                    float(robot_move[1]),
                    float(hand_move[0]),
                    float(hand_move[1]),
                ], dtype=np.float32))

            virtual_hand_obs = None
            if args.hand_source == "virtual":
                virtual_hand_obs = build_virtual_hand_observation(
                    position_hand_env,
                    position_robot_env,
                    virtual_hand_stride,
                    last_hand_actual_move,
                    virtual_hand_robot_history,
                )

            previous_action = last_action.copy()
            inference_start = time.perf_counter()
            if args.controller == "league":
                robot_obs = position_robot_env
                hand_obs = position_hand_env
                distance_obs = np.array([np.linalg.norm(robot_obs - hand_obs)], dtype=np.float32)
                boundary_obs = np.array([
                    robot_obs[0], W_ENV - robot_obs[0],
                    robot_obs[1], H_ENV - robot_obs[1],
                ], dtype=np.float32)
                if history_mode == "interaction":
                    flat_history = fit_history(interaction_history_buffer, history_length * history_channels)
                else:
                    flat_history = fit_history(motion_history_buffer, history_length * history_channels)
                obs_array = np.concatenate((
                    robot_obs,
                    hand_obs,
                    distance_obs,
                    boundary_obs,
                    np.array([args.stride], dtype=np.float32),
                    robot_move,
                    flat_history,
                )).astype(np.float32)
                if obs_array.shape[0] != expected_obs_dim:
                    raise RuntimeError(f"Built observation has {obs_array.shape[0]} dims, expected {expected_obs_dim}")
                action, _ = rl_model.predict(obs_array, deterministic=True)
            else:
                mpc_state.steps = step
                mpc_state.robot_position = position_robot_env.copy()
                mpc_state.hand_position = position_hand_env.copy()
                mpc_state.last_robot_action = robot_move.copy()
                action = mpc_controller.predict(
                    mpc_state,
                    interaction_history=interaction_history_buffer,
                    completed_steps=step,
                )
            policy_inference_ms = (time.perf_counter() - inference_start) * 1000.0

            next_virtual_hand_move = None
            virtual_hand_diagnostics = None
            virtual_hand_policy_inference_ms = None
            if args.hand_source == "virtual":
                hand_inference_start = time.perf_counter()
                hand_action, _ = hand_model.predict(virtual_hand_obs, deterministic=False)
                virtual_hand_policy_inference_ms = (
                    time.perf_counter() - hand_inference_start
                ) * 1000.0
                next_virtual_hand_move, virtual_hand_diagnostics = (
                    apply_virtual_hand_execution(
                        hand_action,
                        virtual_hand_stride,
                        last_hand_actual_move,
                        smoothing_alpha=args.hand_alpha,
                        delay_buffer=virtual_hand_delay_buffer,
                    )
                )

            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            virtual_target_env, desired_target_env, target_clipped, target_step_limited = (
                resolve_common_target_env(
                    action,
                    position_robot_env,
                    args.stride,
                    args.max_step,
                )
            )
            virtual_target_pixel = np.array([
                virtual_target_env[0] * w_px / W_ENV,
                virtual_target_env[1] * h_px / H_ENV,
            ], dtype=np.float32)
            desired_virtual_target = np.array([
                desired_target_env[0] * w_px / W_ENV,
                desired_target_env[1] * h_px / H_ENV,
            ], dtype=np.float32)

            target_position_world = cali.pixel_to_world(virtual_target_pixel.astype(int))
            target_pose = [target_position_world[0], target_position_world[1], Z, RX_C, RY_C, RZ_C]
            if servo_thread is not None:
                servo_thread.publish_target(target_pose, policy_step=step)
            else:
                safe_dt = float(np.clip(
                    control_dt_s or servo_timing.policy_period_s,
                    0.01,
                    0.2,
                ))
                servo_ok = robot_control.servo_robot(
                    target_pose,
                    dt=safe_dt,
                    lookahead_time=servo_timing.lookahead_time_s,
                    gain=servo_timing.gain,
                )
                if servo_ok is False:
                    raise RuntimeError("servoL returned failure")
            last_action = action.copy()

            if args.hand_source == "virtual":
                virtual_hand_pre_clip = virtual_hand_env + next_virtual_hand_move
                virtual_hand_low = np.array(
                    [WORKSPACE_MARGIN, WORKSPACE_MARGIN],
                    dtype=np.float32,
                )
                virtual_hand_high = np.array(
                    [W_ENV - WORKSPACE_MARGIN, H_ENV - WORKSPACE_MARGIN],
                    dtype=np.float32,
                )
                workspace_clipped = bool(
                    np.any(virtual_hand_pre_clip < virtual_hand_low)
                    or np.any(virtual_hand_pre_clip > virtual_hand_high)
                )
                next_virtual_hand_env = np.clip(
                    virtual_hand_pre_clip,
                    virtual_hand_low,
                    virtual_hand_high,
                ).astype(np.float32)
                virtual_hand_actual_move = (
                    next_virtual_hand_env - virtual_hand_env
                ).astype(np.float32)
                virtual_hand_diagnostics["actual_move"] = virtual_hand_actual_move
                virtual_hand_diagnostics["actual_norm_cm"] = float(
                    np.linalg.norm(virtual_hand_actual_move)
                )
                virtual_hand_diagnostics["workspace_clipped"] = workspace_clipped
                last_hand_actual_move = virtual_hand_actual_move.copy()
                virtual_hand_env = next_virtual_hand_env

            task_finished = False
            step_done_reason = ""

            should_render_frame = bool(
                cached_frame is not None
                and (
                    not args.no_display
                    or args.save_video
                    or vision_frame_available
                    or step == 0
                )
            )
            frame_for_log = cached_frame.copy() if should_render_frame else None
            if frame_for_log is not None:
                cv2.circle(frame_for_log, (int(real_robot_pixel[0]), int(real_robot_pixel[1])), 16, (0, 180, 255), -1)
                cv2.circle(frame_for_log, (int(virtual_target_pixel[0]), int(virtual_target_pixel[1])), 10, (255, 0, 255), 2)
                if args.hand_source == "virtual":
                    virtual_hand_pixel = np.array([
                        position_hand_env[0] * w_px / W_ENV,
                        position_hand_env[1] * h_px / H_ENV,
                    ])
                    hand_pixel_tuple = (int(virtual_hand_pixel[0]), int(virtual_hand_pixel[1]))
                    cv2.circle(frame_for_log, hand_pixel_tuple, 16, (255, 255, 0), -1)
                    cv2.putText(
                        frame_for_log,
                        "Virtual Hand",
                        (hand_pixel_tuple[0] + 18, hand_pixel_tuple[1] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )
                if microrobot_fresh and np.all(np.isfinite(cached_microrobot_env)):
                    mic_px = np.array([cached_microrobot_env[0] * w_px / W_ENV, cached_microrobot_env[1] * h_px / H_ENV])
                    cv2.circle(frame_for_log, (int(mic_px[0]), int(mic_px[1])), 14, (255, 100, 0), 2)
                cv2.putText(frame_for_log, f"d={distance_cm:.2f} cm", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 220, 30), 2)
                cv2.putText(frame_for_log, f"t={t_task_s:.1f}s", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 220, 30), 2)
                logger.maybe_save_snapshot(frame_for_log, step, t_task_s, preferred_step=args.snapshot_step, preferred_time_s=args.duration / 2.0)
                logger.write_frame(frame_for_log, fps=args.control_hz)
                if not args.no_display:
                    cv2.imshow("Deployment Chase", frame_for_log)

            servo_snapshot = (
                servo_thread.snapshot()
                if servo_thread is not None
                else None
            )
            logger.record_step({
                "step": step,
                "t_wall_s": time.time(),
                "t_task_s": t_task_s,
                "control_dt_s": control_dt_s,
                "control_loop_hz_inst": control_hz,
                "policy_dt_s": control_dt_s,
                "policy_loop_hz_inst": control_hz,
                "policy_deadline_overrun_s": policy_deadline_overrun_s,
                "policy_deadline_overrun": policy_deadline_overrun_s > 1e-3,
                "vision_frame_available": vision_frame_available,
                "vision_frame_id": frame_id,
                "vision_age_s": vision_age_s,
                "camera_dt_s": camera_dt_s,
                "camera_hz_inst": camera_hz_inst,
                "hand_detected": hand_detected,
                "microrobot_detected": microrobot_detected,
                "dead_reckoning_used": dead_reckoning_used,
                "dead_reckoning_age_s": dead_reckoning_age_s,
                "hand_x_cm": float(position_hand_env[0]),
                "hand_y_cm": float(position_hand_env[1]),
                "robot_x_cm": float(position_robot_env[0]),
                "robot_y_cm": float(position_robot_env[1]),
                "microrobot_x_cm": float(cached_microrobot_env[0]) if microrobot_fresh and np.isfinite(cached_microrobot_env[0]) else None,
                "microrobot_y_cm": float(cached_microrobot_env[1]) if microrobot_fresh and np.isfinite(cached_microrobot_env[1]) else None,
                "distance_cm": distance_cm,
                "in_zpd": in_zpd,
                **build_virtual_hand_log_fields(
                    virtual_hand_diagnostics,
                    virtual_hand_stride,
                    virtual_hand_policy_inference_ms,
                ),
                "policy_inference_ms": policy_inference_ms,
                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "action_norm": float(np.linalg.norm(action)),
                "last_action_x": float(previous_action[0]),
                "last_action_y": float(previous_action[1]),
                "virtual_target_x_px": float(virtual_target_pixel[0]),
                "virtual_target_y_px": float(virtual_target_pixel[1]),
                "desired_target_x_px": float(desired_virtual_target[0]),
                "desired_target_y_px": float(desired_virtual_target[1]),
                "target_clipped": target_clipped,
                "target_step_limited": target_step_limited,
                "robot_world_x": float(robot_pose[0]),
                "robot_world_y": float(robot_pose[1]),
                "robot_world_z": float(robot_pose[2]),
                **build_servo_log_fields(servo_snapshot, robot_pose),
                "safety_stop": False,
                "safety_reason": "",
                "task_finished": task_finished,
                "done_reason": step_done_reason,
            })

            prev_hand_env = position_hand_env.copy()
            prev_robot_env = position_robot_env.copy()
            prev_distance_cm = distance_cm

            if int(t_task_s) != last_status_second:
                last_status_second = int(t_task_s)
                print(
                    f"[progress][{controller_label(args.controller)}] "
                    f"t={t_task_s:5.1f}/{args.duration:.1f} s  |  "
                    f"distance={distance_cm:4.2f} cm  |  "
                    f"ZPD={'IN' if in_zpd else 'OUT'}"
                )

            if not args.no_display and cv2.waitKey(1) == ord("q"):
                done_reason = "manual_q"
                logger.record_event("manual_stop", t_task_s=t_task_s)
                break

            step += 1
            next_policy_deadline += servo_timing.policy_period_s
            sleep_time = next_policy_deadline - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            elif -sleep_time > servo_timing.policy_period_s:
                next_policy_deadline = time.perf_counter()

        if done_reason == "unknown":
            done_reason = "timeout"

    except ServoLoopFailure as exc:
        done_reason = (
            "safety_servo_watchdog"
            if isinstance(exc, ServoLoopSafetyStop)
            else "safety_servo_failure"
        )
        print(f"[safety] {exc}")
        if logger is not None:
            terminal_pose = locals().get("robot_pose")
            terminal_row = {
                "step": int(locals().get("step", 0)),
                "t_wall_s": time.time(),
                "t_task_s": float(locals().get("t_task_s", 0.0)),
                "control_dt_s": locals().get("control_dt_s"),
                "control_loop_hz_inst": locals().get("control_hz"),
                "policy_dt_s": locals().get("control_dt_s"),
                "policy_loop_hz_inst": locals().get("control_hz"),
                "policy_deadline_overrun_s": locals().get(
                    "policy_deadline_overrun_s",
                    0.0,
                ),
                "policy_deadline_overrun": True,
                "safety_stop": True,
                "safety_reason": done_reason,
                "task_finished": True,
                "done_reason": done_reason,
            }
            terminal_row.update(build_servo_log_fields(
                servo_thread.snapshot() if servo_thread is not None else None,
                terminal_pose,
            ))
            terminal_robot_env = locals().get("position_robot_env")
            if terminal_robot_env is not None:
                terminal_row.update({
                    "robot_x_cm": float(terminal_robot_env[0]),
                    "robot_y_cm": float(terminal_robot_env[1]),
                })
            terminal_hand_env = locals().get("position_hand_env")
            if terminal_hand_env is not None:
                terminal_row.update({
                    "hand_x_cm": float(terminal_hand_env[0]),
                    "hand_y_cm": float(terminal_hand_env[1]),
                })
            if "distance_cm" in locals():
                terminal_row["distance_cm"] = float(distance_cm)
                terminal_row["in_zpd"] = bool(locals().get("in_zpd", False))
            if (
                terminal_pose is not None
                and np.asarray(terminal_pose).shape == (6,)
                and np.all(np.isfinite(terminal_pose))
            ):
                terminal_row.update({
                    "robot_world_x": float(terminal_pose[0]),
                    "robot_world_y": float(terminal_pose[1]),
                    "robot_world_z": float(terminal_pose[2]),
                })
            if not locals().get("terminal_step_recorded", False):
                logger.record_step(terminal_row)
            logger.record_event(
                "safety_stop",
                reason=done_reason,
                error=str(exc),
            )
    except KeyboardInterrupt:
        done_reason = "keyboard_interrupt"
        print("[stop] keyboard interrupt")
    except Exception as exc:
        if done_reason == "unknown":
            done_reason = "exception"
        print(f"[error] {exc}")
        traceback.print_exc()
        if logger is not None:
            logger.record_event("exception", error=str(exc))
    finally:
        print("[shutdown] stopping safely")
        servo_cleanup_error = None
        servo_thread_alive_after_cleanup = False
        if servo_thread is not None:
            try:
                servo_thread.stop_and_join(done_reason or "rollout_shutdown")
            except Exception as exc:
                servo_cleanup_error = exc
                servo_thread_alive_after_cleanup = servo_thread.is_alive()
                print(f"[shutdown] servo cleanup warning: {exc}")
                if servo_thread_alive_after_cleanup:
                    done_reason = "safety_servo_thread_unresponsive"
                    print(
                        "[shutdown] CRITICAL: servo thread is still active; "
                        "skipping concurrent RTDE disconnect"
                    )
                if logger is not None:
                    logger.record_event(
                        "servo_cleanup_error",
                        error=str(exc),
                        thread_alive=servo_thread_alive_after_cleanup,
                    )
                    if servo_thread_alive_after_cleanup:
                        cleanup_pose = locals().get("robot_pose")
                        cleanup_row = {
                            "step": int(locals().get("step", 0)) + 1,
                            "t_wall_s": time.time(),
                            "t_task_s": float(locals().get("t_task_s", 0.0)),
                            "safety_stop": True,
                            "safety_reason": done_reason,
                            "task_finished": True,
                            "done_reason": done_reason,
                        }
                        cleanup_row.update(build_servo_log_fields(
                            servo_thread.snapshot(),
                            cleanup_pose,
                        ))
                        logger.record_step(cleanup_row)
        elif robot_control is not None:
            try:
                robot_control.servo_stop()
            except Exception as exc:
                servo_cleanup_error = exc
                print(f"[shutdown] servo stop warning: {exc}")

        if logger is not None:
            summary = logger.close(done_reason=done_reason)
            tiz = summary.get("tiz_fixed_horizon_fraction")
            observed_zpd = summary.get("zpd_observed_occupancy_fraction")
            duration_s = summary.get("duration_s")
            print("-" * 78)
            print("ROLLOUT SUMMARY")
            print("-" * 78)
            print(f"Controller   : {controller_label(args.controller)}")
            print(f"Done reason  : {done_reason}")
            print(
                f"Duration     : "
                f"{f'{float(duration_s):.1f} s' if duration_s is not None else 'N/A'}"
            )
            print(
                f"Fixed TIZ    : "
                f"{f'{100.0 * float(tiz):.1f}%' if tiz is not None else 'N/A'}"
            )
            print(
                f"Observed ZPD : "
                f"{f'{100.0 * float(observed_zpd):.1f}%' if observed_zpd is not None else 'N/A'}"
            )
            print(f"Summary file : {logger.rollout_dir / 'summary.json'}")
            print("-" * 78, flush=True)
        if vision_thread is not None:
            vision_thread.stop()
            vision_thread.join(timeout=2.0)
        if robot_control is not None and not servo_thread_alive_after_cleanup:
            try:
                time.sleep(0.5)
                robot_control.disconnect(
                    stop=servo_thread is None or servo_cleanup_error is not None
                )
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
