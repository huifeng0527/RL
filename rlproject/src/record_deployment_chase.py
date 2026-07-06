import argparse
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

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



W_ENV = 15.0
H_ENV = 10.0
DEFAULT_CONTROL_FREQ = 20.0
RX_C, RY_C, RZ_C = 0.193, 0.067, 5.3
DEFAULT_STRIDE = 0.3
DEFAULT_MAX_SAFE_STRIDE = 0.6

DEFAULT_POLICY_CANDIDATES = [
    REPO_ROOT / "logs" / "league_zpd35_55_noid_warm_entropy_10iter_r5m_h1m_gru_noaux" / "iteration_10" / "robot" / "robot" / "best_model.zip",
    REPO_ROOT / "logs" / "league_paper_gru_multistep_aux_pfsp_window_20iter" / "iteration_20" / "robot" / "robot" / "best_model.zip",
    REPO_ROOT / "rlproject" / "best_model.zip",
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
    def __init__(self, cap, cali, cv_model, w_px, h_px, w_env, h_env, result_queue):
        super().__init__(daemon=True)
        self.cap = cap
        self.cali = cali
        self.cv_model = cv_model
        self.w_px = int(w_px)
        self.h_px = int(h_px)
        self.w_env = float(w_env)
        self.h_env = float(h_env)
        self.result_queue = result_queue
        self.hand_detector = HandDetection()
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
            results = self.cv_model.predict(undistorted_frame, conf=0.7, save=False, imgsz=640, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x2 - x1 > 100:
                        continue
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    robot_trajectory.append((int(cx), int(cy)))
                    microrobot_env = np.array([cx * self.w_env / self.w_px, cy * self.h_env / self.h_px], dtype=np.float32)
                    pixel_per_cm = ((x2 - x1) + (y2 - y1)) / 4
                    cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (255, 120, 20), 2)

            undistorted_frame, hand_positions = self.hand_detector.process_frame(undistorted_frame)
            hand_env = np.zeros(2, dtype=np.float32)
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

        self.hand_detector.release()
        print("[vision] stopped")

    def stop(self):
        self.running = False


def find_default_policy():
    for path in DEFAULT_POLICY_CANDIDATES:
        if path.exists():
            return path
    return None


def parse_args():
    default_policy = find_default_policy()
    parser = argparse.ArgumentParser(description="Record one real-world chase rollout for the paper deployment figure.")
    parser.add_argument("--model", "--policy", dest="model", default=str(default_policy) if default_policy else None, help="PPO robot policy .zip. If omitted, the script uses the default league robot model.")
    parser.add_argument("--vision-model", default=str(DEFAULT_VISION_MODEL), help=argparse.SUPPRESS)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "deployment_rollouts"), help=argparse.SUPPRESS)
    parser.add_argument("--subject", default="pilot", help="Short subject/session label.")
    parser.add_argument("--condition", default="zero_shot_physical", help="Condition label written to metadata.")
    parser.add_argument("--seconds", "--duration", dest="duration", type=float, default=30.0, help="Rollout duration in seconds.")
    parser.add_argument("--zpd-low", type=float, default=3.5, help=argparse.SUPPRESS)
    parser.add_argument("--zpd-high", type=float, default=5.5, help=argparse.SUPPRESS)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_FREQ, help=argparse.SUPPRESS)
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
    parser.add_argument("--countdown", type=int, default=3, help=argparse.SUPPRESS)
    return parser.parse_args()


def require_file(path, label):
    if path is None:
        raise FileNotFoundError(f"No default {label} found; pass it explicitly.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_runtime_dependencies():
    global cv2, PPO, YOLO, CameraCalibration, DeploymentRolloutLogger, HandDetection, URControl, get_workspace

    import cv2 as cv2_module
    from stable_baselines3 import PPO as ppo_class
    from ultralytics import YOLO as yolo_class

    from camera_calibration.camera_calibration import CameraCalibration as calibration_class
    from callbacks.deployment_rollout_logger import DeploymentRolloutLogger as logger_class
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
    robot_control.rtde_c.servoStop()
    center_pixel = np.array([w_px / 2, h_px / 2], dtype=np.float64)
    center_world = cali.pixel_to_world(center_pixel.astype(int))
    target_pose = [center_world[0], center_world[1], 0.116, RX_C, RY_C, RZ_C]
    robot_control.rtde_c.moveL(target_pose, 0.2, 0.2, asynchronous=False)
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


def main():
    args = parse_args()
    model_path = require_file(args.model, "PPO policy")
    vision_model_path = require_file(args.vision_model, "vision model")
    load_runtime_dependencies()

    print(f"[model] policy: {model_path}")
    print(f"[model] vision: {vision_model_path}")

    cv_model = YOLO(str(vision_model_path))
    cali = CameraCalibration()
    robot_control = None
    cap = None
    vision_thread = None
    logger = None
    done_reason = "unknown"

    try:
        robot_control = URControl(args.robot_ip)
        rl_model = PPO.load(str(model_path), custom_objects={"learning_rate": 0.0, "optimizer_class": None})

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

        metadata = {
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "task": "DeploymentChase",
            "policy_path": str(model_path),
            "vision_model_path": str(vision_model_path),
            "control_freq_target_hz": float(args.control_hz),
            "camera_index": int(args.camera),
            "camera_width_requested": int(args.camera_width),
            "camera_height_requested": int(args.camera_height),
            "workspace_width_cm": W_ENV,
            "workspace_height_cm": H_ENV,
            "duration_target_s": float(args.duration),
            "stride_cm": float(args.stride),
            "max_safe_step_cm": float(args.max_step),
            "stop_on_catch": bool(args.stop_on_catch),
            "catch_distance_cm": float(args.catch_distance),
        }
        logger = DeploymentRolloutLogger(
            out_root=args.out_dir,
            subject=args.subject,
            condition=args.condition,
            metadata=metadata,
            zpd_low_cm=args.zpd_low,
            zpd_high_cm=args.zpd_high,
            save_video=args.save_video,
        )
        print(f"[log] {logger.rollout_dir}")

        vision_queue = queue.Queue(maxsize=1)
        vision_thread = VisionThread(cap, cali, cv_model, w_px, h_px, W_ENV, H_ENV, vision_queue)
        vision_thread.start()

        center_pixel = safe_transition_to_center(robot_control, cali, w_px, h_px, args.countdown)
        virtual_target_pixel = center_pixel.copy()
        desired_virtual_target = center_pixel.copy()
        hand_tracker = HandTracker(w_env=W_ENV, h_env=H_ENV, vel_alpha=0.6)
        hand_history_buffer = deque([np.zeros(2, dtype=np.float32)] * 16, maxlen=16)
        last_action = np.zeros(2, dtype=np.float32)
        cached_frame = workspace_frame.copy()
        cached_microrobot_env = np.full(2, np.nan, dtype=np.float32)
        cached_microrobot_detected = False

        start_perf = time.perf_counter()
        last_loop_start = None
        last_status_second = -1
        step = 0

        while True:
            loop_start = time.perf_counter()
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
                if hand_detected:
                    hand_tracker.update_vision(new_vision.hand_env, loop_start)

            position_hand_env = hand_tracker.predict(loop_start)
            dead_reckoning_used = not hand_detected
            dead_reckoning_age_s = None if hand_tracker.last_vision_time is None else loop_start - hand_tracker.last_vision_time
            hand_history_buffer.append(hand_tracker.vel * (1.0 / args.control_hz))

            robot_pose, real_robot_pixel, position_robot_env = get_robot_env_position(robot_control, cali, w_px, h_px)
            distance_target_env = cached_microrobot_env if cached_microrobot_detected else position_robot_env
            distance_cm = float(np.linalg.norm(distance_target_env - position_hand_env))
            in_zpd = args.zpd_low <= distance_cm <= args.zpd_high

            robot_obs = position_robot_env
            hand_obs = position_hand_env
            distance_obs = np.array([np.linalg.norm(robot_obs - hand_obs)], dtype=np.float32)
            boundary_obs = np.array([
                robot_obs[0], W_ENV - robot_obs[0],
                robot_obs[1], H_ENV - robot_obs[1],
            ], dtype=np.float32)
            flat_history = np.array(hand_history_buffer).flatten()
            previous_action = last_action.copy()
            obs_array = np.concatenate((
                robot_obs,
                hand_obs,
                distance_obs,
                boundary_obs,
                np.array([args.stride], dtype=np.float32),
                previous_action,
                flat_history,
            )).astype(np.float32)

            inference_start = time.perf_counter()
            action, _ = rl_model.predict(obs_array, deterministic=True)
            policy_inference_ms = (time.perf_counter() - inference_start) * 1000.0
            action = np.asarray(action, dtype=np.float32)
            action_pixel = action * np.array([w_px / W_ENV, h_px / H_ENV], dtype=np.float32) * args.stride

            desired_virtual_target = virtual_target_pixel.copy() + action_pixel
            desired_before_clip = desired_virtual_target.copy()
            desired_virtual_target[0] = np.clip(desired_virtual_target[0], 50, w_px - 50)
            desired_virtual_target[1] = np.clip(desired_virtual_target[1], 50, h_px - 50)
            target_clipped = not np.allclose(desired_before_clip, desired_virtual_target)

            max_pixel_step = args.max_step * (w_px / W_ENV)
            diff_vec = desired_virtual_target - virtual_target_pixel
            dist_pixel = float(np.linalg.norm(diff_vec))
            target_step_limited = dist_pixel > max_pixel_step
            if target_step_limited:
                virtual_target_pixel += (diff_vec / dist_pixel) * max_pixel_step
            else:
                virtual_target_pixel = desired_virtual_target.copy()

            target_position_world = cali.pixel_to_world(virtual_target_pixel.astype(int))
            target_pose = [target_position_world[0], target_position_world[1], 0.116, RX_C, RY_C, RZ_C]
            safe_dt = float(np.clip(control_dt_s or (1.0 / args.control_hz), 0.01, 0.2))
            robot_control.servo_robot(target_pose, dt=safe_dt)
            last_action = action.copy()

            task_finished = bool(args.stop_on_catch and distance_cm < args.catch_distance)
            step_done_reason = "caught" if task_finished else ""

            frame_for_log = cached_frame.copy() if cached_frame is not None else None
            if frame_for_log is not None:
                cv2.circle(frame_for_log, (int(real_robot_pixel[0]), int(real_robot_pixel[1])), 16, (0, 180, 255), -1)
                cv2.circle(frame_for_log, (int(virtual_target_pixel[0]), int(virtual_target_pixel[1])), 10, (255, 0, 255), 2)
                if cached_microrobot_detected and np.all(np.isfinite(cached_microrobot_env)):
                    mic_px = np.array([cached_microrobot_env[0] * w_px / W_ENV, cached_microrobot_env[1] * h_px / H_ENV])
                    cv2.circle(frame_for_log, (int(mic_px[0]), int(mic_px[1])), 14, (255, 100, 0), 2)
                cv2.putText(frame_for_log, f"d={distance_cm:.2f} cm", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 220, 30), 2)
                cv2.putText(frame_for_log, f"t={t_task_s:.1f}s", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 220, 30), 2)
                logger.maybe_save_snapshot(frame_for_log, step, t_task_s, preferred_step=args.snapshot_step, preferred_time_s=args.duration / 2.0)
                logger.write_frame(frame_for_log, fps=args.control_hz)
                if not args.no_display:
                    cv2.imshow("Deployment Chase", frame_for_log)

            logger.record_step({
                "step": step,
                "t_wall_s": time.time(),
                "t_task_s": t_task_s,
                "control_dt_s": control_dt_s,
                "control_loop_hz_inst": control_hz,
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
                "microrobot_x_cm": float(cached_microrobot_env[0]) if np.isfinite(cached_microrobot_env[0]) else None,
                "microrobot_y_cm": float(cached_microrobot_env[1]) if np.isfinite(cached_microrobot_env[1]) else None,
                "distance_cm": distance_cm,
                "in_zpd": in_zpd,
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
                "safety_stop": False,
                "safety_reason": "",
                "task_finished": task_finished,
                "done_reason": step_done_reason,
            })

            if int(t_task_s) != last_status_second:
                last_status_second = int(t_task_s)
                print(f"[run] t={t_task_s:5.1f}s d={distance_cm:4.2f}cm zpd={int(in_zpd)}")

            if not args.no_display and cv2.waitKey(1) == ord("q"):
                done_reason = "manual_q"
                logger.record_event("manual_stop", t_task_s=t_task_s)
                break

            if task_finished:
                done_reason = "caught"
                logger.record_event("caught", t_task_s=t_task_s, distance_cm=distance_cm)
                break

            step += 1
            elapsed = time.perf_counter() - loop_start
            sleep_time = (1.0 / args.control_hz) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if done_reason == "unknown":
            done_reason = "timeout"

    except KeyboardInterrupt:
        done_reason = "keyboard_interrupt"
        print("[stop] keyboard interrupt")
    except Exception as exc:
        done_reason = "exception"
        print(f"[error] {exc}")
        traceback.print_exc()
        if logger is not None:
            logger.record_event("exception", error=str(exc))
    finally:
        print("[shutdown] stopping safely")
        if logger is not None:
            summary = logger.close(done_reason=done_reason)
            print(f"[summary] {logger.rollout_dir / 'summary.json'}")
            if summary.get("zpd_occupancy_fraction") is not None:
                print(f"[summary] ZPD occupancy: {summary['zpd_occupancy_fraction'] * 100:.1f}%")
        if vision_thread is not None:
            vision_thread.stop()
            vision_thread.join(timeout=2.0)
        if robot_control is not None:
            try:
                robot_control.rtde_c.servoStop()
                time.sleep(0.5)
                robot_control.disconnect()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
