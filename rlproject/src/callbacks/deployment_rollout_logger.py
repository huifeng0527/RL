import csv
import json
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from ..deployment_metrics import compute_fixed_horizon_tiz
except ImportError:
    from deployment_metrics import compute_fixed_horizon_tiz


FIELDNAMES = [
    "step",
    "t_wall_s",
    "t_task_s",
    "control_dt_s",
    "control_loop_hz_inst",
    "policy_dt_s",
    "policy_loop_hz_inst",
    "policy_deadline_overrun_s",
    "policy_deadline_overrun",
    "servo_dt_s",
    "servo_loop_hz_inst",
    "servo_target_age_s",
    "servo_interpolation_phase",
    "servo_command_sequence",
    "servo_policy_step",
    "servo_commanded_world_x",
    "servo_commanded_world_y",
    "servo_commanded_world_z",
    "servo_target_world_x",
    "servo_target_world_y",
    "servo_target_world_z",
    "servo_segment_start_world_x",
    "servo_segment_start_world_y",
    "servo_segment_start_world_z",
    "servo_segment_end_world_x",
    "servo_segment_end_world_y",
    "servo_segment_end_world_z",
    "servo_segment_start_perf",
    "servo_segment_end_perf",
    "servo_segment_elapsed_s",
    "servo_segment_duration_s",
    "servo_pending_segment_count",
    "servo_command_step_limited",
    "servo_command_step_limited_count",
    "servo_publish_rejected_count",
    "servo_tracking_error_cm",
    "servo_target_error_cm",
    "servo_deadline_overrun_s",
    "servo_deadline_overrun_count",
    "servo_watchdog_stopped",
    "servo_stop_reason",
    "vision_frame_available",
    "vision_frame_id",
    "vision_age_s",
    "camera_dt_s",
    "camera_hz_inst",
    "hand_detected",
    "microrobot_detected",
    "dead_reckoning_used",
    "dead_reckoning_age_s",
    "hand_x_cm",
    "hand_y_cm",
    "robot_x_cm",
    "robot_y_cm",
    "microrobot_x_cm",
    "microrobot_y_cm",
    "distance_cm",
    "in_zpd",
    "virtual_hand_stride_cm",
    "virtual_hand_smoothing_alpha",
    "virtual_hand_delay_frames",
    "virtual_hand_action_x",
    "virtual_hand_action_y",
    "virtual_hand_action_norm",
    "virtual_hand_command_dx_cm",
    "virtual_hand_command_dy_cm",
    "virtual_hand_command_norm_cm",
    "virtual_hand_delayed_dx_cm",
    "virtual_hand_delayed_dy_cm",
    "virtual_hand_delayed_norm_cm",
    "virtual_hand_smoothed_dx_cm",
    "virtual_hand_smoothed_dy_cm",
    "virtual_hand_smoothed_norm_cm",
    "virtual_hand_exec_dx_cm",
    "virtual_hand_exec_dy_cm",
    "virtual_hand_exec_norm_cm",
    "virtual_hand_actual_dx_cm",
    "virtual_hand_actual_dy_cm",
    "virtual_hand_actual_norm_cm",
    "virtual_hand_accel_clipped",
    "virtual_hand_workspace_clipped",
    "virtual_hand_policy_inference_ms",
    "policy_inference_ms",
    "action_x",
    "action_y",
    "action_norm",
    "last_action_x",
    "last_action_y",
    "virtual_target_x_px",
    "virtual_target_y_px",
    "desired_target_x_px",
    "desired_target_y_px",
    "planned_anchor_x_cm",
    "planned_anchor_y_cm",
    "planned_target_x_cm",
    "planned_target_y_cm",
    "planned_delta_dx_cm",
    "planned_delta_dy_cm",
    "planned_delta_norm_cm",
    "actual_robot_move_dx_cm",
    "actual_robot_move_dy_cm",
    "actual_robot_move_norm_cm",
    "actual_to_planned_step_projection",
    "planned_endpoint_actual_lag_cm",
    "planned_endpoint_actual_lag_excess_cm",
    "servo_follower_lag_allowance_cm",
    "servo_lag_warning_cm",
    "servo_lag_hard_limit_cm",
    "servo_publish_accepted",
    "servo_publish_reject_reason",
    "target_clipped",
    "target_step_limited",
    "robot_world_x",
    "robot_world_y",
    "robot_world_z",
    "safety_stop",
    "safety_reason",
    "task_finished",
    "done_reason",
]


NUMERIC_ARRAY_FIELDS = [
    "t_task_s",
    "control_dt_s",
    "control_loop_hz_inst",
    "policy_dt_s",
    "policy_loop_hz_inst",
    "policy_deadline_overrun_s",
    "servo_dt_s",
    "servo_loop_hz_inst",
    "servo_target_age_s",
    "servo_interpolation_phase",
    "servo_command_sequence",
    "servo_policy_step",
    "servo_commanded_world_x",
    "servo_commanded_world_y",
    "servo_commanded_world_z",
    "servo_target_world_x",
    "servo_target_world_y",
    "servo_target_world_z",
    "servo_segment_start_world_x",
    "servo_segment_start_world_y",
    "servo_segment_start_world_z",
    "servo_segment_end_world_x",
    "servo_segment_end_world_y",
    "servo_segment_end_world_z",
    "servo_segment_start_perf",
    "servo_segment_end_perf",
    "servo_segment_elapsed_s",
    "servo_segment_duration_s",
    "servo_pending_segment_count",
    "servo_command_step_limited_count",
    "servo_publish_rejected_count",
    "servo_tracking_error_cm",
    "servo_target_error_cm",
    "servo_deadline_overrun_s",
    "servo_deadline_overrun_count",
    "vision_age_s",
    "camera_dt_s",
    "camera_hz_inst",
    "dead_reckoning_age_s",
    "hand_x_cm",
    "hand_y_cm",
    "robot_x_cm",
    "robot_y_cm",
    "microrobot_x_cm",
    "microrobot_y_cm",
    "distance_cm",
    "virtual_hand_stride_cm",
    "virtual_hand_smoothing_alpha",
    "virtual_hand_delay_frames",
    "virtual_hand_action_x",
    "virtual_hand_action_y",
    "virtual_hand_action_norm",
    "virtual_hand_command_dx_cm",
    "virtual_hand_command_dy_cm",
    "virtual_hand_command_norm_cm",
    "virtual_hand_delayed_dx_cm",
    "virtual_hand_delayed_dy_cm",
    "virtual_hand_delayed_norm_cm",
    "virtual_hand_smoothed_dx_cm",
    "virtual_hand_smoothed_dy_cm",
    "virtual_hand_smoothed_norm_cm",
    "virtual_hand_exec_dx_cm",
    "virtual_hand_exec_dy_cm",
    "virtual_hand_exec_norm_cm",
    "virtual_hand_actual_dx_cm",
    "virtual_hand_actual_dy_cm",
    "virtual_hand_actual_norm_cm",
    "virtual_hand_policy_inference_ms",
    "policy_inference_ms",
    "action_x",
    "action_y",
    "action_norm",
    "virtual_target_x_px",
    "virtual_target_y_px",
    "desired_target_x_px",
    "desired_target_y_px",
    "planned_anchor_x_cm",
    "planned_anchor_y_cm",
    "planned_target_x_cm",
    "planned_target_y_cm",
    "planned_delta_dx_cm",
    "planned_delta_dy_cm",
    "planned_delta_norm_cm",
    "actual_robot_move_dx_cm",
    "actual_robot_move_dy_cm",
    "actual_robot_move_norm_cm",
    "actual_to_planned_step_projection",
    "planned_endpoint_actual_lag_cm",
    "planned_endpoint_actual_lag_excess_cm",
    "servo_follower_lag_allowance_cm",
    "servo_lag_warning_cm",
    "servo_lag_hard_limit_cm",
]


BOOLEAN_ARRAY_FIELDS = [
    "policy_deadline_overrun",
    "servo_watchdog_stopped",
    "servo_command_step_limited",
    "servo_publish_accepted",
    "vision_frame_available",
    "hand_detected",
    "microrobot_detected",
    "dead_reckoning_used",
    "in_zpd",
    "virtual_hand_accel_clipped",
    "virtual_hand_workspace_clipped",
    "target_clipped",
    "target_step_limited",
    "safety_stop",
    "task_finished",
]


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "run"


def _clean_value(value):
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return "" if not np.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _as_float(value):
    if value is None or value == "":
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _nanmean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else None


def _nanmedian(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else None


def _nanstd(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=0)) if arr.size else None


def _nanpercentile(values, q):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else None


def _nanmax(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else None


class DeploymentRolloutLogger:
    def __init__(
        self,
        out_root,
        subject="pilot",
        condition="zero_shot_physical",
        metadata=None,
        zpd_low_cm=4.0,
        zpd_high_cm=6.0,
        save_video=False,
    ):
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.subject = _slug(subject)
        self.condition = _slug(condition)
        self.zpd_low_cm = float(zpd_low_cm)
        self.zpd_high_cm = float(zpd_high_cm)
        self.save_video = bool(save_video)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rollout_id = f"{timestamp}_{self.subject}_{self.condition}"
        self.rollout_dir = self.out_root / self.rollout_id
        suffix = 1
        while self.rollout_dir.exists():
            self.rollout_dir = self.out_root / f"{self.rollout_id}_{suffix}"
            suffix += 1
        self.rollout_dir.mkdir(parents=True)
        self.rollout_id = self.rollout_dir.name

        self.metadata = dict(metadata or {})
        self.metadata.update({
            "rollout_id": self.rollout_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "subject": self.subject,
            "condition": self.condition,
            "zpd_low_cm": self.zpd_low_cm,
            "zpd_high_cm": self.zpd_high_cm,
        })
        self.rows = []
        self.events_path = self.rollout_dir / "events.jsonl"
        self.snapshot_path = self.rollout_dir / "snapshot_annotated.png"
        self._snapshot_saved = False
        self._last_snapshot_frame = None
        self._last_snapshot_step = None
        self._video_writer = None
        self._video_path = self.rollout_dir / "annotated_video.mp4"
        self._video_queue = None
        self._video_thread = None
        self._video_dropped_frames = 0
        self._video_written_frames = 0

        with (self.rollout_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        self.record_event("rollout_created")

    def record_event(self, event, **payload):
        data = {
            "t_wall_s": time.time(),
            "event": event,
        }
        data.update({k: _clean_value(v) for k, v in payload.items()})
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def record_step(self, row):
        cleaned = {name: _clean_value(row.get(name, "")) for name in FIELDNAMES}
        self.rows.append(cleaned)

    def maybe_save_snapshot(self, frame, step, t_task_s, preferred_step=None, preferred_time_s=None):
        if frame is None or self._snapshot_saved:
            return
        self._last_snapshot_frame = frame.copy()
        self._last_snapshot_step = int(step)

        should_save = False
        if preferred_step is not None and int(step) >= int(preferred_step):
            should_save = True
        elif preferred_time_s is not None and float(t_task_s) >= float(preferred_time_s):
            should_save = True

        if should_save:
            self._write_snapshot(frame, step, t_task_s)

    def _write_snapshot(self, frame, step=None, t_task_s=None):
        import cv2

        cv2.imwrite(str(self.snapshot_path), frame)
        self._snapshot_saved = True
        self.record_event("snapshot_saved", step=step, t_task_s=t_task_s, path=str(self.snapshot_path))

    def ensure_video_writer(self, frame, fps=20.0):
        """Open the encoder and start its worker thread before the loop runs.

        Creating the mp4v writer costs tens to hundreds of milliseconds, so the
        control loop must not be the thing that pays for it on its first frame.
        """
        if not self.save_video or frame is None or self._video_writer is not None:
            return
        import cv2

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self._video_path), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer at {self._video_path}")
        self._video_writer = writer
        # A short queue: enough to absorb an encoder hiccup, short enough that a
        # stalled encoder drops frames instead of growing memory without bound.
        self._video_queue = queue.Queue(maxsize=8)
        self._video_thread = threading.Thread(
            target=self._video_worker,
            daemon=True,
            name="video-encoder",
        )
        self._video_thread.start()

    def _video_worker(self):
        while True:
            frame = self._video_queue.get()
            if frame is None:
                return
            try:
                self._video_writer.write(frame)
                self._video_written_frames += 1
            except Exception:
                return

    def write_frame(self, frame, fps=20.0):
        """Hand one annotated frame to the encoder thread without blocking.

        The frame is not copied; the caller builds a fresh annotated frame each
        step and must not mutate this one after handing it over.
        """
        if not self.save_video or frame is None:
            return
        self.ensure_video_writer(frame, fps=fps)
        if self._video_queue is None:
            return
        try:
            self._video_queue.put_nowait(frame)
        except queue.Full:
            self._video_dropped_frames += 1

    def close(self, done_reason="unknown"):
        if self._video_writer is not None:
            if self._video_queue is not None:
                self._video_queue.put(None)
            if self._video_thread is not None:
                self._video_thread.join(timeout=10.0)
            self._video_writer.release()
            self._video_writer = None
            self._video_queue = None
            self._video_thread = None
            self.record_event(
                "video_saved",
                path=str(self._video_path),
                frames_written=int(self._video_written_frames),
                frames_dropped=int(self._video_dropped_frames),
            )

        if not self._snapshot_saved and self._last_snapshot_frame is not None:
            self._write_snapshot(self._last_snapshot_frame, step=self._last_snapshot_step)

        for row in self.rows:
            if not row.get("done_reason"):
                row["done_reason"] = done_reason

        self._write_timeseries_csv()
        self._write_timeseries_npz()
        summary = self._compute_summary(done_reason)
        with (self.rollout_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.record_event("rollout_closed", done_reason=done_reason, summary_path=str(self.rollout_dir / "summary.json"))
        return summary

    def _write_timeseries_csv(self):
        path = self.rollout_dir / "timeseries.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)

    def _write_timeseries_npz(self):
        arrays = {}
        for field in NUMERIC_ARRAY_FIELDS:
            arrays[field] = np.asarray([_as_float(row.get(field)) for row in self.rows], dtype=np.float32)
        for field in BOOLEAN_ARRAY_FIELDS:
            arrays[field] = np.asarray([bool(row.get(field)) for row in self.rows], dtype=np.bool_)
        arrays["step"] = np.asarray([int(row.get("step") or 0) for row in self.rows], dtype=np.int32)
        np.savez(self.rollout_dir / "timeseries.npz", **arrays)

    def _compute_summary(self, done_reason):
        duration_values = [_as_float(row.get("t_task_s")) for row in self.rows]
        duration_values = [v for v in duration_values if np.isfinite(v)]
        distances = [_as_float(row.get("distance_cm")) for row in self.rows]
        control_hz = [_as_float(row.get("control_loop_hz_inst")) for row in self.rows]
        policy_hz = [
            _as_float(
                row.get("policy_loop_hz_inst")
                if row.get("policy_loop_hz_inst") not in (None, "")
                else row.get("control_loop_hz_inst")
            )
            for row in self.rows
        ]
        servo_hz = [_as_float(row.get("servo_loop_hz_inst")) for row in self.rows]
        policy_overrun_s = [
            _as_float(row.get("policy_deadline_overrun_s")) for row in self.rows
        ]
        servo_target_age_s = [
            _as_float(row.get("servo_target_age_s")) for row in self.rows
        ]
        servo_tracking_error_cm = [
            _as_float(row.get("servo_tracking_error_cm")) for row in self.rows
        ]
        servo_target_error_cm = [
            _as_float(row.get("servo_target_error_cm")) for row in self.rows
        ]
        servo_overrun_s = [
            _as_float(row.get("servo_deadline_overrun_s")) for row in self.rows
        ]
        servo_overrun_count = [
            _as_float(row.get("servo_deadline_overrun_count")) for row in self.rows
        ]
        servo_command_step_limited_count_values = [
            _as_float(row.get("servo_command_step_limited_count")) for row in self.rows
        ]
        servo_publish_rejected_count_values = [
            _as_float(row.get("servo_publish_rejected_count")) for row in self.rows
        ]
        planned_endpoint_actual_lag_cm = [
            _as_float(row.get("planned_endpoint_actual_lag_cm")) for row in self.rows
        ]
        planned_endpoint_actual_lag_excess_cm = [
            _as_float(row.get("planned_endpoint_actual_lag_excess_cm"))
            for row in self.rows
        ]
        servo_follower_lag_allowance_cm = [
            _as_float(row.get("servo_follower_lag_allowance_cm")) for row in self.rows
        ]
        actual_to_planned_step_projection = [
            _as_float(row.get("actual_to_planned_step_projection")) for row in self.rows
        ]
        policy_overrun = np.asarray([
            bool(row.get("policy_deadline_overrun")) for row in self.rows
        ], dtype=bool)
        servo_watchdog = np.asarray([
            bool(row.get("servo_watchdog_stopped")) for row in self.rows
        ], dtype=bool)
        servo_command_step_limited = np.asarray([
            bool(row.get("servo_command_step_limited")) for row in self.rows
        ], dtype=bool)
        camera_hz = [_as_float(row.get("camera_hz_inst")) for row in self.rows]
        inference_ms = [_as_float(row.get("policy_inference_ms")) for row in self.rows]
        dead_reckoning = np.asarray([bool(row.get("dead_reckoning_used")) for row in self.rows], dtype=bool)
        in_zpd = np.asarray([bool(row.get("in_zpd")) for row in self.rows], dtype=bool)
        safety_stop = np.asarray([bool(row.get("safety_stop")) for row in self.rows], dtype=bool)
        virtual_hand_actual_norm = np.asarray([
            _as_float(row.get("virtual_hand_actual_norm_cm")) for row in self.rows
        ], dtype=float)
        virtual_hand_stride = np.asarray([
            _as_float(row.get("virtual_hand_stride_cm")) for row in self.rows
        ], dtype=float)
        virtual_hand_policy_ms = [
            _as_float(row.get("virtual_hand_policy_inference_ms")) for row in self.rows
        ]
        virtual_hand_rows = np.isfinite(virtual_hand_actual_norm)
        valid_ratio = (
            virtual_hand_rows
            & np.isfinite(virtual_hand_stride)
            & (virtual_hand_stride > 0.0)
        )
        virtual_hand_stride_ratio = (
            virtual_hand_actual_norm[valid_ratio]
            / virtual_hand_stride[valid_ratio]
        )
        virtual_hand_accel_clipped = np.asarray([
            bool(row.get("virtual_hand_accel_clipped")) for row in self.rows
        ], dtype=bool)
        virtual_hand_workspace_clipped = np.asarray([
            bool(row.get("virtual_hand_workspace_clipped")) for row in self.rows
        ], dtype=bool)
        duration_target_s = _as_float(self.metadata.get("duration_target_s"))
        fixed_horizon_metrics = None
        if np.isfinite(duration_target_s) and duration_target_s > 0.0:
            fixed_horizon_metrics = compute_fixed_horizon_tiz(
                self.rows,
                duration_target_s,
                done_reason,
            )
        observed_occupancy = float(np.mean(in_zpd)) if in_zpd.size else None
        servo_command_step_limited_count = _nanmax(
            servo_command_step_limited_count_values
        )
        if servo_command_step_limited_count is None:
            servo_command_step_limited_count = (
                int(np.sum(servo_command_step_limited))
                if servo_command_step_limited.size
                else None
            )
        else:
            servo_command_step_limited_count = int(servo_command_step_limited_count)
        servo_publish_rejected_count = _nanmax(
            servo_publish_rejected_count_values
        )
        if servo_publish_rejected_count is None:
            servo_publish_rejected_count = 0 if self.rows else None
        else:
            servo_publish_rejected_count = int(servo_publish_rejected_count)

        return {
            "rollout_id": self.rollout_id,
            "subject": self.subject,
            "condition": self.condition,
            "duration_s": float(max(duration_values)) if duration_values else 0.0,
            "duration_target_s": duration_target_s if np.isfinite(duration_target_s) else None,
            "num_control_steps": len(self.rows),
            "camera_update_rate_hz_mean": _nanmean(camera_hz),
            "camera_update_rate_hz_median": _nanmedian(camera_hz),
            "control_loop_rate_hz_mean": _nanmean(control_hz),
            "control_loop_rate_hz_median": _nanmedian(control_hz),
            "policy_loop_rate_hz_mean": _nanmean(policy_hz),
            "policy_loop_rate_hz_median": _nanmedian(policy_hz),
            "policy_deadline_overrun_s_mean": _nanmean(policy_overrun_s),
            "policy_deadline_overrun_s_p95": _nanpercentile(
                policy_overrun_s,
                95,
            ),
            "policy_deadline_overrun_count": int(np.sum(policy_overrun)),
            "policy_deadline_overrun_fraction": (
                float(np.mean(policy_overrun)) if policy_overrun.size else None
            ),
            "servo_loop_rate_hz_mean": _nanmean(servo_hz),
            "servo_loop_rate_hz_median": _nanmedian(servo_hz),
            "servo_target_age_s_mean": _nanmean(servo_target_age_s),
            "servo_target_age_s_p95": _nanpercentile(servo_target_age_s, 95),
            "servo_tracking_error_cm_mean": _nanmean(servo_tracking_error_cm),
            "servo_tracking_error_cm_p95": _nanpercentile(
                servo_tracking_error_cm,
                95,
            ),
            "servo_target_error_cm_mean": _nanmean(servo_target_error_cm),
            "servo_target_error_cm_p95": _nanpercentile(
                servo_target_error_cm,
                95,
            ),
            "servo_deadline_overrun_s_mean": _nanmean(servo_overrun_s),
            "servo_deadline_overrun_s_p95": _nanpercentile(servo_overrun_s, 95),
            "servo_deadline_overrun_count": _nanmax(servo_overrun_count),
            "servo_command_step_limited_count": servo_command_step_limited_count,
            "servo_publish_rejected_count": servo_publish_rejected_count,
            "planned_endpoint_actual_lag_cm_mean": _nanmean(
                planned_endpoint_actual_lag_cm
            ),
            "planned_endpoint_actual_lag_cm_p95": _nanpercentile(
                planned_endpoint_actual_lag_cm,
                95,
            ),
            "planned_endpoint_actual_lag_cm_max": _nanmax(
                planned_endpoint_actual_lag_cm
            ),
            "planned_endpoint_actual_lag_excess_cm_mean": _nanmean(
                planned_endpoint_actual_lag_excess_cm
            ),
            "planned_endpoint_actual_lag_excess_cm_p95": _nanpercentile(
                planned_endpoint_actual_lag_excess_cm,
                95,
            ),
            "planned_endpoint_actual_lag_excess_cm_max": _nanmax(
                planned_endpoint_actual_lag_excess_cm
            ),
            "servo_follower_lag_allowance_cm": _nanmax(
                servo_follower_lag_allowance_cm
            ),
            "actual_to_planned_step_projection_mean": _nanmean(
                actual_to_planned_step_projection
            ),
            "actual_to_planned_step_projection_p50": _nanmedian(
                actual_to_planned_step_projection
            ),
            "servo_watchdog_stop_count": int(np.sum(servo_watchdog)),
            "policy_inference_latency_ms_mean": _nanmean(inference_ms),
            "policy_inference_latency_ms_p95": _nanpercentile(inference_ms, 95),
            "virtual_hand_policy_latency_ms_mean": _nanmean(virtual_hand_policy_ms),
            "virtual_hand_policy_latency_ms_p95": _nanpercentile(virtual_hand_policy_ms, 95),
            "virtual_hand_actual_move_cm_mean": _nanmean(virtual_hand_actual_norm),
            "virtual_hand_actual_move_cm_p95": _nanpercentile(virtual_hand_actual_norm, 95),
            "virtual_hand_actual_to_stride_ratio_mean": _nanmean(virtual_hand_stride_ratio),
            "virtual_hand_accel_clipped_fraction": (
                float(np.mean(virtual_hand_accel_clipped[virtual_hand_rows]))
                if np.any(virtual_hand_rows)
                else None
            ),
            "virtual_hand_workspace_clipped_fraction": (
                float(np.mean(virtual_hand_workspace_clipped[virtual_hand_rows]))
                if np.any(virtual_hand_rows)
                else None
            ),
            "dead_reckoning_fraction": float(np.mean(dead_reckoning)) if dead_reckoning.size else None,
            "dead_reckoning_steps": int(np.sum(dead_reckoning)),
            "safety_stop_count": int(np.sum(safety_stop)),
            "distance_cm_mean": _nanmean(distances),
            "distance_cm_median": _nanmedian(distances),
            "distance_cm_std": _nanstd(distances),
            "zpd_time_s": (
                fixed_horizon_metrics["zpd_time_s"]
                if fixed_horizon_metrics is not None
                else None
            ),
            "tiz_fixed_horizon_fraction": (
                fixed_horizon_metrics["tiz_fixed_horizon_fraction"]
                if fixed_horizon_metrics is not None
                else None
            ),
            "zpd_observed_occupancy_fraction": observed_occupancy,
            "zpd_occupancy_fraction": observed_occupancy,
            "zpd_occupancy_steps": int(np.sum(in_zpd)),
            "zpd_low_cm": self.zpd_low_cm,
            "zpd_high_cm": self.zpd_high_cm,
            "done_reason": done_reason,
            "snapshot_path": str(self.snapshot_path) if self.snapshot_path.exists() else None,
            "video_path": str(self._video_path) if self._video_path.exists() else None,
            "video_frames_written": int(self._video_written_frames),
            "video_frames_dropped": int(self._video_dropped_frames),
        }
