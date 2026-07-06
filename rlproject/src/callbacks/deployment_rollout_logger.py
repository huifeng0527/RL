import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np


FIELDNAMES = [
    "step",
    "t_wall_s",
    "t_task_s",
    "control_dt_s",
    "control_loop_hz_inst",
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
    "policy_inference_ms",
    "action_x",
    "action_y",
    "action_norm",
    "virtual_target_x_px",
    "virtual_target_y_px",
    "desired_target_x_px",
    "desired_target_y_px",
]


BOOLEAN_ARRAY_FIELDS = [
    "vision_frame_available",
    "hand_detected",
    "microrobot_detected",
    "dead_reckoning_used",
    "in_zpd",
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
        if frame is None:
            return
        self._last_snapshot_frame = frame.copy()
        self._last_snapshot_step = int(step)
        if self._snapshot_saved:
            return

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

    def write_frame(self, frame, fps=20.0):
        if not self.save_video or frame is None:
            return
        import cv2

        if self._video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(str(self._video_path), fourcc, float(fps), (w, h))
        self._video_writer.write(frame)

    def close(self, done_reason="unknown"):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            self.record_event("video_saved", path=str(self._video_path))

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
        camera_hz = [_as_float(row.get("camera_hz_inst")) for row in self.rows]
        inference_ms = [_as_float(row.get("policy_inference_ms")) for row in self.rows]
        dead_reckoning = np.asarray([bool(row.get("dead_reckoning_used")) for row in self.rows], dtype=bool)
        in_zpd = np.asarray([bool(row.get("in_zpd")) for row in self.rows], dtype=bool)
        safety_stop = np.asarray([bool(row.get("safety_stop")) for row in self.rows], dtype=bool)

        return {
            "rollout_id": self.rollout_id,
            "subject": self.subject,
            "condition": self.condition,
            "duration_s": float(max(duration_values)) if duration_values else 0.0,
            "num_control_steps": len(self.rows),
            "camera_update_rate_hz_mean": _nanmean(camera_hz),
            "camera_update_rate_hz_median": _nanmedian(camera_hz),
            "control_loop_rate_hz_mean": _nanmean(control_hz),
            "control_loop_rate_hz_median": _nanmedian(control_hz),
            "policy_inference_latency_ms_mean": _nanmean(inference_ms),
            "policy_inference_latency_ms_p95": _nanpercentile(inference_ms, 95),
            "dead_reckoning_fraction": float(np.mean(dead_reckoning)) if dead_reckoning.size else None,
            "dead_reckoning_steps": int(np.sum(dead_reckoning)),
            "safety_stop_count": int(np.sum(safety_stop)),
            "distance_cm_mean": _nanmean(distances),
            "distance_cm_median": _nanmedian(distances),
            "distance_cm_std": _nanstd(distances),
            "zpd_occupancy_fraction": float(np.mean(in_zpd)) if in_zpd.size else None,
            "zpd_occupancy_steps": int(np.sum(in_zpd)),
            "zpd_low_cm": self.zpd_low_cm,
            "zpd_high_cm": self.zpd_high_cm,
            "done_reason": done_reason,
            "snapshot_path": str(self.snapshot_path) if self.snapshot_path.exists() else None,
            "video_path": str(self._video_path) if self._video_path.exists() else None,
        }
