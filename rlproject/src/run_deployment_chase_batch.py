from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_SCRIPT = Path(__file__).resolve().with_name("record_deployment_chase.py")
DEFAULT_BATCH_ROOT = REPO_ROOT / "data" / "deployment_batches"
BOUNDARY_BAND_CM = 0.5
VALID_DONE_REASONS = {"caught", "timeout"}

RUN_FIELDS = [
    "run_id", "pair_id", "seed", "order_in_pair", "controller", "status",
    "rollout_dir", "done_reason", "caught", "duration_s", "num_control_steps",
    "zpd_occupancy_fraction", "distance_cm_mean", "distance_cm_std",
    "distance_cm_min", "distance_cm_final", "too_close_fraction",
    "too_far_fraction", "min_boundary_clearance_cm", "boundary_occupancy_fraction",
    "target_clipped_fraction", "target_step_limited_fraction",
    "control_loop_rate_hz_mean", "camera_update_rate_hz_mean",
    "policy_inference_latency_ms_mean", "policy_inference_latency_ms_p95",
    "safety_stop_count", "microrobot_detection_fraction",
    "microrobot_tcp_error_cm_mean", "microrobot_tcp_error_cm_p95",
]

SUMMARY_METRICS = [
    "zpd_occupancy_fraction", "duration_s", "distance_cm_mean",
    "too_close_fraction", "too_far_fraction", "min_boundary_clearance_cm",
    "boundary_occupancy_fraction", "target_clipped_fraction",
    "target_step_limited_fraction", "control_loop_rate_hz_mean",
    "camera_update_rate_hz_mean", "policy_inference_latency_ms_mean",
    "microrobot_detection_fraction", "microrobot_tcp_error_cm_mean",
]

PAIRED_METRICS = [
    "zpd_occupancy_fraction", "duration_s", "distance_cm_mean",
    "too_close_fraction", "too_far_fraction", "min_boundary_clearance_cm",
    "policy_inference_latency_ms_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential paired League-vs-MPC physical tests with a Virtual Hand."
    )
    parser.add_argument("--subject-prefix", default="virtual_batch")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--stride", type=float, default=0.35)
    parser.add_argument("--hand-stride", type=float, default=0.45)
    parser.add_argument("--max-step", type=float, default=0.60)
    parser.add_argument("--catch-distance", type=float, default=1.5)
    parser.add_argument("--inter-run-delay", type=float, default=3.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Connect to hardware. Without this flag the script only prints the plan.",
    )
    return parser.parse_args()


def slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "batch"


def parse_seeds(raw):
    seeds = []
    for item in raw.split(","):
        if not item.strip():
            continue
        seed = int(item)
        if seed < 0:
            raise ValueError("Seeds must be nonnegative")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def to_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def mean_std(values):
    arr = np.asarray([v for v in (to_float(x) for x in values) if v is not None])
    if not arr.size:
        return None, None
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def build_manifest(args, batch_dir):
    rollout_root = batch_dir / "rollouts"
    runs = []
    for pair_index, seed in enumerate(parse_seeds(args.seeds)):
        controllers = ["league", "cv_mpc"] if pair_index % 2 == 0 else ["cv_mpc", "league"]
        pair_id = f"seed{seed:03d}"
        subject = f"{slug(args.subject_prefix)}_{pair_id}"
        for order, controller in enumerate(controllers, start=1):
            condition = f"{controller}_virtual_h1_{pair_id}"
            command = [
                sys.executable, str(ROLLOUT_SCRIPT),
                "--controller", controller,
                "--hand-source", "virtual",
                "--seed", str(seed),
                "--seconds", str(args.seconds),
                "--stride", str(args.stride),
                "--hand-stride", str(args.hand_stride),
                "--max-step", str(args.max_step),
                "--catch-distance", str(args.catch_distance),
                "--subject", subject,
                "--condition", condition,
                "--out-dir", str(rollout_root),
            ]
            if args.save_video:
                command.append("--save-video")
            if args.no_display:
                command.append("--no-display")
            runs.append({
                "run_id": f"{pair_id}_{controller}",
                "pair_id": pair_id,
                "pair_index": pair_index,
                "seed": seed,
                "order_in_pair": order,
                "controller": controller,
                "subject": subject,
                "condition": condition,
                "command": command,
            })
    return {
        "schema_version": 1,
        "batch_id": batch_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch_dir": str(batch_dir),
        "rollout_root": str(rollout_root),
        "termination": "first_catch_or_timeout",
        "parameters": {
            "seconds": args.seconds,
            "stride_cm": args.stride,
            "hand_stride_cm": args.hand_stride,
            "max_step_cm": args.max_step,
            "catch_distance_cm": args.catch_distance,
            "inter_run_delay_s": args.inter_run_delay,
            "save_video": args.save_video,
            "no_display": args.no_display,
        },
        "runs": runs,
    }


def prepare_batch(args):
    if args.resume:
        batch_dir = args.resume.resolve()
        manifest = read_json(batch_dir / "manifest.json")
        return batch_dir, manifest

    for name in ("seconds", "stride", "hand_stride", "max_step", "catch_distance"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.inter_run_delay < 0:
        raise ValueError("--inter-run-delay cannot be negative")
    if not ROLLOUT_SCRIPT.exists():
        raise FileNotFoundError(ROLLOUT_SCRIPT)

    batch_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_{slug(args.subject_prefix)}_virtual_pair"
    )
    batch_dir = args.batch_root.resolve() / batch_id
    (batch_dir / "rollouts").mkdir(parents=True)
    manifest = build_manifest(args, batch_dir)
    write_json(batch_dir / "manifest.json", manifest)
    return batch_dir, manifest


def metadata_matches(metadata, run):
    return (
        metadata.get("subject") == run["subject"]
        and metadata.get("condition") == run["condition"]
        and metadata.get("controller") == run["controller"]
        and metadata.get("hand_source") == "virtual"
        and int(metadata.get("virtual_hand_seed", -1)) == int(run["seed"])
    )


def find_rollout(run, rollout_root, names=None):
    matches = []
    if not rollout_root.exists():
        return None
    for path in rollout_root.iterdir():
        if not path.is_dir() or (names is not None and path.name not in names):
            continue
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            if metadata_matches(read_json(metadata_path), run):
                matches.append(path)
        except (OSError, ValueError, TypeError):
            pass
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def extract_metrics(run, rollout_dir):
    metadata = read_json(rollout_dir / "metadata.json")
    summary = read_json(rollout_dir / "summary.json")
    rows = read_csv(rollout_dir / "timeseries.csv")
    if not rows:
        raise ValueError(f"No timeseries rows in {rollout_dir}")

    distances = np.asarray([
        value for value in (to_float(row.get("distance_cm")) for row in rows)
        if value is not None
    ])
    zpd_low = float(metadata.get("zpd_low_cm", 3.5))
    zpd_high = float(metadata.get("zpd_high_cm", 5.5))

    robot_xy = np.asarray([
        [to_float(row.get("robot_x_cm")), to_float(row.get("robot_y_cm"))]
        for row in rows
    ], dtype=object)
    valid_robot = np.asarray([
        x is not None and y is not None for x, y in robot_xy
    ])
    clearances = np.asarray([])
    if np.any(valid_robot):
        xy = robot_xy[valid_robot].astype(float)
        width = float(metadata.get("workspace_width_cm", 15.0))
        height = float(metadata.get("workspace_height_cm", 10.0))
        margin = float(metadata.get("workspace_margin_cm", 0.3))
        clearances = np.minimum.reduce([
            xy[:, 0] - margin, width - margin - xy[:, 0],
            xy[:, 1] - margin, height - margin - xy[:, 1],
        ])

    tracking_errors = []
    detected = []
    for row in rows:
        detected.append(to_bool(row.get("microrobot_detected")))
        rx, ry = to_float(row.get("robot_x_cm")), to_float(row.get("robot_y_cm"))
        mx, my = to_float(row.get("microrobot_x_cm")), to_float(row.get("microrobot_y_cm"))
        if None not in (rx, ry, mx, my):
            tracking_errors.append(float(np.hypot(mx - rx, my - ry)))

    done_reason = str(summary.get("done_reason", "unknown"))
    values = {
        "run_id": run["run_id"],
        "pair_id": run["pair_id"],
        "seed": run["seed"],
        "order_in_pair": run["order_in_pair"],
        "controller": run["controller"],
        "status": "completed",
        "rollout_dir": str(rollout_dir),
        "done_reason": done_reason,
        "caught": done_reason == "caught",
        "duration_s": to_float(summary.get("duration_s")),
        "num_control_steps": int(summary.get("num_control_steps", len(rows))),
        "zpd_occupancy_fraction": to_float(summary.get("zpd_occupancy_fraction")),
        "distance_cm_mean": float(np.mean(distances)) if distances.size else None,
        "distance_cm_std": float(np.std(distances)) if distances.size else None,
        "distance_cm_min": float(np.min(distances)) if distances.size else None,
        "distance_cm_final": float(distances[-1]) if distances.size else None,
        "too_close_fraction": float(np.mean(distances < zpd_low)) if distances.size else None,
        "too_far_fraction": float(np.mean(distances > zpd_high)) if distances.size else None,
        "min_boundary_clearance_cm": float(np.min(clearances)) if clearances.size else None,
        "boundary_occupancy_fraction": float(np.mean(clearances < BOUNDARY_BAND_CM)) if clearances.size else None,
        "target_clipped_fraction": float(np.mean([to_bool(r.get("target_clipped")) for r in rows])),
        "target_step_limited_fraction": float(np.mean([to_bool(r.get("target_step_limited")) for r in rows])),
        "control_loop_rate_hz_mean": to_float(summary.get("control_loop_rate_hz_mean")),
        "camera_update_rate_hz_mean": to_float(summary.get("camera_update_rate_hz_mean")),
        "policy_inference_latency_ms_mean": to_float(summary.get("policy_inference_latency_ms_mean")),
        "policy_inference_latency_ms_p95": to_float(summary.get("policy_inference_latency_ms_p95")),
        "safety_stop_count": int(summary.get("safety_stop_count", 0)),
        "microrobot_detection_fraction": float(np.mean(detected)) if detected else None,
        "microrobot_tcp_error_cm_mean": float(np.mean(tracking_errors)) if tracking_errors else None,
        "microrobot_tcp_error_cm_p95": float(np.percentile(tracking_errors, 95)) if tracking_errors else None,
    }
    if done_reason not in VALID_DONE_REASONS:
        values["status"] = "failed"
    if values["safety_stop_count"] > 0 or done_reason.startswith("safety_"):
        values["status"] = "safety_stop"
    return values


def incomplete_result(run, status, rollout_dir=None, done_reason=None):
    row = {field: None for field in RUN_FIELDS}
    row.update({
        "run_id": run["run_id"],
        "pair_id": run["pair_id"],
        "seed": run["seed"],
        "order_in_pair": run["order_in_pair"],
        "controller": run["controller"],
        "status": status,
        "rollout_dir": str(rollout_dir) if rollout_dir else None,
        "done_reason": done_reason,
        "caught": False,
        "safety_stop_count": 0,
    })
    return row


def recover_results(manifest, existing_rows):
    by_id = {row["run_id"]: row for row in existing_rows}
    rollout_root = Path(manifest["rollout_root"])
    for run in manifest["runs"]:
        row = by_id.get(run["run_id"])
        if row and row.get("status") == "completed" and Path(row.get("rollout_dir", "")).exists():
            continue
        rollout = find_rollout(run, rollout_root)
        if rollout and (rollout / "summary.json").exists():
            try:
                recovered = extract_metrics(run, rollout)
            except (OSError, ValueError, TypeError):
                continue
            if recovered["status"] == "completed":
                by_id[run["run_id"]] = recovered
    return [by_id[key] for key in sorted(by_id)]


def write_aggregates(batch_dir, manifest, rows):
    rows = sorted(rows, key=lambda row: manifest_run_index(manifest, row["run_id"]))
    write_csv(batch_dir / "runs.csv", rows, RUN_FIELDS)

    summaries = []
    for controller in ("league", "cv_mpc"):
        group = [r for r in rows if r["controller"] == controller and r["status"] == "completed"]
        summary = {
            "controller": controller,
            "n_completed": len(group),
            "catch_rate": float(np.mean([to_bool(r["caught"]) for r in group])) if group else None,
            "safety_stop_count": int(sum(int(float(r.get("safety_stop_count") or 0)) for r in rows if r["controller"] == controller)),
            "done_reason_counts": json.dumps(dict(Counter(r["done_reason"] for r in group)), sort_keys=True),
        }
        for metric in SUMMARY_METRICS:
            summary[f"{metric}_mean"], summary[f"{metric}_std"] = mean_std(r.get(metric) for r in group)
        summaries.append(summary)
    write_csv(batch_dir / "condition_summary.csv", summaries, list(summaries[0]))

    pairs = []
    completed = {(r["pair_id"], r["controller"]): r for r in rows if r["status"] == "completed"}
    for pair_id in sorted({run["pair_id"] for run in manifest["runs"]}):
        league = completed.get((pair_id, "league"))
        mpc = completed.get((pair_id, "cv_mpc"))
        if not league or not mpc:
            continue
        pair = {
            "pair_id": pair_id,
            "seed": league["seed"],
            "league_order_in_pair": league["order_in_pair"],
            "mpc_order_in_pair": mpc["order_in_pair"],
            "league_done_reason": league["done_reason"],
            "mpc_done_reason": mpc["done_reason"],
        }
        for metric in PAIRED_METRICS:
            lv, mv = to_float(league.get(metric)), to_float(mpc.get(metric))
            pair[f"league_{metric}"] = lv
            pair[f"mpc_{metric}"] = mv
            pair[f"delta_{metric}"] = lv - mv if lv is not None and mv is not None else None
        pairs.append(pair)
    pair_fields = list(pairs[0]) if pairs else [
        "pair_id", "seed", "league_order_in_pair", "mpc_order_in_pair",
        "league_done_reason", "mpc_done_reason",
        *[f"{prefix}_{metric}" for metric in PAIRED_METRICS for prefix in ("league", "mpc", "delta")],
    ]
    write_csv(batch_dir / "paired_results.csv", pairs, pair_fields)


def manifest_run_index(manifest, run_id):
    return next(i for i, run in enumerate(manifest["runs"]) if run["run_id"] == run_id)


def print_plan(manifest, completed_ids):
    print(f"[batch] {manifest['batch_id']}")
    print(f"[batch] output: {manifest['batch_dir']}")
    print("[batch] stop rule: first catch or timeout")
    for i, run in enumerate(manifest["runs"], start=1):
        state = "skip" if run["run_id"] in completed_ids else "run"
        print(
            f"[{state}] {i:02d} pair={run['pair_id']} order={run['order_in_pair']} "
            f"controller={run['controller']}"
        )
        if state == "run":
            print("      " + subprocess.list2cmdline(run["command"]))


def main():
    args = parse_args()
    batch_dir, manifest = prepare_batch(args)
    existing_rows = read_csv(batch_dir / "runs.csv")
    rows = recover_results(manifest, existing_rows)
    write_aggregates(batch_dir, manifest, rows)

    completed_ids = {row["run_id"] for row in rows if row["status"] == "completed"}
    print_plan(manifest, completed_ids)
    if not args.execute:
        print("[batch] dry run only; add --execute to connect to hardware")
        return

    rollout_root = Path(manifest["rollout_root"])
    delay = float(manifest["parameters"]["inter_run_delay_s"])
    by_id = {row["run_id"]: row for row in rows}

    for run in manifest["runs"]:
        if run["run_id"] in completed_ids:
            continue
        before = {p.name for p in rollout_root.iterdir()} if rollout_root.exists() else set()
        print(f"\n[batch] starting {run['run_id']}")
        try:
            returncode = subprocess.run(run["command"], cwd=REPO_ROOT).returncode
        except KeyboardInterrupt:
            after = {p.name for p in rollout_root.iterdir()} if rollout_root.exists() else set()
            rollout = find_rollout(run, rollout_root, after - before)
            if rollout is not None and (rollout / "summary.json").exists():
                result = extract_metrics(run, rollout)
                result["status"] = "interrupted"
            else:
                result = incomplete_result(run, "interrupted", rollout, "keyboard_interrupt")
            by_id[run["run_id"]] = result
            write_aggregates(batch_dir, manifest, list(by_id.values()))
            print("\n[batch] interrupted after child safety cleanup")
            break

        after = {p.name for p in rollout_root.iterdir()} if rollout_root.exists() else set()
        rollout = find_rollout(run, rollout_root, after - before)
        if returncode != 0 or rollout is None or not (rollout / "summary.json").exists():
            result = incomplete_result(run, "failed", rollout, f"returncode_{returncode}")
            by_id[run["run_id"]] = result
            write_aggregates(batch_dir, manifest, list(by_id.values()))
            print(f"[batch] failed: returncode={returncode}, rollout={rollout}")
            break

        result = extract_metrics(run, rollout)
        by_id[run["run_id"]] = result
        rows = list(by_id.values())
        write_aggregates(batch_dir, manifest, rows)
        print(
            f"[batch] {run['run_id']} status={result['status']} "
            f"reason={result['done_reason']} TIZ={result['zpd_occupancy_fraction']} "
            f"duration={result['duration_s']}"
        )
        if result["status"] != "completed":
            print("[batch] stopping before the next hardware rollout")
            break
        if delay > 0:
            time.sleep(delay)

    print(f"\n[batch] runs: {batch_dir / 'runs.csv'}")
    print(f"[batch] summary: {batch_dir / 'condition_summary.csv'}")
    print(f"[batch] pairs: {batch_dir / 'paired_results.csv'}")


if __name__ == "__main__":
    main()
