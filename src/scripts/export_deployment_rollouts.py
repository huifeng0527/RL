import argparse
import csv
import json
from pathlib import Path

import numpy as np


SUMMARY_FIELDS = [
    "rollout_id",
    "subject",
    "condition",
    "duration_s",
    "num_control_steps",
    "camera_update_rate_hz_mean",
    "camera_update_rate_hz_median",
    "control_loop_rate_hz_mean",
    "control_loop_rate_hz_median",
    "policy_inference_latency_ms_mean",
    "policy_inference_latency_ms_p95",
    "dead_reckoning_fraction",
    "dead_reckoning_steps",
    "safety_stop_count",
    "distance_cm_mean",
    "distance_cm_median",
    "distance_cm_std",
    "zpd_occupancy_fraction",
    "zpd_occupancy_steps",
    "zpd_low_cm",
    "zpd_high_cm",
    "done_reason",
    "snapshot_path",
    "video_path",
]

DISTANCE_FIELDS = [
    "rollout_id",
    "subject",
    "condition",
    "t_task_s",
    "distance_cm",
    "in_zpd",
    "dead_reckoning_used",
]

PAPER_METRICS = [
    ("Camera update rate", "camera_update_rate_hz_mean", "Hz", "mean_std"),
    ("Control-loop rate", "control_loop_rate_hz_mean", "Hz", "mean_std"),
    ("Policy inference latency", "policy_inference_latency_ms_mean", "ms", "mean_std"),
    ("Dead-reckoning usage", "dead_reckoning_fraction", "%", "percent_mean_std"),
    ("ZPD occupancy", "zpd_occupancy_fraction", "%", "percent_mean_std"),
    ("Rollout duration", "duration_s", "s", "mean_std"),
    ("Safety stops", "safety_stop_count", "count", "sum"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate real-world deployment chase rollouts.")
    parser.add_argument("--rollout-root", default="data/deployment_rollouts")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def has_timeseries_rows(rollout_dir):
    path = Path(rollout_dir) / "timeseries.csv"
    if not path.exists():
        return False
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return next(reader, None) is not None


def find_rollout_dirs(root):
    root = Path(root)
    if not root.exists():
        return []
    skipped = []
    valid = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / "summary.json").exists() and has_timeseries_rows(p):
            valid.append(p)
        elif (p / "timeseries.csv").exists():
            skipped.append(p.name)
    if skipped:
        print(f"Skipped empty rollout folders: {', '.join(skipped)}")
    return sorted(valid)


def read_timeseries_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def to_float(value):
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def format_mean_std(values, scale=1.0, suffix=""):
    arr = np.asarray([to_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)] * scale
    if arr.size == 0:
        return ""
    return f"{np.mean(arr):.2f} ± {np.std(arr, ddof=0):.2f}{suffix}"


def format_sum(values):
    arr = np.asarray([to_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "0"
    return str(int(np.sum(arr)))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_paper_table(summary_rows):
    rows = []
    for label, field, unit, mode in PAPER_METRICS:
        values = [row.get(field) for row in summary_rows]
        if mode == "percent_mean_std":
            value = format_mean_std(values, scale=100.0, suffix="%")
        elif mode == "sum":
            value = format_sum(values)
        else:
            value = format_mean_std(values)
        rows.append({"metric": label, "value": value, "unit": unit})
    return rows


def write_excel(path, summary_rows, distance_rows, paper_rows):
    try:
        import pandas as pd
    except ImportError:
        return False

    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(paper_rows).to_excel(writer, sheet_name="Paper_Timing_Table", index=False)
        pd.DataFrame(distance_rows).to_excel(writer, sheet_name="Distance_Samples", index=False)
    return True


def main():
    args = parse_args()
    rollout_root = Path(args.rollout_root)
    out_dir = Path(args.out_dir) if args.out_dir else rollout_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_dirs = find_rollout_dirs(rollout_root)
    if not rollout_dirs:
        raise SystemExit(f"No rollout folders found under {rollout_root}")

    summary_rows = []
    distance_rows = []
    for rollout_dir in rollout_dirs:
        summary = load_json(rollout_dir / "summary.json")
        metadata = load_json(rollout_dir / "metadata.json") if (rollout_dir / "metadata.json").exists() else {}
        row = {field: summary.get(field, metadata.get(field, "")) for field in SUMMARY_FIELDS}
        row["rollout_id"] = summary.get("rollout_id", rollout_dir.name)
        row["subject"] = summary.get("subject", metadata.get("subject", ""))
        row["condition"] = summary.get("condition", metadata.get("condition", ""))
        summary_rows.append(row)

        for ts in read_timeseries_rows(rollout_dir / "timeseries.csv"):
            distance_rows.append({
                "rollout_id": row["rollout_id"],
                "subject": row["subject"],
                "condition": row["condition"],
                "t_task_s": ts.get("t_task_s", ""),
                "distance_cm": ts.get("distance_cm", ""),
                "in_zpd": to_bool(ts.get("in_zpd", "")),
                "dead_reckoning_used": to_bool(ts.get("dead_reckoning_used", "")),
            })

    paper_rows = build_paper_table(summary_rows)
    write_csv(out_dir / "deployment_rollout_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(out_dir / "deployment_distance_samples.csv", distance_rows, DISTANCE_FIELDS)
    write_csv(out_dir / "deployment_timing_table.csv", paper_rows, ["metric", "value", "unit"])

    excel_path = out_dir / "deployment_rollout_summary.xlsx"
    excel_written = write_excel(excel_path, summary_rows, distance_rows, paper_rows)

    print((out_dir / "deployment_rollout_summary.csv").as_posix())
    print((out_dir / "deployment_distance_samples.csv").as_posix())
    print((out_dir / "deployment_timing_table.csv").as_posix())
    if excel_written:
        print(excel_path.as_posix())
    else:
        print("Excel export skipped: pandas is not installed")


if __name__ == "__main__":
    main()
