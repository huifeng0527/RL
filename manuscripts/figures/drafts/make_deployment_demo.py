import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

C = {
    "black": "#222222",
    "gray": "#666A70",
    "blue": "#2F6FA5",
    "blue_l": "#E8F1FA",
    "orange": "#C76B2D",
    "orange_l": "#FFF0E3",
    "green": "#3F8C4D",
    "green_l": "#EAF6EA",
    "purple": "#7358A6",
    "purple_l": "#F1ECF8",
    "red": "#C92525",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the real-world deployment figure from chase rollout logs.")
    parser.add_argument("--rollout-root", default="data/deployment_rollouts")
    parser.add_argument("--representative-rollout", default=None)
    parser.add_argument("--out-dir", default="manuscripts/figures/drafts")
    parser.add_argument("--output-name", default="fig_deployment_demo")
    parser.add_argument("--zpd-low", type=float, default=None)
    parser.add_argument("--zpd-high", type=float, default=None)
    return parser.parse_args()


def has_timeseries_rows(rollout_dir):
    path = Path(rollout_dir) / "timeseries.csv"
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as f:
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
        if has_timeseries_rows(p):
            valid.append(p)
        elif (p / "timeseries.csv").exists():
            skipped.append(p.name)
    if skipped:
        print(f"Skipped empty rollout folders: {', '.join(skipped)}")
    return sorted(valid)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def load_timeseries(rollout_dir):
    rows = []
    with (Path(rollout_dir) / "timeseries.csv").open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {Path(rollout_dir) / 'timeseries.csv'}")

    def arr(name):
        return np.asarray([to_float(row.get(name)) for row in rows], dtype=float)

    def bool_arr(name):
        return np.asarray([to_bool(row.get(name)) for row in rows], dtype=bool)

    return {
        "t": arr("t_task_s"),
        "hand_x": arr("hand_x_cm"),
        "hand_y": arr("hand_y_cm"),
        "robot_x": arr("robot_x_cm"),
        "robot_y": arr("robot_y_cm"),
        "microrobot_x": arr("microrobot_x_cm"),
        "microrobot_y": arr("microrobot_y_cm"),
        "distance": arr("distance_cm"),
        "in_zpd": bool_arr("in_zpd"),
    }


def finite_xy(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], mask


def representative_rollout(args, rollout_dirs):
    if args.representative_rollout:
        path = Path(args.representative_rollout)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for rollout_dir in rollout_dirs:
        if (rollout_dir / "annotated_video.mp4").exists():
            return rollout_dir
    return rollout_dirs[0]


def choose_zpd(args, metadata, summary):
    low = args.zpd_low
    high = args.zpd_high
    if low is None:
        low = summary.get("zpd_low_cm", metadata.get("zpd_low_cm", 3.5))
    if high is None:
        high = summary.get("zpd_high_cm", metadata.get("zpd_high_cm", 5.5))
    return float(low), float(high)


def load_occupancies(rollout_dirs):
    labels = []
    values = []
    for idx, rollout_dir in enumerate(rollout_dirs, start=1):
        summary = load_json(rollout_dir / "summary.json")
        occupancy = summary.get("zpd_occupancy_fraction")
        if occupancy is None:
            ts = load_timeseries(rollout_dir)
            occupancy = float(np.mean(ts["in_zpd"])) if ts["in_zpd"].size else np.nan
        labels.append(str(idx))
        values.append(to_float(occupancy) * 100.0)
    return labels, np.asarray(values, dtype=float)


def load_all_timeseries(rollout_dirs):
    all_ts = []
    for rollout_dir in rollout_dirs:
        try:
            all_ts.append(load_timeseries(rollout_dir))
        except ValueError:
            continue
    return all_ts


def empty_xy_like(t):
    return np.full_like(t, np.nan, dtype=float)


def load_summary_rows(root):
    path = Path(root) / "aggregate" / "deployment_rollout_summary.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_aggregate_timeseries(root):
    path = Path(root) / "aggregate" / "deployment_distance_samples.csv"
    if not path.exists():
        return [], {}
    grouped = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rollout_id = row.get("rollout_id", "")
            if not rollout_id:
                continue
            grouped.setdefault(rollout_id, []).append(row)

    all_ts = []
    by_id = {}
    for rollout_id, rows in grouped.items():
        t = np.asarray([to_float(row.get("t_task_s")) for row in rows], dtype=float)
        distance = np.asarray([to_float(row.get("distance_cm")) for row in rows], dtype=float)
        order = np.argsort(np.nan_to_num(t, nan=np.inf))
        t = t[order]
        distance = distance[order]
        in_zpd = np.asarray([to_bool(rows[i].get("in_zpd")) for i in order], dtype=bool)
        ts = {
            "t": t,
            "hand_x": empty_xy_like(t),
            "hand_y": empty_xy_like(t),
            "robot_x": empty_xy_like(t),
            "robot_y": empty_xy_like(t),
            "microrobot_x": empty_xy_like(t),
            "microrobot_y": empty_xy_like(t),
            "distance": distance,
            "in_zpd": in_zpd,
        }
        all_ts.append(ts)
        by_id[rollout_id] = ts
    return all_ts, by_id


def select_aggregate_storyboard(root, summary_rows, by_id):
    for row in summary_rows:
        rollout_id = row.get("rollout_id", "")
        video_path = row.get("video_path", "")
        if rollout_id in by_id and video_path and Path(video_path).exists():
            return by_id[rollout_id], {"video_path": video_path, "snapshot_path": row.get("snapshot_path", "")}, row
    for rollout_id, ts in by_id.items():
        return ts, {"video_path": "", "snapshot_path": ""}, {"rollout_id": rollout_id}
    raise ValueError("No aggregate rollout samples found")


def aggregate_distance_profile(all_ts, points=300):
    finite_ranges = []
    for ts in all_ts:
        mask = np.isfinite(ts["t"]) & np.isfinite(ts["distance"])
        if np.count_nonzero(mask) >= 2:
            finite_ranges.append((float(np.nanmin(ts["t"][mask])), float(np.nanmax(ts["t"][mask]))))
    if not finite_ranges:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])

    t_min = min(lo for lo, _ in finite_ranges)
    t_max = max(hi for _, hi in finite_ranges)
    grid = np.linspace(t_min, t_max, points)
    aligned = []
    for ts in all_ts:
        mask = np.isfinite(ts["t"]) & np.isfinite(ts["distance"])
        if np.count_nonzero(mask) < 2:
            continue
        t = ts["t"][mask]
        d = ts["distance"][mask]
        order = np.argsort(t)
        t = t[order]
        d = d[order]
        unique_t, unique_idx = np.unique(t, return_index=True)
        unique_d = d[unique_idx]
        interp = np.interp(grid, unique_t, unique_d, left=np.nan, right=np.nan)
        aligned.append(interp)

    values = np.vstack(aligned) if aligned else np.empty((0, grid.size))
    mean = np.nanmean(values, axis=0) if values.size else np.full_like(grid, np.nan)
    std = np.nanstd(values, axis=0) if values.size else np.full_like(grid, np.nan)
    count = np.sum(np.isfinite(values), axis=0) if values.size else np.zeros_like(grid)
    mean[count == 0] = np.nan
    std[count == 0] = np.nan
    return grid, mean, std, count


def smooth_values(values, window=9):
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return values
    window = min(window, values.size if values.size % 2 else values.size - 1)
    window = max(window, 3)
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_hand_speed(ts):
    speed = np.full_like(ts["t"], np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(ts["t"]) & np.isfinite(ts["hand_x"]) & np.isfinite(ts["hand_y"]))
    if valid.size < 3:
        return speed

    t = ts["t"][valid]
    x = ts["hand_x"][valid]
    y = ts["hand_y"][valid]
    order = np.argsort(t)
    valid = valid[order]
    t = t[order]
    x = x[order]
    y = y[order]
    unique_t, unique_idx = np.unique(t, return_index=True)
    if unique_t.size < 3:
        return speed
    valid = valid[unique_idx]
    x = x[unique_idx]
    y = y[unique_idx]

    vx = np.gradient(smooth_values(x), unique_t)
    vy = np.gradient(smooth_values(y), unique_t)
    raw_speed = np.sqrt(vx ** 2 + vy ** 2)
    speed[valid] = smooth_values(raw_speed)
    return speed


def distance_gradient(ts):
    grad = np.full_like(ts["distance"], np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(ts["t"]) & np.isfinite(ts["distance"]))
    if valid.size < 3:
        return grad
    t = ts["t"][valid]
    d = smooth_values(ts["distance"][valid])
    if np.unique(t).size < 3:
        return grad
    grad[valid] = np.gradient(d, t)
    return grad


def choose_event_index(ts, zpd_low, zpd_high, lo_frac, hi_frac, mode, grad):
    valid = np.flatnonzero(np.isfinite(ts["t"]) & np.isfinite(ts["distance"]))
    if valid.size == 0:
        return None

    t_all = ts["t"][valid]
    t_min = float(np.nanmin(t_all))
    t_max = float(np.nanmax(t_all))
    span = max(t_max - t_min, 1e-6)
    lo = t_min + lo_frac * span
    hi = t_min + hi_frac * span
    candidates = valid[(ts["t"][valid] >= lo) & (ts["t"][valid] <= hi)]
    if candidates.size == 0:
        target = t_min + 0.5 * (lo_frac + hi_frac) * span
        return int(valid[np.argmin(np.abs(ts["t"][valid] - target))])

    if mode == "start":
        return int(candidates[0])
    if mode == "end":
        return int(candidates[-1])

    candidate_grad = grad[candidates]
    if mode == "drop" and np.isfinite(candidate_grad).any():
        return int(candidates[np.nanargmin(candidate_grad)])
    if mode == "rise" and np.isfinite(candidate_grad).any():
        return int(candidates[np.nanargmax(candidate_grad)])
    if mode == "steady":
        center = 0.5 * (zpd_low + zpd_high)
        scale = max(zpd_high - zpd_low, 1e-6)
        finite_grad = np.abs(grad[np.isfinite(grad)])
        grad_scale = max(float(np.nanpercentile(finite_grad, 75)) if finite_grad.size else 1.0, 1e-6)
        score = np.abs(ts["distance"][candidates] - center) / scale + 0.35 * np.nan_to_num(np.abs(candidate_grad), nan=grad_scale) / grad_scale
        return int(candidates[np.argmin(score)])

    target = t_min + 0.5 * (lo_frac + hi_frac) * span
    return int(candidates[np.argmin(np.abs(ts["t"][candidates] - target))])


def select_key_events(ts, zpd_low, zpd_high):
    grad = distance_gradient(ts)
    specs = [
        ("Start", 0.02, 0.08, "start"),
        ("Accelerate", 0.12, 0.32, "drop"),
        ("Steady I", 0.34, 0.50, "steady"),
        ("Slow down", 0.52, 0.68, "rise"),
        ("Steady II", 0.70, 0.86, "steady"),
        ("End", 0.92, 0.98, "end"),
    ]
    events = []
    for number, (label, lo, hi, mode) in enumerate(specs, start=1):
        idx = choose_event_index(ts, zpd_low, zpd_high, lo, hi, mode, grad)
        if idx is None:
            continue
        status, color = zpd_status(ts["distance"][idx], zpd_low, zpd_high)
        events.append({
            "number": number,
            "label": label,
            "index": int(idx),
            "time": float(ts["t"][idx]),
            "distance": float(ts["distance"][idx]),
            "status": status,
            "color": color,
        })
    return events


def extract_video_frames(video_path, sample_times, ts):
    try:
        import cv2
    except ImportError:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return []

    valid_t = ts["t"][np.isfinite(ts["t"])]
    t_min = float(np.nanmin(valid_t)) if valid_t.size else float(sample_times[0])
    t_max = float(np.nanmax(valid_t)) if valid_t.size else float(sample_times[-1])
    t_span = max(t_max - t_min, 1e-6)
    frames = []
    for t in sample_times:
        frame_idx = int(round((float(t) - t_min) / t_span * (frame_count - 1)))
        frame_idx = min(max(frame_idx, 0), frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_event_frames(rollout_dir, ts, events):
    sample_times = np.asarray([event["time"] for event in events], dtype=float)
    video_path = rollout_dir / "annotated_video.mp4"
    frames = extract_video_frames(video_path, sample_times, ts) if video_path.exists() and sample_times.size else []
    if len(frames) < len(events):
        snapshot = rollout_dir / "snapshot_annotated.png"
        if not snapshot.exists():
            snapshot = rollout_dir / "snapshot.png"
        if snapshot.exists():
            image = plt.imread(snapshot)
            frames = (frames + [image] * len(events))[:len(events)]
    return list(zip(events[:len(frames)], frames[:len(events)]))


def nearest_time_index(ts, t):
    diff = np.abs(ts["t"] - float(t))
    diff[~np.isfinite(diff)] = np.inf
    if not np.isfinite(diff).any():
        return None
    return int(np.argmin(diff))


def cm_to_image_xy(x_cm, y_cm, image, metadata):
    image_h, image_w = image.shape[:2]
    workspace_w = float(metadata.get("workspace_width_cm", 15.0))
    workspace_h = float(metadata.get("workspace_height_cm", 10.0))
    return x_cm * image_w / workspace_w, y_cm * image_h / workspace_h


def path_to_image_xy(ts, x_name, y_name, end_idx, image, metadata):
    x = ts[x_name][:end_idx + 1]
    y = ts[y_name][:end_idx + 1]
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return np.asarray([]), np.asarray([])
    return cm_to_image_xy(x[mask], y[mask], image, metadata)


def zpd_status(distance, zpd_low, zpd_high):
    if not np.isfinite(distance):
        return "unknown", C["gray"]
    if distance < zpd_low:
        return "near", C["orange"]
    if distance > zpd_high:
        return "far", C["purple"]
    return "ZPD", C["green"]


def overlay_event_points(ax, image, ts, event, metadata):
    idx = event["index"]
    mic_x = ts["microrobot_x"][idx]
    mic_y = ts["microrobot_y"][idx]
    if not np.isfinite(mic_x) or not np.isfinite(mic_y):
        mic_x = ts["robot_x"][idx]
        mic_y = ts["robot_y"][idx]
    hand_x = ts["hand_x"][idx]
    hand_y = ts["hand_y"][idx]

    if np.isfinite(mic_x) and np.isfinite(mic_y) and np.isfinite(hand_x) and np.isfinite(hand_y):
        mx, my = cm_to_image_xy(mic_x, mic_y, image, metadata)
        hx, hy = cm_to_image_xy(hand_x, hand_y, image, metadata)
        ax.plot([mx, hx], [my, hy], color="white", lw=0.9, alpha=0.85, zorder=4)
        ax.scatter([mx], [my], s=28, color=C["blue"], edgecolors="white", linewidths=0.55, zorder=5)
        ax.scatter([hx], [hy], s=28, color=C["orange"], edgecolors="white", linewidths=0.55, zorder=5)


def plot_event_storyboard(fig, slot, rollout_dir, ts, metadata, events):
    cols = max(len(events), 1)
    sub = slot.subgridspec(2, cols, height_ratios=[0.17, 1.0], hspace=0.035, wspace=0.018)
    title_ax = fig.add_subplot(sub[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.48, "(a) Key deployment frames matched to panel (b)", ha="left", va="center", fontsize=9, fontweight="bold")
    title_ax.text(0.995, 0.48, "video frames from first rollout; numbered times match the ZPD plot", ha="right", va="center", fontsize=7, color=C["gray"])

    frames = load_event_frames(rollout_dir, ts, events)
    for idx in range(cols):
        ax = fig.add_subplot(sub[1, idx])
        ax.set_xticks([])
        ax.set_yticks([])
        if idx < len(frames):
            event, image = frames[idx]
            ax.imshow(image, aspect="auto")
            image_h, image_w = image.shape[:2]
            ax.set_xlim(0, image_w)
            ax.set_ylim(image_h, 0)
            overlay_event_points(ax, image, ts, event, metadata)
            border_color = event["color"]
            ax.text(0.045, 0.10, str(event["number"]), transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, fontweight="bold", color="white", bbox={"boxstyle": "circle,pad=0.24", "facecolor": border_color, "edgecolor": "white", "linewidth": 0.55})
            ax.text(0.045, 0.93, event["label"], transform=ax.transAxes, ha="left", va="top", fontsize=6.2, fontweight="bold", color="white", bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 1.2})
            ax.text(0.045, 0.79, f"t={event['time']:.1f}s, d={event['distance']:.1f} cm", transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color="white", bbox={"facecolor": "black", "alpha": 0.48, "edgecolor": "none", "pad": 1.0})
        else:
            border_color = "#D0D3D8"
            ax.set_facecolor("#F3F4F6")
            ax.text(0.5, 0.5, "frame not found", ha="center", va="center", fontsize=7, color=C["gray"])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor(border_color)


def plot_trajectory(ax, ts, metadata):
    ax.set_title("(b) Closed-loop trajectories", loc="left", fontsize=8.5, fontweight="bold")
    w = float(metadata.get("workspace_width_cm", 15.0))
    h = float(metadata.get("workspace_height_cm", 10.0))
    ax.add_patch(Rectangle((0, 0), w, h, facecolor="#FBFCFD", edgecolor=C["black"], linewidth=0.9))

    mic_x, mic_y, mic_mask = finite_xy(ts["microrobot_x"], ts["microrobot_y"])
    label_robot = "microrobot"
    if mic_x.size < 5:
        mic_x, mic_y, mic_mask = finite_xy(ts["robot_x"], ts["robot_y"])
        label_robot = "robot command"
    hand_x, hand_y, hand_mask = finite_xy(ts["hand_x"], ts["hand_y"])

    if mic_x.size:
        ax.plot(mic_x, mic_y, color=C["blue"], lw=1.7, label=label_robot)
        ax.scatter([mic_x[0]], [mic_y[0]], s=24, color=C["green"], zorder=5)
        ax.scatter([mic_x[-1]], [mic_y[-1]], s=28, color=C["red"], zorder=5)
    if hand_x.size:
        ax.plot(hand_x, hand_y, color=C["orange"], lw=1.4, label="hand")

    ax.set_xlim(-0.2, w + 0.2)
    ax.set_ylim(-0.2, h + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=7)


def plot_dynamics(fig, slot, ts, zpd_low, zpd_high, events):
    sub = slot.subgridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.12)
    ax = fig.add_subplot(sub[0, 0])
    speed_ax = fig.add_subplot(sub[1, 0], sharex=ax)

    ax.set_title("(b) Representative rollout dynamics", loc="left", fontsize=8.5, fontweight="bold")
    mask = np.isfinite(ts["t"]) & np.isfinite(ts["distance"])
    t = ts["t"][mask]
    d = ts["distance"][mask]
    d_smooth = smooth_values(d, window=15) if d.size else d

    ax.axhspan(zpd_low, zpd_high, color=C["green_l"], zorder=0)
    ax.axhline(zpd_low, color=C["green"], lw=0.8, ls="--")
    ax.axhline(zpd_high, color=C["green"], lw=0.8, ls="--")
    if t.size:
        ax.plot(t, d_smooth, color=C["purple"], lw=1.75, zorder=2, label="smoothed distance")
        ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
        y_min = min(float(np.nanmin(d_smooth)), zpd_low) - 0.5
        y_max = max(float(np.nanmax(d_smooth)), zpd_high) + 0.5
    else:
        y_min = zpd_low - 0.5
        y_max = zpd_high + 0.5
    y_span = max(y_max - y_min, 1e-6)
    ax.set_ylim(y_min, y_max)
    ax.text(0.015, 0.90, "target ZPD band", transform=ax.transAxes, fontsize=7, color=C["green"], va="top")

    speed = compute_hand_speed(ts)
    speed_mask = np.isfinite(ts["t"]) & np.isfinite(speed)
    if np.count_nonzero(speed_mask) >= 2:
        speed_t = ts["t"][speed_mask]
        speed_values = smooth_values(speed[speed_mask], window=15)
        speed_ax.plot(speed_t, speed_values, color=C["orange"], lw=1.35, alpha=0.9)
        speed_max = float(np.nanpercentile(speed_values, 98))
        speed_ax.set_ylim(0, max(speed_max * 1.25, 1.0))

    for event in events:
        et = event["time"]
        ed = event["distance"]
        if not np.isfinite(et) or not np.isfinite(ed):
            continue
        ed_plot = float(np.interp(et, t, d_smooth)) if t.size else ed
        ax.axvline(et, color=event["color"], lw=0.75, ls=":", alpha=0.72, zorder=1)
        speed_ax.axvline(et, color=event["color"], lw=0.75, ls=":", alpha=0.72, zorder=1)
        ax.scatter([et], [ed_plot], s=34, color=event["color"], edgecolors="white", linewidths=0.65, zorder=4)
        label_y = ed_plot + 0.055 * y_span
        va = "bottom"
        if label_y > y_max - 0.08 * y_span:
            label_y = ed_plot - 0.065 * y_span
            va = "top"
        ax.text(et, label_y, str(event["number"]), ha="center", va=va, fontsize=7.0, fontweight="bold", color="white", bbox={"boxstyle": "circle,pad=0.22", "facecolor": event["color"], "edgecolor": "white", "linewidth": 0.55}, zorder=5)

    ax.set_ylabel("distance $d_t$ (cm)")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.tick_params(axis="x", labelbottom=False)
    ax.spines[["top", "right"]].set_visible(False)

    speed_ax.set_xlabel("time (s)")
    speed_ax.set_ylabel("hand speed\n(cm/s)", color=C["orange"])
    speed_ax.tick_params(axis="y", labelcolor=C["orange"], length=2, pad=1, labelsize=7)
    speed_ax.spines[["top", "right"]].set_visible(False)


def plot_occupancy(ax, rollout_dirs):
    ax.set_title("(c) Rollout ZPD occupancy", loc="left", fontsize=8.5, fontweight="bold")
    labels, values = load_occupancies(rollout_dirs)
    x = np.arange(len(values))
    colors = [C["blue"] if np.isfinite(v) else "#CCCCCC" for v in values]
    ax.bar(x, np.nan_to_num(values, nan=0.0), color=colors, width=0.68)
    finite = values[np.isfinite(values)]
    if finite.size:
        mean = float(np.mean(finite))
        std = float(np.std(finite, ddof=0))
        ax.axhline(mean, color=C["red"], lw=1.0, ls="--")
        ax.text(0.02, 0.94, f"mean {mean:.1f} ± {std:.1f}%", transform=ax.transAxes, fontsize=7, color=C["red"], va="top")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_xlabel("rollout")
    ax.set_ylabel("ZPD occupancy (%)")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    args = parse_args()
    rollout_dirs = find_rollout_dirs(args.rollout_root)
    if not rollout_dirs:
        raise SystemExit(f"No rollout folders with timeseries.csv found under {args.rollout_root}")

    rep_dir = representative_rollout(args, rollout_dirs)
    ts = load_timeseries(rep_dir)
    all_ts = load_all_timeseries(rollout_dirs)
    metadata = load_json(rep_dir / "metadata.json")
    summary = load_json(rep_dir / "summary.json")
    zpd_low, zpd_high = choose_zpd(args, metadata, summary)

    events = select_key_events(ts, zpd_low, zpd_high)

    fig = plt.figure(figsize=(7.45, 4.35), dpi=450)
    gs = fig.add_gridspec(2, 1, left=0.07, right=0.988, top=0.90, bottom=0.12, hspace=0.36, height_ratios=[1.24, 1.22])
    fig.suptitle("Zero-shot physical deployment of the simulation-trained policy", fontsize=11, fontweight="bold", y=0.984)

    plot_event_storyboard(fig, gs[0, 0], rep_dir, ts, metadata, events)
    plot_dynamics(fig, gs[1, 0], ts, zpd_low, zpd_high, events)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        path = out_dir / f"{args.output_name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035, facecolor="white")
        print(path.as_posix())
    plt.close(fig)


if __name__ == "__main__":
    main()
