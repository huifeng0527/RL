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
    return sorted(valid, key=lambda p: p.stat().st_mtime)


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
    return rollout_dirs[-1]


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


def sample_montage_times(ts, count):
    t = ts["t"][np.isfinite(ts["t"])]
    if t.size == 0:
        return np.linspace(0.0, 1.0, count)
    t_min = float(np.nanmin(t))
    t_max = float(np.nanmax(t))
    if t_max <= t_min:
        return np.full(count, t_min, dtype=float)
    return np.linspace(t_min, t_max, count)


def extract_video_frames(video_path, sample_times):
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

    t_min = float(sample_times[0])
    t_span = max(float(sample_times[-1] - sample_times[0]), 1e-6)
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


def load_montage_frames(rollout_dir, ts, count):
    sample_times = sample_montage_times(ts, count)
    video_path = rollout_dir / "annotated_video.mp4"
    frames = extract_video_frames(video_path, sample_times) if video_path.exists() else []
    if len(frames) < count:
        snapshot = rollout_dir / "snapshot_annotated.png"
        if not snapshot.exists():
            snapshot = rollout_dir / "snapshot.png"
        if snapshot.exists():
            image = plt.imread(snapshot)
            frames = (frames + [image] * count)[:count]
    return list(zip(frames[:count], sample_times[:len(frames)]))


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


def overlay_process_trace(ax, image, ts, t, metadata, zpd_low, zpd_high):
    end_idx = nearest_time_index(ts, t)
    if end_idx is None:
        return C["gray"], f"t={t:.1f}s"

    mic_x, mic_y = path_to_image_xy(ts, "microrobot_x", "microrobot_y", end_idx, image, metadata)
    hand_x, hand_y = path_to_image_xy(ts, "hand_x", "hand_y", end_idx, image, metadata)

    if mic_x.size > 1:
        ax.plot(mic_x, mic_y, color=C["blue"], lw=1.35, alpha=0.98)
    if hand_x.size > 1:
        ax.plot(hand_x, hand_y, color=C["orange"], lw=1.25, alpha=0.98)

    distance = ts["distance"][end_idx] if end_idx < ts["distance"].size else np.nan
    status, status_color = zpd_status(distance, zpd_low, zpd_high)

    if mic_x.size and hand_x.size:
        ax.plot([mic_x[-1], hand_x[-1]], [mic_y[-1], hand_y[-1]], color="white", lw=0.8, alpha=0.9)
        ax.scatter([mic_x[-1]], [mic_y[-1]], s=15, color=C["blue"], edgecolors="white", linewidths=0.45, zorder=5)
        ax.scatter([hand_x[-1]], [hand_y[-1]], s=15, color=C["orange"], edgecolors="white", linewidths=0.45, zorder=5)

    if np.isfinite(distance):
        return status_color, f"t={t:.1f}s  d={distance:.1f} cm  {status}"
    return status_color, f"t={t:.1f}s  {status}"


def plot_montage(fig, slot, rollout_dir, ts, metadata, zpd_low, zpd_high, rows=2, cols=5):
    count = rows * cols
    sub = slot.subgridspec(rows + 1, cols, height_ratios=[0.16] + [1.0] * rows, hspace=0.035, wspace=0.015)
    title_ax = fig.add_subplot(sub[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.48, "(a) Time-lapse deployment frames with cumulative traces", ha="left", va="center", fontsize=9, fontweight="bold")
    title_ax.text(0.995, 0.48, "blue: microrobot; orange: hand", ha="right", va="center", fontsize=7, color=C["gray"])

    frames = load_montage_frames(rollout_dir, ts, count)
    for idx in range(count):
        ax = fig.add_subplot(sub[idx // cols + 1, idx % cols])
        ax.set_xticks([])
        ax.set_yticks([])
        if idx < len(frames):
            image, t = frames[idx]
            ax.imshow(image, aspect="auto")
            image_h, image_w = image.shape[:2]
            ax.set_xlim(0, image_w)
            ax.set_ylim(image_h, 0)
            border_color, label = overlay_process_trace(ax, image, ts, t, metadata, zpd_low, zpd_high)
            ax.text(0.025, 0.92, label, transform=ax.transAxes, ha="left", va="top", fontsize=5.9, color="white", bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 1.0})
        else:
            border_color = "#D0D3D8"
            ax.set_facecolor("#F3F4F6")
            ax.text(0.5, 0.5, "frame not found", ha="center", va="center", fontsize=7, color=C["gray"])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
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


def plot_distance(ax, ts, zpd_low, zpd_high):
    ax.set_title("(c) Distance to ZPD band", loc="left", fontsize=8.5, fontweight="bold")
    mask = np.isfinite(ts["t"]) & np.isfinite(ts["distance"])
    t = ts["t"][mask]
    d = ts["distance"][mask]
    ax.axhspan(zpd_low, zpd_high, color=C["green_l"], zorder=0)
    ax.plot(t, d, color=C["purple"], lw=1.6)
    ax.axhline(zpd_low, color=C["green"], lw=0.8, ls="--")
    ax.axhline(zpd_high, color=C["green"], lw=0.8, ls="--")
    if t.size:
        ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
    y_min = min(np.nanmin(d) if d.size else zpd_low, zpd_low) - 0.5
    y_max = max(np.nanmax(d) if d.size else zpd_high, zpd_high) + 0.5
    ax.set_ylim(y_min, y_max)
    ax.text(0.03, 0.92, "target ZPD band", transform=ax.transAxes, fontsize=7, color=C["green"], va="top")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance $d_t$ (cm)")
    ax.spines[["top", "right"]].set_visible(False)


def plot_occupancy(ax, rollout_dirs):
    ax.set_title("(d) Rollout ZPD occupancy", loc="left", fontsize=8.5, fontweight="bold")
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
    metadata = load_json(rep_dir / "metadata.json")
    summary = load_json(rep_dir / "summary.json")
    zpd_low, zpd_high = choose_zpd(args, metadata, summary)

    fig = plt.figure(figsize=(7.3, 5.25), dpi=450)
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.988, top=0.915, bottom=0.105, wspace=0.34, hspace=0.40, height_ratios=[1.42, 1.0])
    fig.suptitle("Zero-shot physical deployment of the simulation-trained policy", fontsize=11, fontweight="bold", y=0.985)

    plot_montage(fig, gs[0, :], rep_dir, ts, metadata, zpd_low, zpd_high)
    plot_trajectory(fig.add_subplot(gs[1, 0]), ts, metadata)
    plot_distance(fig.add_subplot(gs[1, 1]), ts, zpd_low, zpd_high)
    plot_occupancy(fig.add_subplot(gs[1, 2]), rollout_dirs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        path = out_dir / f"{args.output_name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035, facecolor="white")
        print(path.as_posix())
    plt.close(fig)


if __name__ == "__main__":
    main()
