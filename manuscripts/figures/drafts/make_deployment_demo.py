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


def find_rollout_dirs(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "timeseries.csv").exists()],
        key=lambda p: p.stat().st_mtime,
    )


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


def plot_snapshot(ax, rollout_dir):
    snapshot = rollout_dir / "snapshot_annotated.png"
    if not snapshot.exists():
        snapshot = rollout_dir / "snapshot.png"
    ax.set_title("(a) Physical deployment snapshot", loc="left", fontsize=9, fontweight="bold")
    ax.axis("off")
    if snapshot.exists():
        image = plt.imread(snapshot)
        ax.imshow(image)
    else:
        ax.add_patch(Rectangle((0.08, 0.12), 0.84, 0.72, facecolor="#F3F4F6", edgecolor="#AEB7C2", linewidth=1.0, linestyle=(0, (4, 3))))
        ax.text(0.5, 0.50, "snapshot not found", ha="center", va="center", fontsize=8, color=C["gray"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


def plot_trajectory(ax, ts, metadata):
    ax.set_title("(b) Representative closed-loop rollout", loc="left", fontsize=9, fontweight="bold")
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
    ax.set_title("(c) Interaction difficulty in one rollout", loc="left", fontsize=9, fontweight="bold")
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
    ax.set_title("(d) ZPD occupancy across rollouts", loc="left", fontsize=9, fontweight="bold")
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

    fig = plt.figure(figsize=(7.3, 4.6), dpi=450)
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.985, top=0.90, bottom=0.115, wspace=0.26, hspace=0.42)
    fig.suptitle("Zero-shot physical deployment of the simulation-trained policy", fontsize=11, fontweight="bold", y=0.985)

    plot_snapshot(fig.add_subplot(gs[0, 0]), rep_dir)
    plot_trajectory(fig.add_subplot(gs[0, 1]), ts, metadata)
    plot_distance(fig.add_subplot(gs[1, 0]), ts, zpd_low, zpd_high)
    plot_occupancy(fig.add_subplot(gs[1, 1]), rollout_dirs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        path = out_dir / f"{args.output_name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035, facecolor="white")
        print(path.as_posix())
    plt.close(fig)


if __name__ == "__main__":
    main()
