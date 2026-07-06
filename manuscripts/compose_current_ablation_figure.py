from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("MLP", "1_MLP_Only", "#4C78A8"),
    ("MLP+Seq", "2_MLP_LSTM", "#54A24B"),
    ("Old Aux", "3_MLP_LSTM_AUX", "#F58518"),
]

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def polish_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)


def load_metrics(ablation_dir: Path) -> dict:
    loaded = {}
    for label, subdir, color in METHODS:
        path = ablation_dir / subdir / "ablation_metrics.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[label] = {
            "steps": np.array([d["timestep"] for d in data], dtype=float) / 1_000_000,
            "reward": np.array([d["reward_mean"] for d in data], dtype=float),
            "reward_std": np.array([d.get("reward_std", 0.0) for d in data], dtype=float),
            "length": np.array([d["episode_length_mean"] for d in data], dtype=float),
            "length_std": np.array([d.get("episode_length_std", 0.0) for d in data], dtype=float),
            "raw": data,
            "color": color,
        }
    return loaded


def summarize(metrics: dict) -> list[dict]:
    rows = []
    for label, data in metrics.items():
        reward_idx = int(np.argmax(data["reward"]))
        length_idx = int(np.argmax(data["length"]))
        rows.append({
            "method": label,
            "best_reward_step": float(data["steps"][reward_idx]),
            "best_reward": float(data["reward"][reward_idx]),
            "length_at_best_reward": float(data["length"][reward_idx]),
            "best_length_step": float(data["steps"][length_idx]),
            "best_length": float(data["length"][length_idx]),
            "reward_at_best_length": float(data["reward"][length_idx]),
            "final_reward": float(data["reward"][-1]),
            "final_length": float(data["length"][-1]),
        })
    return rows


def write_summary(out_dir: Path, rows: list[dict]):
    with (out_dir / "network_ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "network_ablation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot_curves(ax, metrics: dict, key: str, ylabel: str, title: str):
    for label, data in metrics.items():
        y = data[key]
        std = data[f"{key}_std"]
        ax.plot(data["steps"], y, "-o", lw=1.9, ms=3.4, label=label, color=data["color"])
        if np.any(std > 0):
            ax.fill_between(data["steps"], y - std, y + std, color=data["color"], alpha=0.12, linewidth=0)
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    polish_axis(ax)


def plot_best_bars(ax, rows: list[dict]):
    labels = [r["method"] for r in rows]
    values = np.array([r["best_reward"] for r in rows], dtype=float)
    colors = [dict((label, color) for label, _, color in METHODS)[label] for label in labels]
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=colors, alpha=0.88)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Best evaluation reward")
    ax.set_title("Best observed performance")
    for x, value in zip(xs, values):
        va = "bottom" if value >= 0 else "top"
        y = value + (0.35 if value >= 0 else -0.35)
        ax.text(x, y, f"{value:.1f}", ha="center", va=va, fontsize=8)
    polish_axis(ax)


def make_figure(out_dir: Path, metrics: dict, rows: list[dict], with_title: bool):
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), dpi=300, constrained_layout=True)
    plot_curves(axes[0], metrics, "reward", "Evaluation reward", "Reward curve")
    plot_curves(axes[1], metrics, "length", "Episode length", "Episode length curve")
    plot_best_bars(axes[2], rows)
    axes[0].legend(frameon=False, loc="lower right")
    if with_title:
        fig.suptitle("Network ablation under mixed scripted/RL-hand evaluation", fontsize=13, y=1.04)
    out_name = "paper_network_ablation.png" if with_title else "paper_network_ablation_no_title.png"
    fig.savefig(out_dir / out_name, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compose current network ablation figure")
    parser.add_argument("--ablation_dir", type=Path, default=Path("logs/ablation_scripted_h1_mix"))
    parser.add_argument("--out_dir", type=Path, default=Path("manuscripts/current_network_ablation"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.ablation_dir)
    rows = summarize(metrics)
    write_summary(args.out_dir, rows)
    make_figure(args.out_dir, metrics, rows, with_title=True)
    make_figure(args.out_dir, metrics, rows, with_title=False)
    print((args.out_dir / "paper_network_ablation.png").as_posix())
    print((args.out_dir / "paper_network_ablation_no_title.png").as_posix())
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
