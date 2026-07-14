from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


ROBOT_ORDER = ["scripted_only", "single_h1", "league"]
ROBOT_LABELS = {
    "scripted_only": "Scripted-only",
    "single_h1": "Single-hand",
    "league": "League",
}
COLORS = {
    "scripted_only": "#4C78A8",
    "single_h1": "#F58518",
    "league": "#54A24B",
}


def polish_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)


def panel_label(ax, label: str):
    ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def latest_csv(input_dir: Path) -> Path:
    files = sorted(input_dir.glob("comparison_results_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No comparison_results_*.csv found in {input_dir}")
    return files[-1]


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def group_name(row: dict) -> str:
    if row["test_type"] == "scripted":
        return "Scripted tests"
    if row["test_type"] == "learned":
        return "Learned-hand tests"
    if row["test_type"] == "mouse":
        return "Mouse hand"
    return row["test_type"]


def summarize(rows: list[dict]) -> list[dict]:
    groups = []
    for row in rows:
        g = group_name(row)
        if g not in groups:
            groups.append(g)
    summary = []
    for robot in ROBOT_ORDER:
        robot_rows = [r for r in rows if r["robot_name"] == robot]
        if not robot_rows:
            continue
        for group in groups:
            group_rows = [r for r in robot_rows if group_name(r) == group]
            valid_rows = [r for r in group_rows if np.isfinite(f(r, "tiz_mean"))]
            if not valid_rows:
                continue
            vals = [f(r, "tiz_mean") for r in valid_rows]
            zpd = [f(r, "zpd_coverage_mean") for r in valid_rows]
            close = [f(r, "too_close_rate") for r in valid_rows]
            far = [f(r, "too_far_rate") for r in valid_rows]
            summary.append({
                "robot_name": robot,
                "robot_label": ROBOT_LABELS.get(robot, robot),
                "test_group": group,
                "n_tests": len(vals),
                "tiz_mean": float(np.mean(vals)),
                "tiz_worst": float(np.min(vals)),
                "tiz_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "zpd_coverage_mean": float(np.mean(zpd)),
                "too_close_rate": float(np.mean(close)),
                "too_far_rate": float(np.mean(far)),
            })
    return summary


def save_summary(out_dir: Path, summary: list[dict]):
    with (out_dir / "policy_comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "policy_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def matrix(summary: list[dict], key: str, groups: list[str], robots: list[str]) -> np.ndarray:
    by_key = {(r["robot_name"], r["test_group"]): r for r in summary}
    values = np.full((len(robots), len(groups)), np.nan, dtype=float)
    for i, robot in enumerate(robots):
        for j, group in enumerate(groups):
            row = by_key.get((robot, group))
            if row is not None:
                values[i, j] = float(row[key])
    return values


def plot_grouped_bars(ax, summary: list[dict], key: str, ylabel: str, title: str, ylim=(0, 1.0)):
    groups = []
    robots = []
    for row in summary:
        if row["test_group"] not in groups:
            groups.append(row["test_group"])
        if row["robot_name"] not in robots:
            robots.append(row["robot_name"])
    robots = [r for r in ROBOT_ORDER if r in robots] + [r for r in robots if r not in ROBOT_ORDER]
    values = matrix(summary, key, groups, robots)
    xs = np.arange(len(groups))
    width = 0.24 if len(robots) >= 3 else 0.32
    offsets = (np.arange(len(robots)) - (len(robots) - 1) / 2) * width
    for i, robot in enumerate(robots):
        ax.bar(xs + offsets[i], values[i], width=width, label=ROBOT_LABELS.get(robot, robot), color=COLORS.get(robot), alpha=0.88)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    polish_axis(ax)


def plot_failure_decomp(ax, summary: list[dict]):
    final_rows = [r for r in summary if r["robot_name"] == "league"]
    if not final_rows:
        ax.axis("off")
        return
    groups = [r["test_group"] for r in final_rows]
    close = np.array([float(r["too_close_rate"]) for r in final_rows])
    far = np.array([float(r["too_far_rate"]) for r in final_rows])
    zpd = np.array([float(r["zpd_coverage_mean"]) for r in final_rows])
    xs = np.arange(len(groups))
    ax.bar(xs, close, color="#d62728", alpha=0.82, label="Too close")
    ax.bar(xs, far, bottom=close, color="#ff7f0e", alpha=0.82, label="Too far")
    ax.plot(xs, zpd, "o-", color="#1f77b4", lw=1.7, ms=3.5, label="ZPD coverage")
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("Episode fraction")
    ax.set_ylim(0, 1.0)
    ax.set_title("League robot failure decomposition")
    polish_axis(ax)
    ax.legend(frameon=False, loc="upper right")


def make_figure(out_dir: Path, summary: list[dict], with_title: bool):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), dpi=300, constrained_layout=True)
    plot_grouped_bars(axes[0], summary, "tiz_mean", "Mean TIZ", "Mean robustness", ylim=(0, 0.8))
    panel_label(axes[0], "a")
    plot_grouped_bars(axes[1], summary, "tiz_worst", "Worst-test TIZ", "Worst-case robustness", ylim=(0, 0.8))
    panel_label(axes[1], "b")
    plot_failure_decomp(axes[2], summary)
    panel_label(axes[2], "c")
    axes[0].legend(frameon=False, loc="upper left")
    if with_title:
        fig.suptitle("Fixed-test robustness comparison", fontsize=13, y=1.04)
    name = "paper_policy_comparison.png" if with_title else "paper_policy_comparison_no_title.png"
    fig.savefig(out_dir / name, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compose fixed-test policy comparison figure")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    input_csv = args.csv or latest_csv(args.input_dir)
    out_dir = args.out_dir or args.input_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_csv)
    summary = summarize(rows)
    if not summary:
        raise RuntimeError(f"No summary rows from {input_csv}")
    save_summary(out_dir, summary)
    make_figure(out_dir, summary, with_title=True)
    make_figure(out_dir, summary, with_title=False)
    print((out_dir / "paper_policy_comparison.png").as_posix())
    print((out_dir / "paper_policy_comparison_no_title.png").as_posix())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
