"""Draw paper-style ablation figures from raw per-episode Excel data.

The Excel workbook should be produced by export_ablation_review.py. Smoothing is
controlled only here by --smooth_window, measured in environment timesteps.

Example:
    python src/scripts/plot_ablation_from_excel.py \
        --excel logs/ablation_gru_h1_h10_0626_2136/ablation_review_raw.xlsx \
        --smooth_window 500000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_ORDER = ["1_MLP", "2_GRU_Seq", "3_GRU_Aux"]
GROUP_LABELS = {
    "1_MLP": "MLP",
    "2_GRU_Seq": "GRU",
    "3_GRU_Aux": "GRU + Aux",
}
GROUP_COLORS = {
    "1_MLP": "#4C72B0",
    "2_GRU_Seq": "#DD8452",
    "3_GRU_Aux": "#55A868",
}
RAW_SHEET_PREFIX = "Episode_Raw"
MAX_PLOT_POINTS = 1500


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def save_figure(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def read_raw_episode_data(excel_path):
    xls = pd.ExcelFile(excel_path)
    raw_sheets = [s for s in xls.sheet_names if s.startswith(RAW_SHEET_PREFIX)]
    if not raw_sheets:
        raise ValueError(f"No {RAW_SHEET_PREFIX} sheets found in {excel_path}")
    frames = []
    for sheet in raw_sheets:
        frames.append(pd.read_excel(xls, sheet_name=sheet))
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["group", "timestep", "episode_index"])
    return data


def read_eval_data(excel_path):
    try:
        eval_df = pd.read_excel(excel_path, sheet_name="Eval_All")
    except ValueError:
        eval_df = pd.DataFrame()
    return eval_df


def ordered_groups(df):
    available = list(df["group"].dropna().unique())
    return [g for g in GROUP_ORDER if g in available] + [g for g in available if g not in GROUP_ORDER]


def bin_by_timestep(sub, metric, max_points=MAX_PLOT_POINTS):
    x = sub["timestep"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)
    if len(x) <= max_points:
        return x, y

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    bin_edges = np.linspace(x_min, x_max, max_points + 1)
    bin_idx = np.clip(np.searchsorted(bin_edges, x, side="right") - 1, 0, max_points - 1)
    sums = np.bincount(bin_idx, weights=y, minlength=max_points)
    counts = np.bincount(bin_idx, minlength=max_points)
    valid = counts > 0
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return x_centers[valid], sums[valid] / counts[valid]


def smooth_by_timesteps(x, y, smooth_window):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    smooth_window = float(smooth_window)
    if smooth_window <= 1 or len(y) < 3:
        return x, y

    bin_width = float(np.median(np.diff(x))) if len(x) > 1 else smooth_window
    rolling_points = max(1, int(round(smooth_window / max(bin_width, 1.0))))
    if rolling_points <= 1:
        return x, y
    y_smooth = pd.Series(y).rolling(window=rolling_points, center=True, min_periods=max(1, rolling_points // 4)).mean().to_numpy()
    valid = ~np.isnan(y_smooth)
    return x[valid], y_smooth[valid]


def plot_group_metric(ax, raw_df, group, metric, smooth_window, linewidth=1.7):
    sub = raw_df[raw_df["group"] == group].sort_values("timestep")
    x, y = bin_by_timestep(sub, metric)
    x, y = smooth_by_timesteps(x, y, smooth_window)
    ax.plot(
        x / 1e6,
        y,
        color=GROUP_COLORS.get(group, "gray"),
        linewidth=linewidth,
        label=GROUP_LABELS.get(group, group),
    )


def plot_training_curves(raw_df, out_dir, smooth_window):
    plt = setup_matplotlib()
    panels = [
        ("reward", "Episode reward", "A"),
        ("episode_length", "Episode length", "B"),
        ("zpd_steps", "Steps in ZPD", "C"),
        ("workspace_coverage", "Workspace coverage", "D"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes = axes.ravel()
    groups = ordered_groups(raw_df)
    for ax, (metric, ylabel, panel) in zip(axes, panels):
        for group in groups:
            plot_group_metric(ax, raw_df, group, metric, smooth_window)
        ax.set_xlabel("Timesteps (M)")
        ax.set_ylabel(ylabel)
        ax.text(-0.12, 1.06, panel, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_ablation_training_curves")
    plt.close(fig)


def plot_zpd_workspace(raw_df, out_dir, smooth_window):
    plt = setup_matplotlib()
    panels = [
        ("zpd_steps", "Steps in ZPD", "A"),
        ("workspace_coverage", "Workspace coverage", "B"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    groups = ordered_groups(raw_df)
    for ax, (metric, ylabel, panel) in zip(axes, panels):
        for group in groups:
            plot_group_metric(ax, raw_df, group, metric, smooth_window, linewidth=1.9)
        ax.set_xlabel("Timesteps (M)")
        ax.set_ylabel(ylabel)
        ax.text(-0.12, 1.06, panel, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_ablation_zpd_workspace")
    plt.close(fig)


def tail_summary(raw_df, tail_fraction=0.1):
    rows = []
    for group in ordered_groups(raw_df):
        sub = raw_df[raw_df["group"] == group].sort_values("timestep")
        tail = sub.iloc[int(len(sub) * (1.0 - tail_fraction)):]
        row = {
            "group": group,
            "label": GROUP_LABELS.get(group, group),
            "episodes": len(sub),
        }
        for metric in ["reward", "episode_length", "zpd_steps", "zpd_coverage", "workspace_coverage"]:
            row[f"{metric}_mean"] = float(tail[metric].mean())
            row[f"{metric}_std"] = float(tail[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary_bars(raw_df, out_dir, tail_fraction=0.1, show_error_bars=False):
    plt = setup_matplotlib()
    summary = tail_summary(raw_df, tail_fraction=tail_fraction)
    groups = ordered_groups(summary)
    metrics = [
        ("reward", "Episode reward"),
        ("episode_length", "Episode length"),
        ("zpd_steps", "Steps in ZPD"),
        ("workspace_coverage", "Workspace coverage"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(9.0, 2.8))
    for ax, (metric, ylabel) in zip(axes, metrics):
        means = []
        sems = []
        colors = []
        for group in groups:
            row = summary[summary["group"] == group].iloc[0]
            means.append(row[f"{metric}_mean"])
            sems.append(row[f"{metric}_std"] / np.sqrt(max(row["episodes"] * tail_fraction, 1)))
            colors.append(GROUP_COLORS.get(group, "gray"))
        x = np.arange(len(groups))
        yerr = sems if show_error_bars else None
        ax.bar(x, means, yerr=yerr, capsize=3 if show_error_bars else 0, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_ablation_final_bars")
    plt.close(fig)


def plot_eval_curves(eval_df, out_dir):
    if eval_df.empty:
        return
    plt = setup_matplotlib()
    panels = [
        ("reward_mean", "Eval reward", "A"),
        ("zpd_coverage_mean", "Eval ZPD coverage", "B"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    groups = ordered_groups(eval_df)
    for ax, (metric, ylabel, panel) in zip(axes, panels):
        for group in groups:
            sub = eval_df[eval_df["group"] == group].sort_values("timestep")
            ax.plot(
                sub["timestep_m"],
                sub[metric],
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                color=GROUP_COLORS.get(group, "gray"),
                label=GROUP_LABELS.get(group, group),
            )
        ax.set_xlabel("Timesteps (M)")
        ax.set_ylabel(ylabel)
        ax.text(-0.12, 1.06, panel, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_ablation_eval_curves")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot paper figures from raw ablation Excel data.")
    parser.add_argument("--excel", default="logs/ablation_gru_h1_h10_0626_2136/ablation_review_raw.xlsx")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--smooth_window", type=int, default=500000, help="Moving-average window in environment timesteps. This is the only smoothing control.")
    parser.add_argument("--tail_fraction", type=float, default=0.1)
    parser.add_argument("--show_error_bars", action="store_true")
    args = parser.parse_args()

    excel_path = Path(args.excel)
    out_dir = Path(args.output_dir) if args.output_dir else excel_path.parent / f"paper_figures_raw_smooth{args.smooth_window}"

    raw_df = read_raw_episode_data(excel_path)
    eval_df = read_eval_data(excel_path)

    plot_training_curves(raw_df, out_dir, smooth_window=args.smooth_window)
    plot_zpd_workspace(raw_df, out_dir, smooth_window=args.smooth_window)
    plot_summary_bars(raw_df, out_dir, tail_fraction=args.tail_fraction, show_error_bars=args.show_error_bars)
    plot_eval_curves(eval_df, out_dir)
    print(f"Figures written to: {out_dir}")


if __name__ == "__main__":
    main()
