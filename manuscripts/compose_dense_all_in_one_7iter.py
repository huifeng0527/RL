from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from stable_baselines3 import PPO

import generate_pfsp_window_aux_visual as aux

OUT = Path("manuscripts/current_league_pfsp_window_7iter")

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def panel_label(ax, label):
    ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def polish_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)


def plot_cross_heatmap(ax, tis):
    sns.heatmap(
        tis,
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=0.8,
        annot=True,
        fmt=".2f",
        square=False,
        linewidths=0.35,
        cbar=True,
        cbar_kws={"label": "TIS", "shrink": 0.82, "pad": 0.01},
        annot_kws={"fontsize": 6.5},
    )
    n = tis.shape[0]
    ax.set_xticklabels([f"H{i}" for i in range(1, n + 1)], rotation=0)
    ax.set_yticklabels([f"R{i}" for i in range(1, n + 1)], rotation=0)
    ax.set_xlabel("Hand generation")
    ax.set_ylabel("Robot generation")
    ax.set_title("Cross-iteration validation")
    panel_label(ax, "a")


def plot_summary(ax, tis):
    x = np.arange(1, tis.shape[0] + 1)
    mean = tis.mean(axis=1)
    worst = tis.min(axis=1)
    ax.plot(x, mean, "-o", lw=1.8, ms=3.5, label="Mean")
    ax.plot(x, worst, "-s", lw=1.8, ms=3.5, label="Worst")
    ax.set_ylim(0, 0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Robot gen.")
    ax.set_ylabel("TIS")
    ax.set_title("Robustness")
    polish_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "b")


def plot_pfsp_heatmap(ax, pfsp_rows):
    final_by_iter = {}
    for row in pfsp_rows:
        it = int(row["iteration"])
        t = int(row["timesteps"])
        final_by_iter[it] = max(final_by_iter.get(it, 0), t)
    iters = sorted(final_by_iter)
    max_pool = max(int(r["pool_size"]) for r in pfsp_rows)
    heat = np.full((len(iters), max_pool), np.nan)
    for i, it in enumerate(iters):
        rows = [r for r in pfsp_rows if int(r["iteration"]) == it and int(r["timesteps"]) == final_by_iter[it]]
        for r in rows:
            heat[i, int(r["opponent_index"])] = float(r["preference_ratio"])
    im = ax.imshow(np.ma.masked_invalid(heat), aspect="auto", cmap="RdBu_r", vmin=0, vmax=2.2)
    ax.set_yticks(range(len(iters)))
    ax.set_yticklabels([f"I{i}" for i in iters])
    ax.set_xticks(range(max_pool))
    ax.set_xticklabels(["Scr"] + [f"H{i}" for i in range(1, max_pool)], rotation=30, ha="right")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", fontsize=6)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Robot iter.")
    ax.set_title("PFSP preference / uniform")
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("ratio")
    cbar.ax.tick_params(labelsize=6)
    panel_label(ax, "c")


def plot_concentration(ax, pfsp_rows):
    final_by_iter = {}
    for row in pfsp_rows:
        iteration = int(row["iteration"])
        timestep = int(row["timesteps"])
        final_by_iter[iteration] = max(final_by_iter.get(iteration, 0), timestep)

    iterations = []
    pool_sizes = []
    effective_counts = []
    concentration = []
    for iteration in sorted(final_by_iter):
        rows = [
            r for r in pfsp_rows
            if int(r["iteration"]) == iteration and int(r["timesteps"]) == final_by_iter[iteration]
        ]
        if not rows:
            continue
        probs = np.array([float(r["probability"]) for r in sorted(rows, key=lambda x: int(x["opponent_index"]))], dtype=float)
        probs = probs / probs.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        effective = float(np.exp(entropy))
        pool_size = len(probs)
        iterations.append(iteration)
        pool_sizes.append(pool_size)
        effective_counts.append(effective)
        concentration.append(effective / pool_size)

    ax.plot(iterations, pool_sizes, "o--", lw=1.5, ms=3.2, color="#8c8c8c", label="Pool size")
    ax.plot(iterations, effective_counts, "o-", lw=1.8, ms=3.6, color="#1f77b4", label="Effective opponents")
    ax.set_xlabel("Robot iteration")
    ax.set_ylabel("Opponent count")
    ax.set_xticks(iterations)
    ymax = max(pool_sizes) + 0.7 if pool_sizes else 1
    ax.set_ylim(0, ymax)
    ax.set_title("PFSP concentration")
    polish_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(iterations, concentration, "s-", lw=1.4, ms=3.0, color="#d62728", alpha=0.75, label="Effective / pool")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Relative spread")
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax2.tick_params(width=0.8, length=3, labelsize=7)
    panel_label(ax, "d")


def plot_within(axs, pfsp_rows, iterations=(3, 5, 7)):
    for ax, it in zip(axs, iterations):
        rows = [r for r in pfsp_rows if int(r["iteration"]) == it]
        by_opp = {}
        for row in rows:
            by_opp.setdefault(int(row["opponent_index"]), []).append(row)
        # Keep readable: scripted, latest/newer hands, plus one older hard hand.
        keep = {0, max(1, it - 1), max(1, it - 2)}
        if it == 3:
            keep = {0, 1, 2}
        if it == 5:
            keep = {0, 1, 3, 4}
        if it == 7:
            keep = {0, 3, 4, 5, 6}
        for opp in sorted(by_opp):
            if opp not in keep:
                continue
            series = sorted(by_opp[opp], key=lambda x: float(x["iteration_progress"]))
            xs = [float(s["iteration_progress"]) for s in series]
            ys = [float(s["preference_ratio"]) for s in series]
            label = "Scr" if opp == 0 else f"H{opp}"
            lw = 1.4 if opp != 0 else 1.7
            ax.plot(xs, ys, lw=lw, label=label)
        ax.axhline(1.0, color="black", ls=":", lw=0.9)
        ax.set_ylim(0.35, 1.75)
        ax.set_title(f"Iter {it}")
        ax.set_xlabel("phase progress")
        polish_axis(ax)
        ax.legend(frameon=False, ncol=2, loc="best")
    axs[0].set_ylabel("Preference ratio")
    panel_label(axs[0], "e")


def plot_aux(axs, examples):
    for idx, (ax, ex) in enumerate(zip(axs, examples[:4]), start=1):
        past_hand, _, _, _, _, pred_future, true_future = aux.relative_paths(ex)
        ax.plot(past_hand[:, 0], past_hand[:, 1], "o-", color="#8c8c8c", lw=0.9, ms=2.0, label="past")
        ax.scatter([0], [0], marker="*", s=45, color="black", zorder=5, label="now")
        ax.plot(true_future[:, 0], true_future[:, 1], "o-", color="#1f77b4", lw=1.4, ms=2.3, label="actual")
        ax.plot(pred_future[:, 0], pred_future[:, 1], "s--", color="#ff7f0e", lw=1.4, ms=2.3, label="pred.")
        ax.set_title(f"Aux ex.{idx}, MSE={ex['trajectory_mse']:.2f}")
        ax.set_xlabel("rel. x")
        ax.set_ylabel("rel. y")
        polish_axis(ax)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="datalim")
    panel_label(axs[0], "f")
    handles, labels = axs[0].get_legend_handles_labels()
    return handles, labels


def plot_aux_error(ax, horizon_errors):
    mean = np.mean(horizon_errors, axis=0)
    stderr = np.std(horizon_errors, axis=0) / np.sqrt(max(1, len(horizon_errors)))
    steps = np.arange(1, aux.FUTURE_HORIZON + 1)
    ax.plot(steps, mean, "o-", color="#9467bd", lw=1.8, ms=3.5)
    ax.fill_between(steps, mean - stderr, mean + stderr, color="#9467bd", alpha=0.2)
    ax.set_xlabel("Future step")
    ax.set_ylabel("Trajectory MSE")
    ax.set_title("Aux error by horizon")
    ax.set_xticks(steps)
    polish_axis(ax)
    panel_label(ax, "g")


cross = load_json(OUT / "cross_iter_validation_7iter.json")
tis = np.asarray(cross["tis_matrix"], dtype=float)
pfsp_rows = load_csv(OUT / "pfsp_sampling_snapshots_7iter.csv")

robot_model = PPO.load(str(aux.ROBOT_PATH), verbose=0)
hand_model = PPO.load(str(aux.HAND_PATH), custom_objects={"learning_rate": 0.0, "optimizer_class": None}, verbose=0)
examples, horizon_errors = aux.collect_examples(robot_model, hand_model)

# Dense all-in-one: 4 rows x 4 cols; filled panels, no pasted image whitespace.
fig = plt.figure(figsize=(13.2, 10.2), dpi=300, constrained_layout=True)
gs = fig.add_gridspec(4, 4, width_ratios=[1.15, 1.15, 1.0, 1.0], height_ratios=[1.08, 0.92, 0.95, 0.95])

ax = fig.add_subplot(gs[0:2, 0:2])
plot_cross_heatmap(ax, tis)
ax = fig.add_subplot(gs[0, 2])
plot_summary(ax, tis)
ax = fig.add_subplot(gs[0, 3])
plot_concentration(ax, pfsp_rows)
ax = fig.add_subplot(gs[1, 2:4])
plot_pfsp_heatmap(ax, pfsp_rows)
within_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
plot_within(within_axes, pfsp_rows, iterations=(3, 5, 7))
ax = fig.add_subplot(gs[2, 3])
plot_aux_error(ax, horizon_errors)
aux_axes = [fig.add_subplot(gs[3, i]) for i in range(4)]
handles, labels = plot_aux(aux_axes, examples)
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=8)
fig.suptitle("Simulation analysis: cross-generation validation, PFSP curriculum, and auxiliary prediction", fontsize=13, y=1.02)
fig.savefig(OUT / "paper_dense_all_in_one_7iter.png", bbox_inches="tight")
plt.close(fig)

# Variant: no global title, for insertion into manuscript.
fig = plt.figure(figsize=(13.2, 10.0), dpi=300, constrained_layout=True)
gs = fig.add_gridspec(4, 4, width_ratios=[1.15, 1.15, 1.0, 1.0], height_ratios=[1.08, 0.92, 0.95, 0.95])
ax = fig.add_subplot(gs[0:2, 0:2]); plot_cross_heatmap(ax, tis)
ax = fig.add_subplot(gs[0, 2]); plot_summary(ax, tis)
ax = fig.add_subplot(gs[0, 3]); plot_concentration(ax, pfsp_rows)
ax = fig.add_subplot(gs[1, 2:4]); plot_pfsp_heatmap(ax, pfsp_rows)
within_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]; plot_within(within_axes, pfsp_rows, iterations=(3, 5, 7))
ax = fig.add_subplot(gs[2, 3]); plot_aux_error(ax, horizon_errors)
aux_axes = [fig.add_subplot(gs[3, i]) for i in range(4)]
handles, labels = plot_aux(aux_axes, examples)
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.012), fontsize=8)
fig.savefig(OUT / "paper_dense_all_in_one_no_title_7iter.png", bbox_inches="tight")
plt.close(fig)

for path in [OUT / "paper_dense_all_in_one_7iter.png", OUT / "paper_dense_all_in_one_no_title_7iter.png"]:
    print(path.as_posix())
