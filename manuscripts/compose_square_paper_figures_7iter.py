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


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_cross_heatmap(ax, tis_matrix):
    sns.heatmap(
        tis_matrix,
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=max(0.8, float(np.nanmax(tis_matrix))),
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.45,
        cbar_kws={"label": "Mean TIS", "shrink": 0.78},
        annot_kws={"fontsize": 7},
    )
    n = tis_matrix.shape[0]
    ax.set_xticklabels([f"H{i}" for i in range(1, n + 1)], rotation=0, fontsize=8)
    ax.set_yticklabels([f"R{i}" for i in range(1, n + 1)], rotation=0, fontsize=8)
    ax.set_xlabel("Hand generation", fontsize=9)
    ax.set_ylabel("Robot generation", fontsize=9)
    ax.set_title("(a) Cross-iteration validation", loc="left", fontweight="bold", fontsize=10)


def plot_cross_summary(ax, tis_matrix):
    x = np.arange(1, tis_matrix.shape[0] + 1)
    mean = tis_matrix.mean(axis=1)
    worst = tis_matrix.min(axis=1)
    ax.plot(x, mean, "-o", lw=2.0, ms=4.2, color="#1f77b4", label="Mean across hands")
    ax.plot(x, worst, "-s", lw=2.0, ms=4.2, color="#d62728", label="Worst hand")
    ax.set_xlabel("Robot generation", fontsize=9)
    ax.set_ylabel("TIS", fontsize=9)
    ax.set_xticks(x)
    ax.set_ylim(0, max(0.8, float(np.max(mean)) + 0.08))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("(b) Robustness summary", loc="left", fontweight="bold", fontsize=10)


def plot_pfsp_heatmap(ax, status_rows):
    iters = []
    probs_by_iter = []
    for row in status_rows:
        iteration = int(row["iteration"])
        pool_size = int(row["pool_size"])
        if pool_size <= 0:
            continue
        iters.append(iteration)
    max_pool = max(int(r["pool_size"]) for r in status_rows)
    heat = np.full((len(iters), max_pool), np.nan)
    for i, iteration in enumerate(iters):
        row = next(r for r in status_rows if int(r["iteration"]) == iteration)
        # Reconstruct final snapshot ratios from the snapshots table for this iteration.
        rows = [s for s in PFSP_ROWS if int(s["iteration"]) == iteration]
        final_t = max(int(s["timesteps"]) for s in rows)
        final_rows = [s for s in rows if int(s["timesteps"]) == final_t]
        for s in final_rows:
            j = int(s["opponent_index"])
            heat[i, j] = float(s["preference_ratio"])
    im = ax.imshow(np.ma.masked_invalid(heat), aspect="auto", cmap="RdBu_r", vmin=0, vmax=max(2.2, float(np.nanmax(heat))))
    ax.set_yticks(np.arange(len(iters)))
    ax.set_yticklabels([f"Iter {i}" for i in iters], fontsize=8)
    ax.set_xticks(np.arange(max_pool))
    ax.set_xticklabels(["Scripted"] + [f"H{i}" for i in range(1, max_pool)], rotation=35, ha="right", fontsize=8)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", fontsize=6.5)
    ax.set_xlabel("Opponent", fontsize=9)
    ax.set_ylabel("Robot iteration", fontsize=9)
    ax.set_title("(c) PFSP preference vs. uniform", loc="left", fontweight="bold", fontsize=10)
    return im


def plot_pfsp_age(ax, pfsp_rows):
    age_groups = {}
    scripted = []
    for row in pfsp_rows:
        ratio = float(row["preference_ratio"])
        if int(row["opponent_index"]) == 0:
            scripted.append(ratio)
        elif row["opponent_age"] != "":
            age_groups.setdefault(int(row["opponent_age"]), []).append(ratio)
    ages = sorted(age_groups)
    means = []
    sems = []
    for age in ages:
        values = np.asarray(age_groups[age], dtype=float)
        means.append(values.mean())
        sems.append(values.std() / np.sqrt(max(1, len(values))))
    ax.errorbar(ages, means, yerr=sems, marker="o", lw=2.0, capsize=3, color="#1f77b4", label="Learned hands")
    if scripted:
        ax.axhline(np.mean(scripted), color="#ff7f0e", ls="--", lw=1.6, label="Scripted mean")
    ax.axhline(1.0, color="black", ls=":", lw=1.1, label="Uniform")
    ax.set_xlabel("Opponent age", fontsize=9)
    ax.set_ylabel("Preference ratio", fontsize=9)
    ax.set_xticks(ages)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5)
    ax.tick_params(labelsize=8)
    ax.set_title("(d) Age-aligned PFSP behavior", loc="left", fontweight="bold", fontsize=10)


def plot_pfsp_curves(ax, pfsp_rows):
    # Compact: show representative learned opponents from iter 7, plus iter-3 hand transition as inset-like overlay.
    for iteration, alpha in [(3, 0.55), (7, 0.95)]:
        rows = [r for r in pfsp_rows if int(r["iteration"]) == iteration]
        if not rows:
            continue
        by_opp = {}
        for row in rows:
            by_opp.setdefault(int(row["opponent_index"]), []).append(row)
        for opp in sorted(by_opp):
            if iteration == 7 and opp not in {0, 3, 4, 5, 6}:
                continue
            if iteration == 3 and opp not in {1, 2}:
                continue
            series = sorted(by_opp[opp], key=lambda x: float(x["iteration_progress"]))
            x = [float(s["iteration_progress"]) for s in series]
            y = [float(s["preference_ratio"]) for s in series]
            if iteration == 3:
                label = f"Iter3-H{opp}" if opp else "Iter3-scripted"
                ls = "--"
            else:
                label = "Scripted" if opp == 0 else f"Iter7-H{opp}"
                ls = "-"
            ax.plot(x, y, lw=1.6, alpha=alpha, ls=ls, label=label)
    ax.axhline(1.0, color="black", ls=":", lw=1.1)
    ax.set_xlabel("Progress within robot phase", fontsize=9)
    ax.set_ylabel("Preference ratio", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.tick_params(labelsize=8)
    ax.set_title("(e) Within-iteration PFSP updates", loc="left", fontweight="bold", fontsize=10)


def plot_aux_examples(axs, examples):
    for idx, (ax, ex) in enumerate(zip(axs, examples[:4]), start=1):
        past_hand, past_robot, current_robot, true_future, robot_future, pred_future, target_future = aux.relative_paths(ex)
        ax.plot(past_hand[:, 0], past_hand[:, 1], "o-", color="#8c8c8c", lw=1.0, ms=2.2, label="past hand")
        ax.scatter([0], [0], marker="*", s=55, color="black", zorder=5, label="current")
        ax.plot(target_future[:, 0], target_future[:, 1], "o-", color="#1f77b4", lw=1.7, ms=2.8, label="actual")
        ax.plot(pred_future[:, 0], pred_future[:, 1], "s--", color="#ff7f0e", lw=1.7, ms=2.8, label="predicted")
        ax.set_title(f"Ex. {idx}, MSE={ex['trajectory_mse']:.2f}", fontsize=8)
        ax.set_xlabel("rel. x", fontsize=7)
        ax.set_ylabel("rel. y", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")
    handles, labels = axs[0].get_legend_handles_labels()
    return handles, labels


def plot_aux_error(ax, horizon_errors):
    mean = np.mean(horizon_errors, axis=0)
    stderr = np.std(horizon_errors, axis=0) / np.sqrt(max(1, len(horizon_errors)))
    steps = np.arange(1, aux.FUTURE_HORIZON + 1)
    ax.plot(steps, mean, "o-", color="#9467bd", lw=2, ms=4)
    ax.fill_between(steps, mean - stderr, mean + stderr, color="#9467bd", alpha=0.2)
    ax.set_xlabel("Future step", fontsize=9)
    ax.set_ylabel("Trajectory MSE", fontsize=9)
    ax.set_xticks(steps)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.set_title("(f) Prediction error by horizon", loc="left", fontweight="bold", fontsize=10)


CROSS = load_json(OUT / "cross_iter_validation_7iter.json")
TIS = np.asarray(CROSS["tis_matrix"], dtype=float)
STATUS = load_csv(OUT / "league_training_status_7iter.csv")
PFSP_ROWS = load_csv(OUT / "pfsp_sampling_snapshots_7iter.csv")

# Re-collect aux examples so the all-in-one figure is native-vector, not pasted image.
robot_model = PPO.load(str(aux.ROBOT_PATH), verbose=0)
hand_model = PPO.load(str(aux.HAND_PATH), custom_objects={"learning_rate": 0.0, "optimizer_class": None}, verbose=0)
AUX_EXAMPLES, HORIZON_ERRORS = aux.collect_examples(robot_model, hand_model)

# Two-figure version.
fig = plt.figure(figsize=(10.2, 10.2), dpi=300)
gs = fig.add_gridspec(3, 2, height_ratios=[1.18, 1.0, 0.82], hspace=0.42, wspace=0.34)
ax = fig.add_subplot(gs[0:2, 0])
plot_cross_heatmap(ax, TIS)
ax = fig.add_subplot(gs[0, 1])
plot_cross_summary(ax, TIS)
ax = fig.add_subplot(gs[1, 1])
plot_pfsp_age(ax, PFSP_ROWS)
ax = fig.add_subplot(gs[2, 0])
im = plot_pfsp_heatmap(ax, STATUS)
ax = fig.add_subplot(gs[2, 1])
plot_pfsp_curves(ax, PFSP_ROWS)
fig.suptitle("League training validation and PFSP curriculum analysis", fontsize=14, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(OUT / "paper_bigfig_league_square_7iter.png", bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(10.2, 10.2), dpi=300)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.72], hspace=0.38, wspace=0.28)
aux_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
handles, labels = plot_aux_examples(aux_axes, AUX_EXAMPLES)
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
ax = fig.add_subplot(gs[2, :])
plot_aux_error(ax, HORIZON_ERRORS)
fig.suptitle("Auxiliary future-motion prediction behavior", fontsize=14, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "paper_bigfig_aux_square_7iter.png", bbox_inches="tight")
plt.close(fig)

# Single all-in-one figure.
fig = plt.figure(figsize=(12.0, 12.0), dpi=300)
gs = fig.add_gridspec(4, 4, hspace=0.54, wspace=0.48)
ax = fig.add_subplot(gs[0:2, 0:2])
plot_cross_heatmap(ax, TIS)
ax = fig.add_subplot(gs[0, 2:4])
plot_cross_summary(ax, TIS)
ax = fig.add_subplot(gs[1, 2])
plot_pfsp_age(ax, PFSP_ROWS)
ax = fig.add_subplot(gs[1, 3])
plot_pfsp_curves(ax, PFSP_ROWS)
ax = fig.add_subplot(gs[2, 0:2])
plot_pfsp_heatmap(ax, STATUS)
aux_axes = [fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[2, 3]), fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
handles, labels = plot_aux_examples(aux_axes, AUX_EXAMPLES)
ax = fig.add_subplot(gs[3, 2:4])
plot_aux_error(ax, HORIZON_ERRORS)
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Simulation analysis: league validation, PFSP curriculum, and auxiliary prediction", fontsize=14, y=0.995)
fig.tight_layout(rect=(0, 0.025, 1, 0.975))
fig.savefig(OUT / "paper_all_in_one_square_7iter.png", bbox_inches="tight")
plt.close(fig)

for path in [
    OUT / "paper_bigfig_league_square_7iter.png",
    OUT / "paper_bigfig_aux_square_7iter.png",
    OUT / "paper_all_in_one_square_7iter.png",
]:
    print(path.as_posix())
