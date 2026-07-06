from pathlib import Path
import csv
import json
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = Path(r"C:/Users/admin/Desktop/research/RL/logs/league_paper_gru_multistep_aux_20iter")
OUT_DIR = Path(r"C:/Users/admin/Desktop/research/RL/manuscripts/current_league_preview_8iter")
OUT_DIR.mkdir(exist_ok=True)

STATUS_PATH = RUN_DIR / "training_status.jsonl"
EPISODE_PATH = RUN_DIR / "sampled_episodes.jsonl"
LATEST_PATH = RUN_DIR / "training_status_latest.json"


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def as_float(x):
    return float(x) if x is not None else np.nan


status_rows = load_jsonl(STATUS_PATH)
episode_rows = load_jsonl(EPISODE_PATH)
latest = json.loads(LATEST_PATH.read_text(encoding="utf-8")) if LATEST_PATH.exists() else {}
current_iteration = int(latest.get("iteration") or max((r.get("iteration", 0) for r in status_rows), default=0))
current_phase = latest.get("phase", "unknown")
current_progress = float(latest.get("progress") or 0.0)

# Latest status snapshot per phase/iteration.
latest_phase = {}
for r in status_rows:
    if r.get("event") != "phase_progress":
        continue
    phase = r.get("phase")
    iteration = r.get("iteration")
    if phase not in {"robot", "hand"} or iteration is None:
        continue
    key = (int(iteration), phase)
    prev = latest_phase.get(key)
    if prev is None or int(r.get("timesteps") or 0) >= int(prev.get("timesteps") or 0):
        latest_phase[key] = r

robot_summary = []
hand_summary = []
for (iteration, phase), r in sorted(latest_phase.items()):
    row = {
        "iteration": iteration,
        "phase": phase,
        "timesteps": int(r.get("timesteps") or 0),
        "total_timesteps": int(r.get("total_timesteps") or 0),
        "progress": as_float(r.get("progress")),
        "recent_tis_mean": as_float(r.get("recent_tis_mean")),
        "recent_zpd_coverage_mean": as_float(r.get("recent_zpd_coverage_mean")),
        "recent_episode_length_mean": as_float(r.get("recent_episode_length_mean")),
        "recent_episode_count": int(r.get("recent_episode_count") or 0),
        "done_reason_counts": json.dumps(r.get("done_reason_counts", {}), ensure_ascii=False),
    }
    if phase == "robot":
        pfsp = r.get("pfsp") or {}
        probs = pfsp.get("pfsp_probs") or []
        row["pfsp_pool_size"] = int(pfsp.get("pfsp_pool_size") or 0)
        row["dominant_opponent"] = ""
        row["dominant_prob"] = np.nan
        if probs:
            idx = int(np.argmax(probs))
            row["dominant_opponent"] = "scripted" if idx == 0 else f"hand_{idx}"
            row["dominant_prob"] = float(probs[idx])
        robot_summary.append(row)
    else:
        hand_summary.append(row)

# CSV: summary by latest status snapshot of each iteration.
summary_path = OUT_DIR / "league_iteration_status_summary_current.csv"
with summary_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "iteration", "phase", "timesteps", "total_timesteps", "progress",
        "recent_tis_mean", "recent_zpd_coverage_mean", "recent_episode_length_mean",
        "recent_episode_count", "pfsp_pool_size", "dominant_opponent", "dominant_prob",
        "done_reason_counts",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in robot_summary + hand_summary:
        writer.writerow({k: row.get(k, "") for k in fieldnames})

# PFSP snapshots from robot progress logs.
pfsp_snapshots = []
for r in status_rows:
    if r.get("event") != "phase_progress" or r.get("phase") != "robot":
        continue
    pfsp = r.get("pfsp") or {}
    probs = pfsp.get("pfsp_probs") or []
    if not probs:
        continue
    iteration = int(r.get("iteration") or 0)
    timesteps = int(r.get("timesteps") or 0)
    global_progress = (iteration - 1) + timesteps / max(1, int(r.get("total_timesteps") or 1))
    for idx, prob in enumerate(probs):
        opponent = "scripted" if idx == 0 else f"hand_{idx}"
        pfsp_snapshots.append({
            "iteration": iteration,
            "timesteps": timesteps,
            "global_progress": global_progress,
            "opponent_index": idx,
            "opponent": opponent,
            "prob": float(prob),
        })

pfsp_path = OUT_DIR / "pfsp_sampling_snapshots_current.csv"
with pfsp_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["iteration", "timesteps", "global_progress", "opponent_index", "opponent", "prob"])
    writer.writeheader()
    writer.writerows(pfsp_snapshots)

# Episode samples: empirical opponent exposure and outcome proxy.
robot_eps = [r for r in episode_rows if r.get("phase") == "robot" or r.get("training_mode") == "robot"]
by_opp = defaultdict(list)
by_iter = defaultdict(list)
by_iter_opp = defaultdict(list)
for r in robot_eps:
    name = r.get("selected_opponent_name") or "scripted_hand"
    if name == "scripted_hand":
        name = "scripted"
    iteration = int(r.get("iteration") or 0)
    by_opp[name].append(r)
    by_iter[iteration].append(r)
    by_iter_opp[(iteration, name)].append(r)

opp_rows = []
for name, rows in sorted(by_opp.items(), key=lambda kv: (kv[0] != "scripted", kv[0])):
    tis = np.array([as_float(r.get("tis")) for r in rows], dtype=float)
    zpd = np.array([as_float(r.get("zpd_coverage")) for r in rows], dtype=float)
    length = np.array([as_float(r.get("episode_length")) for r in rows], dtype=float)
    done = Counter(r.get("done_reason", "unknown") for r in rows)
    opp_rows.append({
        "opponent": name,
        "sample_count": len(rows),
        "mean_tis": float(np.nanmean(tis)) if len(tis) else np.nan,
        "mean_zpd_coverage": float(np.nanmean(zpd)) if len(zpd) else np.nan,
        "mean_episode_length": float(np.nanmean(length)) if len(length) else np.nan,
        "success_rate_tis_ge_0_4": float(np.nanmean(tis >= 0.4)) if len(tis) else np.nan,
        "robot_caught_count": int(done.get("Robot Caught", 0)),
        "robot_out_count": int(done.get("Robot Out", 0)),
    })

opp_summary_path = OUT_DIR / "sampled_episode_summary_by_opponent_current.csv"
with opp_summary_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "opponent", "sample_count", "mean_tis", "mean_zpd_coverage", "mean_episode_length",
        "success_rate_tis_ge_0_4", "robot_caught_count", "robot_out_count",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(opp_rows)

# Figure 1: latest robot-phase metrics by iteration.
robot_iters = np.array([r["iteration"] for r in robot_summary], dtype=int)
robot_tis = np.array([r["recent_tis_mean"] for r in robot_summary], dtype=float)
robot_zpd = np.array([r["recent_zpd_coverage_mean"] for r in robot_summary], dtype=float)
robot_len = np.array([r["recent_episode_length_mean"] for r in robot_summary], dtype=float)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.0), dpi=180, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(robot_iters, robot_tis, marker="o", linewidth=2.0, label="TIS", color="#4C78A8")
ax1.plot(robot_iters, robot_zpd, marker="s", linewidth=2.0, label="ZPD coverage", color="#72B7B2")
ax1.set_ylabel("Rate")
ax1.set_ylim(0, max(0.7, np.nanmax(robot_zpd) + 0.08 if len(robot_zpd) else 0.7))
ax1.grid(alpha=0.25)
ax1.legend(frameon=False)
ax1.set_title("League training progress from status logs\ncurrent run: GRU + multi-step auxiliary, interim 8-iteration snapshot")
ax2.plot(robot_iters, robot_len, marker="^", linewidth=2.0, color="#54A24B")
ax2.set_ylabel("Episode length")
ax2.set_xlabel("League iteration")
ax2.grid(alpha=0.25)
ax2.set_xticks(robot_iters)
if current_phase == "robot" and current_iteration in robot_iters and current_progress < 0.999:
    ax1.axvline(current_iteration, color="k", linestyle="--", alpha=0.25)
    ax1.text(current_iteration, ax1.get_ylim()[1] * 0.93, "iter 8 partial", ha="right", va="top", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_league_progress_status_current.png")
plt.close(fig)

# Figure 2: PFSP final probability heatmap per iteration.
max_opp = 0
pfsp_by_iter = {}
for r in robot_summary:
    # find last status row for this iteration to recover probs
    iteration = r["iteration"]
    candidates = [x for x in status_rows if x.get("event") == "phase_progress" and x.get("phase") == "robot" and int(x.get("iteration") or 0) == iteration]
    if not candidates:
        continue
    last = max(candidates, key=lambda x: int(x.get("timesteps") or 0))
    probs = (last.get("pfsp") or {}).get("pfsp_probs") or []
    if probs:
        pfsp_by_iter[iteration] = probs
        max_opp = max(max_opp, len(probs))

if max_opp:
    heat = np.full((len(robot_iters), max_opp), np.nan)
    for i, iteration in enumerate(robot_iters):
        probs = pfsp_by_iter.get(int(iteration), [])
        heat[i, :len(probs)] = probs
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    im = ax.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.25, np.nanmax(heat)))
    ax.set_yticks(np.arange(len(robot_iters)))
    ax.set_yticklabels([f"Iter {i}" for i in robot_iters])
    ax.set_xticks(np.arange(max_opp))
    ax.set_xticklabels(["scripted"] + [f"H{i}" for i in range(1, max_opp)], rotation=45, ha="right")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="PFSP sampling probability")
    ax.set_title("PFSP opponent sampling distribution by robot iteration\nlast status snapshot within each iteration")
    ax.set_xlabel("Opponent in pool")
    ax.set_ylabel("Robot training iteration")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_pfsp_sampling_heatmap_current.png")
    plt.close(fig)

    pfsp_by_iteration = defaultdict(list)
    for snap in pfsp_snapshots:
        pfsp_by_iteration[int(snap["iteration"])].append(snap)

    process_iterations = sorted(i for i in pfsp_by_iteration if i >= 2)
    if process_iterations:
        ncols = 2
        nrows = int(np.ceil(len(process_iterations) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, max(3.0, 2.45 * nrows)), dpi=180, sharex=True, sharey=True)
        axes = np.atleast_1d(axes).reshape(-1)
        cmap = plt.get_cmap("tab10")
        legend_handles = {}

        for ax_idx, iteration in enumerate(process_iterations):
            ax = axes[ax_idx]
            rows = pfsp_by_iteration[iteration]
            by_opponent = defaultdict(list)
            for row in rows:
                by_opponent[int(row["opponent_index"])].append(row)

            for opponent_idx in sorted(by_opponent):
                series = sorted(by_opponent[opponent_idx], key=lambda x: x["timesteps"])
                x = np.array([s["timesteps"] / 1_000_000 for s in series], dtype=float)
                y = np.array([s["prob"] for s in series], dtype=float)
                label = "scripted" if opponent_idx == 0 else f"hand {opponent_idx}"
                line, = ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=2.2,
                    linewidth=1.2,
                    color=cmap(opponent_idx % 10),
                    label=label,
                )
                legend_handles.setdefault(label, line)

            ax.set_title(f"Iteration {iteration} (fixed pool size = {max(by_opponent) + 1})", fontsize=10)
            ax.grid(alpha=0.25)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(0, 3.05)
            if ax_idx % ncols == 0:
                ax.set_ylabel("PFSP probability")
            if ax_idx >= (nrows - 1) * ncols:
                ax.set_xlabel("Robot phase timestep (M)")

        for ax in axes[len(process_iterations):]:
            ax.axis("off")

        handles = list(legend_handles.values())
        labels = list(legend_handles.keys())
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), fontsize=8, frameon=False)
        fig.suptitle(
            "Within-iteration PFSP sampling dynamics\n"
            "Each panel keeps opponent pool size fixed; changes reflect PFSP updates, not pool expansion",
            y=0.995,
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.965))
        fig.savefig(OUT_DIR / "fig3_pfsp_sampling_curves_current.png")
        fig.savefig(OUT_DIR / "fig3_pfsp_within_iteration_process_current.png")
        plt.close(fig)

# Figure 4: empirical sampled-episode counts and outcome proxy by opponent.
if opp_rows:
    labels = [r["opponent"].replace("iteration_", "H") for r in opp_rows]
    counts = np.array([r["sample_count"] for r in opp_rows], dtype=float)
    mean_tis = np.array([r["mean_tis"] for r in opp_rows], dtype=float)
    success = np.array([r["success_rate_tis_ge_0_4"] for r in opp_rows], dtype=float)

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.0), dpi=180, sharex=True)
    ax1.bar(x, counts, color="#BAB0AC")
    ax1.set_ylabel("Sampled episodes")
    ax1.set_title("Empirical opponent exposure and outcome proxy\nfrom sampled_episodes.jsonl, one episode every 100 completions")
    ax1.grid(axis="y", alpha=0.25)
    ax2.plot(x, mean_tis, marker="o", label="Mean TIS", color="#4C78A8")
    ax2.plot(x, success, marker="s", label="TIS ≥ 0.4 rate", color="#F58518")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, max(0.75, np.nanmax([np.nanmax(mean_tis), np.nanmax(success)]) + 0.08))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_xlabel("Selected opponent")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_sampled_opponent_outcomes_current.png")
    plt.close(fig)

# Figure 5: by-iteration sampled episode TIS/ZPD summary.
iter_rows = []
for iteration, rows in sorted(by_iter.items()):
    if iteration <= 0:
        continue
    tis = np.array([as_float(r.get("tis")) for r in rows], dtype=float)
    zpd = np.array([as_float(r.get("zpd_coverage")) for r in rows], dtype=float)
    length = np.array([as_float(r.get("episode_length")) for r in rows], dtype=float)
    done = Counter(r.get("done_reason", "unknown") for r in rows)
    iter_rows.append({
        "iteration": iteration,
        "sample_count": len(rows),
        "mean_tis": float(np.nanmean(tis)),
        "mean_zpd_coverage": float(np.nanmean(zpd)),
        "mean_episode_length": float(np.nanmean(length)),
        "success_rate_tis_ge_0_4": float(np.nanmean(tis >= 0.4)),
        "robot_caught_rate": done.get("Robot Caught", 0) / len(rows),
        "robot_out_rate": done.get("Robot Out", 0) / len(rows),
    })

iter_summary_path = OUT_DIR / "sampled_episode_summary_by_iteration_current.csv"
with iter_summary_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["iteration", "sample_count", "mean_tis", "mean_zpd_coverage", "mean_episode_length", "success_rate_tis_ge_0_4", "robot_caught_rate", "robot_out_rate"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(iter_rows)

if iter_rows:
    xs = np.array([r["iteration"] for r in iter_rows], dtype=int)
    tis = np.array([r["mean_tis"] for r in iter_rows], dtype=float)
    zpd = np.array([r["mean_zpd_coverage"] for r in iter_rows], dtype=float)
    success = np.array([r["success_rate_tis_ge_0_4"] for r in iter_rows], dtype=float)
    caught = np.array([r["robot_caught_rate"] for r in iter_rows], dtype=float)
    out = np.array([r["robot_out_rate"] for r in iter_rows], dtype=float)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.0), dpi=180, sharex=True)
    ax1.plot(xs, tis, marker="o", label="Mean TIS", color="#4C78A8")
    ax1.plot(xs, zpd, marker="s", label="Mean ZPD coverage", color="#72B7B2")
    ax1.plot(xs, success, marker="^", label="TIS ≥ 0.4 rate", color="#F58518")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, max(0.75, np.nanmax([np.nanmax(tis), np.nanmax(zpd), np.nanmax(success)]) + 0.08))
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, ncol=3, fontsize=8)
    ax1.set_title("Sampled robot episodes across league iterations")
    ax2.bar(xs - 0.18, caught, width=0.36, label="Robot caught", color="#E45756")
    ax2.bar(xs + 0.18, out, width=0.36, label="Robot out", color="#B279A2")
    ax2.set_ylabel("Done reason rate")
    ax2.set_xlabel("League iteration")
    ax2.set_xticks(xs)
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_sampled_iteration_outcomes_current.png")
    plt.close(fig)

# Draft markdown in Chinese.
def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{float(x):.{digits}f}"

robot_table_lines = ["| Iter | Phase progress | Pool size | TIS ↑ | ZPD coverage ↑ | Episode length | Dominant PFSP opponent |", "|---:|---:|---:|---:|---:|---:|---|"]
for r in robot_summary:
    robot_table_lines.append(
        f"| {r['iteration']} | {fmt(r['progress'], 2)} | {r.get('pfsp_pool_size', '')} | {fmt(r['recent_tis_mean'])} | {fmt(r['recent_zpd_coverage_mean'])} | {fmt(r['recent_episode_length_mean'], 1)} | {r.get('dominant_opponent', '') or 'scripted'} ({fmt(r.get('dominant_prob'), 2) if r.get('dominant_prob') == r.get('dominant_prob') else '--'}) |"
    )

opp_table_lines = ["| Opponent | Sampled episodes | Mean TIS ↑ | Mean ZPD coverage ↑ | TIS ≥ 0.4 rate ↑ | Mean length |", "|---|---:|---:|---:|---:|---:|"]
for r in opp_rows:
    opp_table_lines.append(
        f"| {r['opponent']} | {r['sample_count']} | {fmt(r['mean_tis'])} | {fmt(r['mean_zpd_coverage'])} | {fmt(r['success_rate_tis_ge_0_4'])} | {fmt(r['mean_episode_length'], 1)} |"
    )

notes = []
if current_phase == "robot" and current_iteration == 8 and current_progress < 0.999:
    notes.append("第 8 轮 robot 训练在生成该预览时尚未完全结束，因此第 8 轮指标应视为 partial snapshot。")
notes.append("这些图来自训练状态日志和 sampled episode 日志，不包含额外的 test_env baseline A/B 评估。")
notes.append("严格的 robot_i × hand_j cross-evaluation heatmap 需要单独加载每代 checkpoint 做 pairwise rollout；本预览没有启动该额外评估，以免影响正在运行的 20 轮训练。")
notes.append("sampled_episodes.jsonl 是每 100 个 episode 抽样记录一次，因此 opponent outcome 表反映的是抽样近似，而不是完整 episode 统计。")

md = f"""# League Training 8-Iteration Interim Results — 中文初稿预览

> **数据状态：** 本文档使用 `logs/league_paper_gru_multistep_aux_20iter` 当前日志生成。当前最新状态为 iteration {current_iteration} / phase `{current_phase}` / progress {current_progress:.2f}。这不是最终 20 轮结果，也不包含 `test_env.py` 的 baseline A/B 对比。

## 1. 当前训练设置

本轮实验使用最终拟保留的模型结构：**MLP + GRU interaction-history encoder + multi-step auxiliary prediction**。配置为：

- extractor: `gru`
- auxiliary mode: `multi_risk`
- history mode: `interaction`
- history length: 16
- future horizon: 8
- opponent-id: disabled
- planned league iterations: 20
- robot steps per iteration: 3M
- hand steps per iteration: 0.5M

这组结果适合用于论文中 **iterative league training 内部分析** 的预览，包括：league progress、PFSP 采样分布、按 opponent 分组的 sampled outcome。正式的 cross-iteration validation heatmap 仍建议在训练完成后单独评估所有 robot-hand checkpoint pair。

## 2. League progress：训练过程中的交互质量变化

![League progress](fig1_league_progress_status_current.png)

**图 1. League training progress from status logs.** 该图使用每个 robot phase 的最新 status snapshot，展示 TIS、ZPD coverage 和 episode length 的变化。这里 episode length 只作为辅助解释，不作为主要成功指标。

{chr(10).join(robot_table_lines)}

从当前 8 轮以内的日志看，robot 的 TIS 相比第 1 轮整体提高，第 3 轮达到当前最高值，后续几轮虽有波动但大多维持在约 0.4 附近；ZPD coverage 也明显高于早期阶段。这说明 league training 至少在训练日志窗口内提高了 robot 维持康复交互区间的能力，但最终趋势仍需要等 20 轮完成后确认。

需要谨慎的是，这张图不是固定 evaluation set 上的最终验证结果，而是训练过程 status 的 recent-window 指标。它可以作为“训练过程趋势图”，但正式论文中最好用独立 rollout 的 cross-evaluation 结果来支撑最终结论。

## 3. PFSP sampling dynamics：对手采样分布是否形成课程

![PFSP heatmap](fig2_pfsp_sampling_heatmap_current.png)

**图 2. PFSP opponent sampling heatmap.** 每一行表示某个 robot iteration 内最后一个 status snapshot 的 PFSP 采样概率。列表示当前 opponent pool 中的 scripted hand 和 learned hand generations。

![PFSP process curves](fig3_pfsp_sampling_curves_current.png)

**图 3. Within-iteration PFSP sampling process.** 每个小图对应一个 robot training iteration，横轴是该轮 robot phase 内部的训练步数，纵轴是各 opponent 的 PFSP sampling probability。由于同一个小图内 opponent pool size 固定，曲线变化主要反映 PFSP 根据近期对局表现进行的权重更新，而不是因为进入下一轮后新增 opponent 导致的自然稀释。

当前日志中可以看到，scripted hand 通常保留一部分稳定采样概率，而 learned hand generations 在同一轮训练内部会出现不同程度的上升、下降或保持。这比只比较每轮最后一个 snapshot 更有意义：如果某个 hand 在同一 iteration 内逐渐升高，说明它在当前 robot 训练阶段持续暴露策略弱点；如果逐渐下降，则说明 robot 可能已经在近期窗口中适应了该 opponent。

因此，论文中更适合用图 3 作为 PFSP adaptive curriculum 的主证据，而图 2 的 heatmap 只作为补充概览。

## 4. Sampled opponent outcome：采样对手与训练信号质量

![Sampled opponent outcomes](fig4_sampled_opponent_outcomes_current.png)

**图 4. Empirical opponent exposure and outcome proxy.** 上半部分是 sampled episode 中不同 opponent 被采样到的次数；下半部分是这些 episode 的平均 TIS 和 `TIS ≥ 0.4` 比例。由于 `sampled_episodes.jsonl` 每 100 个 episode 记录一次，该图是抽样近似。

{chr(10).join(opp_table_lines)}

这张图可以和 PFSP 采样分布结合分析：如果某个 opponent 被 PFSP 分配较高概率，同时对应的 mean TIS 或 `TIS ≥ 0.4` 比例较低，说明它可能是当前 robot 的困难样本；如果某个 opponent 的结果已经较稳定但仍保留一定采样概率，则说明 PFSP 的 floor probability 有助于避免遗忘旧对手。

## 5. Sampled iteration outcome：按 league iteration 的 sampled episode 结果

![Sampled iteration outcomes](fig5_sampled_iteration_outcomes_current.png)

**图 5. Sampled robot episodes across league iterations.** 上半部分展示按 iteration 汇总的 mean TIS、mean ZPD coverage 和 `TIS ≥ 0.4` 比例；下半部分展示 sampled episode 中的终止原因比例。

这张图的作用不是替代正式 cross-evaluation，而是帮助解释训练过程中 robot 的失败模式是否发生变化。当前 sampled episodes 显示，早期 iteration 的 TIS 较低，后续 iteration 的 ZPD 指标有明显提升；同时终止原因仍然以 `Robot Caught` 为主，说明 robot 仍面临 aggressive hand 的追赶压力。这一结果支持继续训练到 20 轮后再做完整 validation，而不是仅凭第 8 轮中途结果下最终结论。

## 6. 可写入论文的阶段性叙事

基于当前 8 轮以内的训练日志，league training 已经表现出三个值得保留在论文结果部分的现象。

首先，interaction-quality metrics 在训练过程中呈现上升趋势。与第 1 轮相比，后续 robot phase 的 recent TIS 和 ZPD coverage 明显提高，说明策略并非仅仅延长 episode，而是在更大比例的时间内维持了目标康复距离区间。

其次，PFSP 采样分布呈现非均匀、动态调整的特征。scripted hand 被保留为稳定参考 opponent，同时 learned hand generations 获得不同程度的采样权重。这可以支持本文将 iterative league training 解释为 adaptive curriculum，而不是简单 opponent replay。

第三，按 opponent 分组的 sampled outcome 可以用于解释哪些 hand generation 对当前 robot 更有训练价值。高采样概率与较低 TIS/成功率同时出现时，可以被解释为 PFSP 正在聚焦 robot 的薄弱对手；而旧 opponent 的低概率保留则对应防遗忘机制。

## 7. 当前不能过度声明的内容

{chr(10).join(f'- {n}' for n in notes)}

因此，这版初稿适合用于审阅图表结构和结果叙事。等 20 轮全部完成后，建议补充正式的 robot_i × hand_j cross-evaluation heatmap，并用最终 checkpoint 的独立评估结果替换这里的 training-status trend。
"""

(OUT_DIR / "league_training_8iter_interim_draft_zh.md").write_text(md, encoding="utf-8")

print(f"Output directory: {OUT_DIR}")
for p in sorted(OUT_DIR.iterdir()):
    print(p.name)
