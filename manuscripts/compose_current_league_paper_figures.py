from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from stable_baselines3 import PPO

from src.custom_env import RehabilitationEnv
from src.observation_schema import INTERACTION_HISTORY_CHANNELS, HISTORY_CHANNELS, model_obs_dim, obs_dim


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


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def model_path(run_dir: Path, iteration: int, agent: str) -> Path | None:
    base = run_dir / f"iteration_{iteration}" / agent / agent
    best = base / "best_model.zip"
    final = base / "final_model.zip"
    if best.exists():
        return best
    if final.exists():
        return final
    return None


def complete_generations(run_dir: Path, max_iterations: int) -> list[int]:
    gens = []
    for i in range(1, max_iterations + 1):
        if model_path(run_dir, i, "robot") is not None and model_path(run_dir, i, "hand") is not None:
            gens.append(i)
    return gens


def infer_history_mode(robot_model: PPO, history_length: int) -> str:
    expected_dim = model_obs_dim(robot_model)
    interaction_dim = obs_dim(history_length, 0, INTERACTION_HISTORY_CHANNELS)
    motion_dim = obs_dim(history_length, 0, HISTORY_CHANNELS)
    if expected_dim == interaction_dim:
        return "interaction"
    if expected_dim == motion_dim:
        return "motion"
    return "interaction" if expected_dim > motion_dim else "motion"


def adapt_obs(obs: np.ndarray, expected_dim: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[0] == expected_dim:
        return obs
    if obs.shape[0] < expected_dim:
        pad = np.zeros(expected_dim - obs.shape[0], dtype=np.float32)
        return np.concatenate([obs, pad]).astype(np.float32)
    return obs[:expected_dim].astype(np.float32)


def evaluate_pair(robot_path: Path, hand_path: Path | None, episodes: int, max_steps: int, history_length: int) -> dict:
    robot_model = PPO.load(str(robot_path), verbose=0)
    hand_model = None
    if hand_path is not None:
        hand_model = PPO.load(str(hand_path), custom_objects={"learning_rate": 0.0, "optimizer_class": None}, verbose=0)

    history_mode = infer_history_mode(robot_model, history_length)
    env = RehabilitationEnv(
        training_mode="robot",
        hand_model=hand_model,
        history_length=history_length,
        history_mode=history_mode,
    )
    env.random_noise = False
    env.max_steps = max_steps
    z_min = float(env.zpd_min)
    z_max = float(env.zpd_max)
    expected_dim = model_obs_dim(robot_model)

    values = {"tis": [], "zpd": [], "length": [], "too_close": [], "too_far": [], "avg_dist": []}
    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        distances = []
        while not (terminated or truncated):
            action, _ = robot_model.predict(adapt_obs(obs, expected_dim), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            distances.append(float(info["dist"]))
        if not distances:
            continue
        distances_arr = np.asarray(distances, dtype=float)
        in_zpd = (distances_arr >= z_min) & (distances_arr <= z_max)
        values["tis"].append(float(np.sum(in_zpd) / max_steps))
        values["zpd"].append(float(np.mean(in_zpd)))
        values["length"].append(float(len(distances_arr)))
        values["too_close"].append(float(np.mean(distances_arr < z_min)))
        values["too_far"].append(float(np.mean(distances_arr > z_max)))
        values["avg_dist"].append(float(np.mean(distances_arr)))
    env.close()

    return {
        "tis_mean": float(np.mean(values["tis"])),
        "tis_std": float(np.std(values["tis"])),
        "zpd_coverage_mean": float(np.mean(values["zpd"])),
        "episode_length_mean": float(np.mean(values["length"])),
        "too_close_rate_mean": float(np.mean(values["too_close"])),
        "too_far_rate_mean": float(np.mean(values["too_far"])),
        "avg_distance_mean": float(np.mean(values["avg_dist"])),
        "zpd_min": z_min,
        "zpd_max": z_max,
    }


def build_cross_eval(run_dir: Path, out_dir: Path, gens: list[int], episodes: int, max_steps: int, history_length: int, refresh: bool) -> dict:
    cache = out_dir / f"cross_eval_{len(gens)}gen.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))

    tis = np.zeros((len(gens), len(gens)), dtype=float)
    zpd = np.zeros_like(tis)
    results = []
    for i, r_gen in enumerate(gens):
        row = []
        robot_zip = model_path(run_dir, r_gen, "robot")
        for j, h_gen in enumerate(gens):
            hand_zip = model_path(run_dir, h_gen, "hand")
            print(f"Evaluating R{r_gen} vs H{h_gen}")
            metrics = evaluate_pair(robot_zip, hand_zip, episodes, max_steps, history_length)
            metrics.update({
                "robot": f"R{r_gen}",
                "hand": f"H{h_gen}",
                "robot_generation": r_gen,
                "hand_generation": h_gen,
                "robot_path": str(robot_zip),
                "hand_path": str(hand_zip),
                "episodes": episodes,
                "max_steps": max_steps,
            })
            row.append(metrics)
            tis[i, j] = metrics["tis_mean"]
            zpd[i, j] = metrics["zpd_coverage_mean"]
            print(f"  TIS={metrics['tis_mean']:.3f}, ZPD={metrics['zpd_coverage_mean']:.3f}")
        results.append(row)

    payload = {
        "run_dir": str(run_dir),
        "generations": gens,
        "episodes": episodes,
        "max_steps": max_steps,
        "results": results,
        "tis_matrix": tis.tolist(),
        "zpd_matrix": zpd.tolist(),
    }
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (out_dir / f"cross_eval_{len(gens)}gen_tis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Robot/Hand"] + [f"H{i}" for i in gens])
        for i, r_gen in enumerate(gens):
            writer.writerow([f"R{r_gen}"] + [f"{tis[i, j]:.6f}" for j in range(len(gens))])
    return payload


def build_scripted_eval(run_dir: Path, out_dir: Path, gens: list[int], episodes: int, max_steps: int, history_length: int, refresh: bool) -> list[dict]:
    cache = out_dir / "scripted_eval.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    rows = []
    for gen in gens:
        robot_zip = model_path(run_dir, gen, "robot")
        print(f"Evaluating R{gen} vs scripted hand")
        metrics = evaluate_pair(robot_zip, None, episodes, max_steps, history_length)
        metrics.update({"robot": f"R{gen}", "robot_generation": gen, "hand": "Scripted"})
        rows.append(metrics)
    cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def build_pfsp_rows(run_dir: Path, out_dir: Path, max_iterations: int) -> list[dict]:
    status_rows = load_jsonl(run_dir / "training_status.jsonl")
    pfsp_rows = []
    summary_latest = {}
    for row in status_rows:
        if row.get("event") != "phase_progress" or row.get("phase") != "robot":
            continue
        try:
            iteration = int(row.get("iteration"))
        except Exception:
            continue
        if iteration < 1 or iteration > max_iterations:
            continue
        timestep = int(row.get("timesteps") or 0)
        if timestep >= int(summary_latest.get(iteration, {}).get("timesteps") or -1):
            summary_latest[iteration] = row
        probs = (row.get("pfsp") or {}).get("pfsp_probs") or []
        total = max(1, int(row.get("total_timesteps") or 1))
        pool_size = len(probs)
        for index, prob in enumerate(probs):
            pfsp_rows.append({
                "iteration": iteration,
                "timesteps": timestep,
                "iteration_progress": timestep / total,
                "opponent_index": index,
                "opponent": "Scr" if index == 0 else f"H{index}",
                "probability": float(prob),
                "pool_size": pool_size,
                "uniform_probability": 1.0 / pool_size if pool_size else 0.0,
            })

    if pfsp_rows:
        with (out_dir / "pfsp_sampling_snapshots.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pfsp_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pfsp_rows)
    if summary_latest:
        keys = [
            "iteration", "timesteps", "total_timesteps", "recent_tis_mean", "recent_zpd_coverage_mean",
            "recent_episode_length_mean", "recent_too_close_rate_mean", "recent_too_far_rate_mean",
        ]
        with (out_dir / "league_training_status_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for iteration in sorted(summary_latest):
                writer.writerow(summary_latest[iteration])
    return pfsp_rows


def cvar(values: np.ndarray, alpha: float = 0.2) -> float:
    k = max(1, int(math.ceil(len(values) * alpha)))
    return float(np.mean(np.sort(values)[:k]))


def polish_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)


def panel_label(ax, label: str):
    ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def plot_cross_heatmap(ax, tis: np.ndarray, gens: list[int]):
    vmin = max(0.0, float(np.nanmin(tis)) - 0.02)
    vmax = min(0.4, float(np.nanmax(tis)) + 0.03)
    sns.heatmap(
        tis,
        ax=ax,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        annot=False,
        linewidths=0.25,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "TIZ", "shrink": 0.78, "pad": 0.01},
        square=True,
    )
    ax.set_xticklabels([f"H{i}" for i in gens], rotation=45, ha="right")
    ax.set_yticklabels([f"R{i}" for i in gens], rotation=0)
    ax.set_xlabel("Hand generation")
    ax.set_ylabel("Robot generation")
    ax.set_title("Cross-iteration validation")
    panel_label(ax, "a")


def plot_robustness(ax, tis: np.ndarray, gens: list[int]):
    mean = tis.mean(axis=1)
    worst = tis.min(axis=1)
    ax.plot(gens, mean, "-o", lw=1.8, ms=3.5, label="Mean")
    ax.plot(gens, worst, "-s", lw=1.8, ms=3.5, label="Worst hand")
    ax.set_ylim(0, 0.4)
    ax.set_xticks(gens)
    ax.set_xlabel("Robot generation")
    ax.set_ylabel("TIZ")
    ax.set_title("Robustness across learned hands")
    polish_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "b")


def plot_frontier(ax, tis: np.ndarray, gens: list[int]):
    mean = tis.mean(axis=1)
    worst = tis.min(axis=1)
    sc = ax.scatter(mean, worst, c=gens, cmap="viridis", s=38, edgecolor="black", linewidth=0.35)
    ax.plot(mean, worst, color="0.55", lw=0.8, alpha=0.7)
    for gen, x, y in zip(gens, mean, worst):
        ax.text(x + 0.004, y + 0.004, f"R{gen}", fontsize=6.2)
    ax.set_xlabel("Mean TIZ")
    ax.set_ylabel("Worst-hand TIZ")
    ax.set_title("Robustness frontier")
    polish_axis(ax)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Robot generation")
    panel_label(ax, "c")


def plot_final_failure(cross: dict, ax):
    gens = cross["generations"]
    final_results = cross["results"][-1]
    too_close = np.array([float(item["too_close_rate_mean"]) for item in final_results], dtype=float)
    too_far = np.array([float(item["too_far_rate_mean"]) for item in final_results], dtype=float)
    zpd = np.array([float(item["zpd_coverage_mean"]) for item in final_results], dtype=float)
    xs = np.arange(len(gens))
    ax.bar(xs, too_close, color="#d62728", alpha=0.82, label="Too close")
    ax.bar(xs, too_far, bottom=too_close, color="#ff7f0e", alpha=0.82, label="Too far")
    ax.plot(xs, zpd, "o-", color="#1f77b4", lw=1.7, ms=3.2, label="ZPD coverage")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"H{i}" for i in gens], rotation=45, ha="right")
    ax.set_xlabel("Test hand")
    ax.set_ylabel("Episode fraction")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"R{gens[-1]} failure decomposition")
    polish_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "d")


def plot_pfsp(axs, pfsp_rows: list[dict], requested: list[int]):
    if not pfsp_rows:
        for ax in axs:
            ax.axis("off")
        return
    available = sorted({int(r["iteration"]) for r in pfsp_rows})
    iterations = [i for i in requested if i in available]
    if len(iterations) < len(axs):
        iterations = [available[0], available[-1]][:len(axs)]
    panel_names = ["c", ""]
    panel_titles = ["Early PFSP sampling", "Late PFSP sampling"]
    for ax_idx, (ax, iteration) in enumerate(zip(axs, iterations)):
        rows = [r for r in pfsp_rows if int(r["iteration"]) == iteration]
        by_opp = {}
        for row in rows:
            by_opp.setdefault(int(row["opponent_index"]), []).append(row)

        learned_opps = [opp for opp in sorted(by_opp) if opp != 0]
        final_probs = {}
        for opp in learned_opps:
            series = sorted(by_opp[opp], key=lambda x: float(x["iteration_progress"]))
            final_probs[opp] = float(series[-1]["probability"])
        if len(learned_opps) <= 5:
            keep = set(learned_opps)
        else:
            keep = set(sorted(learned_opps, key=lambda opp: final_probs[opp], reverse=True)[:5])
            keep.add(min(learned_opps))
            keep.add(max(learned_opps))

        plotted = []
        for opp in sorted(keep):
            series = sorted(by_opp[opp], key=lambda x: float(x["iteration_progress"]))
            xs = [float(s["iteration_progress"]) for s in series]
            ys = [float(s["probability"]) for s in series]
            plotted.extend(ys)
            ax.plot(xs, ys, lw=1.5, label=f"H{opp}")
        pool_size = max(int(r["pool_size"]) for r in rows) if rows else 1
        learned_uniform = 0.8 / max(pool_size - 1, 1)
        ax.axhline(learned_uniform, color="black", ls=":", lw=0.9, label="Learned uniform")
        if plotted:
            ymin = max(0.0, min(plotted + [learned_uniform]) - 0.015)
            ymax = max(plotted + [learned_uniform]) + 0.015
            if ymax - ymin < 0.05:
                center = 0.5 * (ymin + ymax)
                ymin = max(0.0, center - 0.035)
                ymax = center + 0.035
            ax.set_ylim(ymin, ymax)
        ax.set_title(f"{panel_titles[ax_idx]} (iter. {iteration})")
        ax.set_xlabel("Robot-phase progress")
        polish_axis(ax)
        ax.legend(frameon=False, ncol=2, loc="best")
        if panel_names[ax_idx]:
            panel_label(ax, panel_names[ax_idx])
    axs[0].set_ylabel("Sampling probability")


def plot_sorted_matrix(out_dir: Path, tis: np.ndarray, gens: list[int]) -> dict:
    robot_strength = tis.mean(axis=1)
    hand_easiness = tis.mean(axis=0)
    robot_order = np.argsort(robot_strength)
    hand_order = np.argsort(-hand_easiness)
    sorted_tis = tis[np.ix_(robot_order, hand_order)]
    summary = {
        "robot_strength": {f"R{gens[i]}": float(robot_strength[i]) for i in range(len(gens))},
        "hand_easiness": {f"H{gens[j]}": float(hand_easiness[j]) for j in range(len(gens))},
        "robot_order_weak_to_strong": [f"R{gens[i]}" for i in robot_order],
        "hand_order_easy_to_hard": [f"H{gens[j]}" for j in hand_order],
    }
    (out_dir / "sorted_strength_difficulty_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=300, constrained_layout=True)
    labels_r = [f"R{i}" for i in gens]
    labels_h = [f"H{i}" for i in gens]
    sorted_r = [f"R{gens[i]}\n{robot_strength[i]:.2f}" for i in robot_order]
    sorted_h = [f"H{gens[j]}\n{hand_easiness[j]:.2f}" for j in hand_order]
    for ax, matrix, row_labels, col_labels, title in [
        (axes[0], tis, labels_r, labels_h, "Generation order"),
        (axes[1], sorted_tis, sorted_r, sorted_h, "Sorted strength/difficulty"),
    ]:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=max(0.65, float(np.nanmax(tis)) + 0.05),
            annot=True,
            fmt=".2f",
            linewidths=0.3,
            cbar=True,
            cbar_kws={"label": "TIZ", "shrink": 0.82, "pad": 0.01},
            annot_kws={"fontsize": 6.3},
        )
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticklabels(row_labels, rotation=0)
        ax.set_xlabel("Hand" if ax is axes[0] else "Hand, easy → hard")
        ax.set_ylabel("Robot" if ax is axes[0] else "Robot, weak → strong")
        ax.set_title(title)
        polish_axis(ax)
    fig.savefig(out_dir / "empirical_tis_matrix_sorted_strength_difficulty.png", bbox_inches="tight")
    plt.close(fig)
    return summary


def compose_overview(out_dir: Path, cross: dict, pfsp_rows: list[dict], no_title: bool):
    tis = np.asarray(cross["tis_matrix"], dtype=float)
    gens = cross["generations"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.6, 4.1 if no_title else 4.3),
        dpi=300,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )
    plot_cross_heatmap(axes[0], tis, gens)
    plot_robustness(axes[1], tis, gens)
    if not no_title:
        fig.suptitle("League training validation", fontsize=13, y=1.04)
    name = "paper_league_overview_no_title.png" if no_title else "paper_league_overview.png"
    fig.savefig(out_dir / name, bbox_inches="tight")
    plt.close(fig)


def write_summary(out_dir: Path, cross: dict, scripted_rows: list[dict], sorted_summary: dict):
    tis = np.asarray(cross["tis_matrix"], dtype=float)
    gens = cross["generations"]
    mean = tis.mean(axis=1)
    worst = tis.min(axis=1)
    cvar20 = np.array([cvar(row, 0.2) for row in tis], dtype=float)
    rows = []
    for i, gen in enumerate(gens):
        rows.append({
            "robot": f"R{gen}",
            "generation": gen,
            "learned_hand_mean_tis": float(mean[i]),
            "learned_hand_worst_tis": float(worst[i]),
            "learned_hand_cvar20_tis": float(cvar20[i]),
            "scripted_tis": next((float(r["tis_mean"]) for r in scripted_rows if int(r["robot_generation"]) == gen), np.nan),
        })
    with (out_dir / "league_robustness_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "generations": gens,
        "first_robot": rows[0],
        "final_robot": rows[-1],
        "mean_gain_final_minus_first": float(mean[-1] - mean[0]),
        "worst_gain_final_minus_first": float(worst[-1] - worst[0]),
        "best_mean_robot": rows[int(np.argmax(mean))],
        "best_worst_robot": rows[int(np.argmax(worst))],
        "sorted_matrix": sorted_summary,
    }
    (out_dir / "league_result_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Compose paper-ready league figures for the current simulation run")
    parser.add_argument("--run_dir", type=Path, default=Path("logs/league_zpd35_55_noid_warm_entropy_10iter_r5m_h1m_gru_noaux"))
    parser.add_argument("--out_dir", type=Path, default=Path("manuscripts/current_league_zpd35_55_noid_warm_entropy_10iter"))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--history_length", type=int, default=16)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gens = complete_generations(args.run_dir, args.iterations)
    if not gens:
        raise RuntimeError(f"No complete robot/hand generations found in {args.run_dir}")
    print(f"Using complete generations: {gens}")

    cross = build_cross_eval(args.run_dir, args.out_dir, gens, args.episodes, args.max_steps, args.history_length, args.refresh)
    scripted_rows = build_scripted_eval(args.run_dir, args.out_dir, gens, args.episodes, args.max_steps, args.history_length, args.refresh)
    pfsp_rows = build_pfsp_rows(args.run_dir, args.out_dir, args.iterations)
    sorted_summary = plot_sorted_matrix(args.out_dir, np.asarray(cross["tis_matrix"], dtype=float), gens)
    compose_overview(args.out_dir, cross, pfsp_rows, no_title=False)
    compose_overview(args.out_dir, cross, pfsp_rows, no_title=True)
    summary = write_summary(args.out_dir, cross, scripted_rows, sorted_summary)

    print((args.out_dir / "paper_league_overview.png").as_posix())
    print((args.out_dir / "paper_league_overview_no_title.png").as_posix())
    print((args.out_dir / "empirical_tis_matrix_sorted_strength_difficulty.png").as_posix())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
