import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.scripts.evaluate_paper_experiments import evaluate_model, model_catalog, rel


catalog = model_catalog()
protocols = ["sluggish", "spasm", "delayed", "noisy", "unseen_rl"]
roots = [
    rel("logs", "dual_iterative_0427_1314"),
    rel("logs", "dual_iterative_0428_2245"),
    rel("logs", "dual_iterative_0426_2334"),
]
rows = []

for root in roots:
    paths = sorted(
        root.glob("iteration_*/robot/robot/best_model.zip"),
        key=lambda p: int(p.parents[2].name.split("_")[-1]),
    )
    for path in paths:
        method = f"{root.name}/{path.parents[2].name}"
        vals = []
        print("checking", method, flush=True)
        for protocol in protocols:
            row = evaluate_model(
                path,
                method,
                protocol,
                seed=3030,
                episodes=15,
                max_steps=100,
                unseen_hand_path=catalog["unseen_hand"],
            )
            rows.append(
                {
                    "method": method,
                    "path": str(path),
                    "protocol": protocol,
                    "zpd": row.zpd_coverage,
                    "reward": row.reward_mean,
                    "length": row.episode_length_mean,
                    "catch": row.catch_rate,
                }
            )
            vals.append(row.zpd_coverage)
        print(method, "mean", sum(vals) / len(vals), "worst", min(vals), flush=True)

df = pd.DataFrame(rows)
df.to_csv(rel(".codex_tmp", "league_sweep.csv"), index=False)
rank = (
    df.groupby(["method", "path"])
    .agg(
        mean_zpd=("zpd", "mean"),
        worst_zpd=("zpd", "min"),
        mean_reward=("reward", "mean"),
        mean_len=("length", "mean"),
    )
    .reset_index()
    .sort_values(["worst_zpd", "mean_zpd"], ascending=False)
)
rank.to_csv(rel(".codex_tmp", "league_sweep_rank.csv"), index=False)
print(rank.head(10).to_string(index=False))
