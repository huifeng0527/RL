---
title: League Ablation Plan - GRU Strategy Encoder and Contrastive Auxiliary Loss
date: 2026-06-19
related_logs:
  - logs/league_ablation_10iter/A_mlp
  - logs/league_ablation_10iter/B_gru
status: planned_and_partially_running
---

# League Ablation Plan: GRU Strategy Encoder + Contrastive Auxiliary Loss

## Background

Recent league-training experiments showed that the core failure mode is not simply poor PPO optimization or lack of short-term motion prediction. The robot collapses when the hand opponent pool becomes diverse because the current opponent strategy is hidden from the robot observation. Similar geometric states can require different evasive responses depending on which hand policy is active, causing opponent-policy aliasing.

Prior observations established:

- Explicit opponent ID stabilizes multi-opponent league training and acts as a diagnostic upper bound.
- Interaction history is more useful than raw motion history for inferring opponent behavior.
- GRU auxiliary prediction improved early league iterations but did not fully solve 3+ opponent strategy aliasing.
- Future-motion auxiliary loss alone can decrease while the policy still collapses, so prediction loss is not sufficient evidence of robust strategy inference.

## Narrative decision

The main paper narrative should avoid framing the method as “LSTM vs GRU.” Instead, the recurrent module should be described as an **interaction-history strategy encoder**. In the implementation and main experiments, this encoder is instantiated as a GRU because the history window is short and a compact recurrent latent is sufficient.

LSTM should not remain in the main experimental story. It can be mentioned only as an older baseline or supplementary comparison if needed. The main ablation should instead show the incremental value of:

1. history encoding,
2. single-step auxiliary dynamics prediction,
3. multi-step trajectory and catch-risk prediction,
4. contrastive strategy representation,
5. explicit opponent identity as a diagnostic upper bound.

## Final A-F league ablation design

All groups should be run in the iterative league setting for 10 iterations.

Common settings:

```text
iterations = 10
first_robot_steps = 3,000,000
robot_steps = 3,000,000
hand_steps = 1,000,000
n_envs = 4
history_length = 16
future_horizon = 8
```

### A. MLP

```text
extractor = mlp
aux = none
opponent_id = false
history_mode = motion
```

Purpose: no-history baseline.

### B. MLP + GRU

```text
extractor = gru
aux = none
opponent_id = false
history_mode = interaction
```

Purpose: isolate the value of recurrent interaction-history encoding.

### C. MLP + GRU + single-step auxiliary prediction

```text
extractor = gru
aux = single
opponent_id = false
history_mode = interaction
```

Purpose: test whether short-horizon hand-displacement prediction improves the recurrent representation.

### D. MLP + GRU + multi-step trajectory / catch-risk auxiliary prediction

```text
extractor = gru
aux = multi_risk
opponent_id = false
history_mode = interaction
strategy_traj_weight = 0.1
strategy_risk_weight = 0.02
contrastive_weight = 0.0
```

Purpose: test whether longer-horizon motion prediction and catch-risk prediction are more useful than single-step prediction.

### E. MLP + GRU + multi-step/risk + contrastive strategy loss

```text
extractor = gru
aux = contrastive
opponent_id = false
history_mode = interaction
strategy_traj_weight = 0.1
strategy_risk_weight = 0.02
strategy_contrastive_weight = 0.05
strategy_contrastive_temperature = 0.1
```

Purpose: final no-ID method. Uses opponent labels only as auxiliary supervision for the strategy embedding, not as policy observation input.

### F. MLP + GRU + opponent ID

```text
extractor = gru
aux = none
opponent_id = true
history_mode = interaction
```

Purpose: diagnostic upper bound. Tests whether explicitly exposing opponent identity resolves the league instability without additional auxiliary objectives.

## Execution schedule

Only two groups should run concurrently to avoid CPU and memory overload.

### Batch 1: running

```text
A_mlp -> logs/league_ablation_10iter/A_mlp
B_gru -> logs/league_ablation_10iter/B_gru
```

Background task IDs at launch:

```text
A_mlp: bowzqd8w2
B_gru: blgwn418i
```

### Batch 2: next

```text
C_gru_single_aux
D_gru_multistep_risk
```

### Batch 3: final

```text
E_gru_multistep_risk_contrastive
F_gru_opponent_id
```

## Implementation decisions already made

- The training entry now supports `--extractor mlp|gru` and `--aux none|single|multi_risk|contrastive`.
- Hand training is fixed to a GRU feature extractor instead of LSTM, so LSTM is removed from the main experimental path.
- The GRU extractor supports optional opponent ID appended after the history segment.
- The contrastive auxiliary loss operates on the GRU strategy embedding.
- Multi-step/risk auxiliary training keeps only:
  - `future_traj_head`
  - `catch_risk_head`
- The previous `min_distance_head` and min-distance auxiliary loss were removed from the GRU strategy auxiliary path.
- Final model saving was changed so `final_model.zip` does not overwrite `best_model.zip`.

## Recommended evaluation metrics

For each group and iteration, compare:

- eval mean reward,
- eval mean episode length,
- ZPD coverage,
- too-close rate,
- too-far rate,
- done reason distribution,
- worst-opponent performance,
- variance across hand opponents,
- pool score if convergence evaluation is enabled,
- auxiliary losses where applicable.

A separate cross-opponent evaluation should be run after training: final robot vs each hand opponent in the pool. This is important because the main hypothesis concerns robustness across a diverse opponent pool, not only average training reward.

## Paper interpretation

The main paper should frame opponent ID as a diagnostic upper bound, not the deployment method. The contrastive no-ID group tests whether implicit strategy representation can reduce the gap to explicit identity conditioning.

Expected interpretation patterns:

- If F >> B, hidden opponent identity is a major bottleneck.
- If E improves over D, contrastive strategy representation helps resolve opponent-policy aliasing.
- If E approaches F, the learned strategy embedding can approximate explicit opponent identity conditioning without exposing identity to the policy.
- If F remains much stronger than E, implicit strategy inference remains the main limitation and should be discussed honestly.
