---
title: Strategy GRU AUX Training Observations
date: 2026-06-17
related_logs:
  - logs/dual_iterative_0617_1152
  - logs/dual_iterative_0617_1540
---

# Strategy GRU AUX Training Observations

## Background

After the opponent-id diagnostic experiment confirmed that multi-opponent pool failure is primarily due to hidden opponent identity / strategy aliasing, we implemented a non-ID alternative: `StrategyGRUAuxExtractor` with a GRU strategy encoder and auxiliary prediction heads (future trajectory, catch risk, min distance). The goal was to let the robot implicitly infer opponent strategy from interaction history without explicit opponent labels.

## Experiment 1 (0617_1152): strategy_gru_aux with 16-step hand movement history

**Config** (run_config.json):
```json
{
  "extractor_name": "strategy_gru_aux",
  "history_length": 16,
  "history_mode": "motion",
  "future_horizon": 8,
  "include_opponent_id": false
}
```

**Result**: Worse than baseline aux_lstm even on single scripted hand.
- iteration_1 eval reward tail5 ≈ -33.34, ep_len tail5 ≈ 48.16
- Baseline aux_lstm iteration_1: eval reward tail5 ≈ -4.07, ep_len tail5 ≈ 78.17

**Diagnosis**: This run used `history_length=16` with `history_mode=motion` (hand movement only, 2 channels). The GRU encoder only saw 32-dim hand movement history—identical input to the old LSTM but with a different architecture. The auxiliary loss optimized successfully (loss dropped), but the learned representation did not help the policy. Multi-opponent iterations showed the same pattern as the no-id baseline: scripted hand OK, learned hands collapsed.

**Root cause**: The input was insufficient. Only seeing hand deltas (not robot-hand interaction) and only 16 steps was not enough to distinguish opponent strategies.

## Experiment 2 (0617_1540): strategy_gru_aux with 32-step interaction history

**Config** (run_config.json):
```json
{
  "extractor_name": "strategy_gru_aux",
  "history_length": 32,
  "history_mode": "interaction",
  "future_horizon": 8,
  "include_opponent_id": false
}
```

Each interaction step = 8 channels:
```text
[relative_pos_x, relative_pos_y, current_distance, distance_delta,
 robot_actual_move_x, robot_actual_move_y, hand_move_x, hand_move_y]
```

Final robot obs_dim = 12 + 32 * 8 = 268

**Result**: Complete failure, even on single scripted hand.
- iteration_1 at 2.29M/3M steps: episode_length ≈ 16, 98% Robot Caught
- Baseline aux_lstm iteration_1: episode_length tail ≈ 78

**Diagnosis**: Three interacting problems:

1. **Input dimension explosion**: obs_dim went from 44 to 268 (6x), but the policy network `[256, 256, 256, 64]` and GRU hidden_dim=32 were not scaled up. The 1-layer GRU with hidden=32 was bottlenecking 256-dim input into a 32-dim latent—massive information loss.

2. **Auxiliary loss overwhelmed policy gradient**: With `traj_weight=1.0`, the auxiliary future-trajectory prediction task dominated training. The auxiliary loss dropped successfully (3.1 → 0.17), but the policy got worse. This proves that **aux loss convergence does not imply useful representation**—the encoder can learn to predict future hand deltas without learning anything that helps the catch/hide game.

3. **32-step context was overkill for scripted hand**: A scripted hand's strategy is trivial (chase robot). 32 steps of interaction history is massively redundant for this task. The excess input dimensionality made learning harder, not easier.

## Key Lessons

1. **Auxiliary task loss dropping ≠ policy improving.** Must monitor both. If aux loss drops but reward stagnates/gets worse, the auxiliary gradient is corrupting the representation.

2. **Input dimension scaling matters.** Going from 44-dim to 268-dim requires proportional scaling of network capacity. A 32-dim GRU latent cannot compress 256-dim input effectively.

3. **Interaction history is the right direction, but 32 steps * 8 channels is too much for a first attempt.** Start with 16 steps * 8 channels = 128 dims (3x the original, not 6x).

4. **Auxiliary loss weight is critical.** PPO's policy/value gradient and auxiliary task gradient compete. If aux weight is too high, the encoder optimizes for prediction accuracy at the expense of policy utility.

## Recommended Fix (applied same day)

Applied to source code before next run:

```text
1. history_length = 16 (not 32): 16 * 8 = 128 dims, manageable
2. GRU hidden_dim = 64 (not 32): larger latent capacity
3. aux weights reduced 10x: traj=0.1, risk=0.02, min_dist=0.01
4. Keep aux task (do not remove)
```

Pending: run this configuration and compare to aux_lstm baseline and opponent-id diagnostic.
