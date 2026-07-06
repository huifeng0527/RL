---
title: Strategy GRU AUX Experiment 3 - Interaction History (16 steps)
date: 2026-06-17
related_logs:
  - logs/dual_iterative_0617_1656
---

# Experiment 3: strategy_gru_aux with 16-step interaction history + reduced aux weights

## Config (run_config.json)

```json
{
  "extractor_name": "strategy_gru_aux",
  "history_length": 16,
  "history_mode": "interaction",
  "future_horizon": 8,
  "include_opponent_id": false,
  "scripted_hand_sample_prob": null,
  "strategy_aux_weights": {
    "traj": 0.1,
    "risk": 0.02,
    "min_dist": 0.01
  }
}
```

Compared to Experiment 2 (0617_1540):
- history_length: 32 -> 16 (obs_dim 268 -> 140)
- GRU hidden_dim: 32 -> 64
- aux weights: reduced 10x (traj 1.0->0.1, risk 0.2->0.02, min_dist 0.1->0.01)

## Results

### TensorBoard

```text
iteration_1 (single scripted hand):
  eval reward tail5 = -8.13, ep_len tail5 = 73.17
  strategy_aux loss tail5 = 0.030
  -> Learning well, close to old aux_lstm baseline

iteration_2 (scripted + iteration_1 hand):
  eval reward tail5 = -2.70, ep_len tail5 = 70.78
  strategy_aux loss tail5 = 0.057
  -> Still good, strong improvement

iteration_3 (scripted + iteration_1 + iteration_2 hands):
  eval reward tail5 = -40.79, ep_len tail5 = 15.64
  strategy_aux loss tail5 = 0.062
  -> Complete collapse
```

### Per-opponent episode breakdown

```text
iteration 1: n=2039
  scripted_hand: rew=-41.4, tail_rew=-30.7, len=14.9, tail_len=44.5
  -> Single scripted hand: learning OK

iteration 2: n=474
  scripted_hand (20%): rew=-45.3, tail_rew=-45.3, len=10.8, tail_len=10.8
  iteration_1 (80%):    rew=-23.8, tail_rew=-11.6, len=52.9, tail_len=74.5
  -> Against learned hand: very strong (tail_len=74.5)
  -> Against scripted hand: weak (but only 20% of training)

iteration 3: n=1365
  scripted_hand (20%): rew=-41.8, len=8.7
  iteration_1 (17%):    rew=-25.9, len=48.3, truncated=99/232
  iteration_2 (63%):    rew=-41.9, len=7.8
  -> iteration_1 learned hand: still manageable
  -> iteration_2 learned hand: complete failure (almost all caught)
  -> scripted hand: complete failure (almost all caught)
```

## Diagnosis

1. **Iterations 1-2 are much better than previous experiments.** The combination of reduced input dimensionality (16*8=128 vs 32*8=256), larger GRU capacity (hidden=64), and reduced aux weights (10x lower) is working.

2. **Iteration 3 collapse is a multi-opponent generalization problem, not a training instability problem.** The robot learns well against scripted_hand and iteration_1 hand, but iteration_2 introduces a hand strategy that the robot cannot handle.

3. **The iteration_2 learned hand appears to have found a strategy that exploits a blind spot in the robot's interaction-history encoder.** The robot can recognize iteration_1 hand patterns (tail_len=53.5) but not iteration_2 patterns (tail_len=8.6) or even scripted_hand patterns (tail_len=8.7).

4. **Auxiliary loss dropped consistently** across all iterations (0.34->0.03, 0.52->0.06, 0.40->0.06). Unlike Experiment 2 where aux loss dropped but policy got worse, here the aux loss and policy are correlated for iterations 1-2.

## Key Insight

This experiment validates that:
- Interaction history is the right direction (iterations 1-2 work)
- The remaining problem is specifically multi-opponent strategy diversity
- This is consistent with the opponent-id diagnostic: the core issue is strategy aliasing across opponents, not insufficient training

The interaction-history encoder can handle 1-2 opponents well but struggles when a 3rd opponent with a new/different strategy enters the pool.

## Next Steps

The multi-opponent problem could be addressed by:
- Curriculum learning: slowly introduce new opponents
- Larger GRU or multi-layer GRU for richer latent representation
- Contrastive loss to force different opponent embeddings apart
- Or simply: the opponent-id experiment already proved the hypothesis; interaction history alone may not be sufficient for 3+ diverse opponents
