# Memory Index

This directory contains experimental observations and analysis notes for the rehabilitation RL project. Each file records a specific experiment or finding.

## Files

### opponent_id_league_observation.md
**Date**: 2026-06-16
**Logs**: logs/dual_iterative_0616_1612 (no-id), logs/dual_iterative_0616_2246 (with-id)
**Summary**: Diagnosis experiment comparing multi-opponent pool training with and without explicit opponent-id one-hot. Confirmed that pool training failure is primarily caused by hidden opponent identity / strategy aliasing, not by PPO instability or auxiliary task issues. Adding opponent id stabilized training and restored performance across iterations.

### strategy_gru_aux_training_observations.md
**Date**: 2026-06-17
**Logs**: logs/dual_iterative_0617_1152, logs/dual_iterative_0617_1540
**Summary**: Two failed attempts at a non-ID alternative using GRU strategy encoder with auxiliary prediction heads. Experiment 1 used 16-step hand movement history (too weak, same as old LSTM). Experiment 2 used 32-step interaction history but obs_dim exploded to 268, GRU hidden_dim was too small (32), and aux weights were too high (1.0), causing policy collapse. Key lesson: aux loss dropping does not mean the policy is improving.

### strategy_gru_aux_experiment3_16step_interaction.md
**Date**: 2026-06-17
**Log**: logs/dual_iterative_0617_1656
**Summary**: Third attempt with corrected parameters: 16-step interaction history (128 dims), GRU hidden_dim=64, aux weights reduced 10x. Iterations 1-2 showed strong improvement (close to old aux_lstm baseline), but iteration 3 collapsed when a third opponent with a different strategy entered the pool. Validates that interaction history is the right direction, but the remaining problem is multi-opponent strategy generalization.

### league_ablation_gru_contrastive_plan_10iter.md
**Date**: 2026-06-19
**Logs**: logs/league_ablation_10iter/A_mlp, logs/league_ablation_10iter/B_gru, planned C-F groups
**Summary**: Final A-F 10-iteration league ablation plan. Main narrative drops LSTM from the core story and compares MLP, GRU interaction history, single-step aux, multi-step/risk aux, contrastive strategy loss, and opponent-ID upper bound. Also records two-at-a-time execution schedule and interpretation logic.
